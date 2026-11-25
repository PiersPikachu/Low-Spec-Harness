#!/usr/bin/env python3
"""
lb_train.py - can LittleBit-2 be repaired on THIS laptop? Block-wise reconstruction of a sub-1-bit Qwen3-0.6B.

E3 showed the LittleBit-2 initialisation alone collapses (PPL 6.5e7 at 1.0 bpp). The papers fix it with end-to-end distillation on ~1B
tokens on H100s. Here instead, following EfficientQAT's Block-AP (Chen et al., ACL 2025): train ONE transformer block at a time to
reproduce the original block's output, so memory never exceeds one block, and calibrate on text the model generates itself
(LLM-QAT, Liu et al. 2024) so no dataset is needed.
  student block l  : LittleBit-2 linears (primary + residual path, Joint-ITQ init, SmoothSign gradient), norms exact
  input            : the student's own stream (errors of earlier blocks included)
  target           : the original block applied to the original stream
Evaluation: perplexity and top-1 agreement with the exact model on the same 512 README tokens as E3 (lb_probe.py).
  lb_train.py [--bpp 1.0] [--seqs 64] [--len 512] [--steps 300] [--lr 5e-4] [--bs 4] [--layers N (debug)]
"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import argparse, json, math, time
from pathlib import Path
import torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
MODEL = Path(r"C:\Users\1fred\Desktop\AI\JEPA\_stage_memory\hf_model\Qwen_Qwen3-0.6B")
MATS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
        "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
dev = "cuda"


class SmoothSign(torch.autograd.Function):            # LittleBit: forward sign, backward d/dx tanh(100 x)
    @staticmethod
    def forward(ctx, x): ctx.save_for_backward(x); return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)
    @staticmethod
    def backward(ctx, g): (x,) = ctx.saved_tensors; t = torch.tanh(100 * x); return g * 100 * (1 - t * t)


def rank1(X):
    U, S, Vt = torch.linalg.svd(X, full_matrices=False); s = torch.sqrt(S[0]); return U[:, 0] * s, Vt[0] * s


def lb2_init(W, r, gen, iters=50):
    """LittleBit-2 init of one path: truncated SVD -> Joint-ITQ rotation -> latent factors + rank-1 tri-scales"""
    U, S, Vt = torch.linalg.svd(W, full_matrices=False); s = torch.sqrt(S[:r]); Uf = U[:, :r] * s; Vf = Vt[:r].T * s
    Z = torch.cat([Uf, Vf]); R = torch.linalg.qr(torch.randn(r, r, generator=gen, device=W.device))[0]
    for _ in range(iters):
        B = torch.where(Z @ R >= 0, 1.0, -1.0); P, _, Qt = torch.linalg.svd(B.T @ Z); R = (P @ Qt).T
    Ua, Va = Uf @ R, Vf @ R; h, lu = rank1(Ua.abs()); g, lv = rank1(Va.abs())
    return Ua, Va, h, g, lu * lv


class LBLinear(torch.nn.Module):
    def __init__(self, W, bpp, gen):
        super().__init__(); o, i = W.shape; self.r = r = max(1, int((bpp * o * i - 32 * (o + i)) // (2 * (o + i + 16))))
        self.bits = 2 * r * (o + i + 16) + 32 * (o + i); self.paths = torch.nn.ModuleList()
        R = W
        for _ in range(2):                                    # primary, then residual on what the primary misses
            Ua, Va, h, g, l = lb2_init(R, r, gen); p = torch.nn.ParameterDict(dict(U=Ua, V=Va, h=h, g=g, l=l))
            self.paths.append(p); R = R - self.weight_of(p)
    @staticmethod
    def weight_of(p):
        with torch.no_grad(): return ((p["h"][:, None] * torch.sign(p["U"])) * p["l"]) @ (torch.sign(p["V"]) * p["g"][:, None]).T
    def forward(self, x):
        y = 0
        for p in self.paths:
            y = y + (((x * p["g"]) @ SmoothSign.apply(p["V"])) * p["l"]) @ SmoothSign.apply(p["U"]).T * p["h"]
        return y


def rms(x, w, eps): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def block(x, P, lin, cfg, cos, sin):
    B, T, _ = x.shape; nh, nkv, hd, eps = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    h = rms(x, P["input_layernorm"], eps)
    q = lin["q"](h).view(B, T, nh, hd); k = lin["k"](h).view(B, T, nkv, hd); v = lin["v"](h).view(B, T, nkv, hd)
    q = rms(q, P["self_attn.q_norm"], eps); k = rms(k, P["self_attn.k_norm"], eps)
    def rope(t): a, b = t[..., :hd // 2], t[..., hd // 2:]; return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)
    q, k = rope(q).transpose(1, 2), rope(k).transpose(1, 2); v = v.transpose(1, 2)
    att = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).transpose(1, 2).reshape(B, T, nh * hd)
    x = x + lin["o"](att); h = rms(x, P["post_attention_layernorm"], eps)
    return x + lin["down"](F.silu(lin["gate"](h)) * lin["up"](h))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bpp", type=float, default=1.0); ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--len", type=int, default=512); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--layers", type=int, default=0); a = ap.parse_args()
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cfg = json.load(open(MODEL / "config.json"))
    tok = AutoTokenizer.from_pretrained(str(MODEL)); nl = a.layers or cfg["num_hidden_layers"]; eps = cfg["rms_norm_eps"]
    # ---- calibration text generated by the model itself (LLM-QAT): random first token, sampled continuation
    cal_path = HERE / f"lb_cal_{a.seqs}x{a.len}.pt"
    if cal_path.exists(): cal = torch.load(cal_path).to(dev)
    else:
        m = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16).to(dev).eval(); g = torch.Generator().manual_seed(0); rows = []
        for i in range(0, a.seqs, 16):
            start = torch.randint(0, 150000, (16, 1), generator=g).to(dev)
            with torch.no_grad(): out = m.generate(start, max_new_tokens=a.len - 1, min_new_tokens=a.len - 1, do_sample=True, top_k=50, top_p=0.95, temperature=1.0, pad_token_id=0)
            rows.append(out[:, :a.len])
        cal = torch.cat(rows); torch.save(cal.cpu(), cal_path); del m; torch.cuda.empty_cache(); print(f"calibration: {tuple(cal.shape)} tokens generated", flush=True)
    Wt = {k: v.float() for k, v in load_file(str(MODEL / "model.safetensors")).items()}
    ev = torch.tensor(tok(open(MODEL / "README.md", encoding="utf-8").read())["input_ids"][:512], device=dev)[None]
    hd = cfg["head_dim"]; T = a.len; fr = cfg["rope_theta"] ** (-torch.arange(0, hd // 2, device=dev).float() * 2 / hd)
    ang = torch.arange(max(T, 512), device=dev).float()[:, None] * fr[None]; COS, SIN = ang.cos()[:, None], ang.sin()[:, None]
    Pc = {l: {k[len(f"model.layers.{l}."):-len(".weight")]: v.to(dev) for k, v in Wt.items() if k.startswith(f"model.layers.{l}.") and "norm" in k} for l in range(cfg["num_hidden_layers"])}
    P = lambda l: Pc[l]
    def exact(l):
        Ws = {n: Wt[f"model.layers.{l}.{m}.weight"].to(dev) for n, m in MATS.items()}
        return {n: (lambda x, W=Ws[n]: x @ W.T) for n in MATS}
    E = Wt["model.embed_tokens.weight"].to(dev); fnorm = Wt["model.norm.weight"].to(dev)
    Ht = E[cal]; Hs = Ht.clone(); students = []; gen = torch.Generator(device=dev).manual_seed(0); bits = params = 0; t0 = time.time()
    for l in range(nl):
        with torch.no_grad():                                                   # target: exact block on the exact stream
            ex = exact(l); Yt = torch.cat([block(Ht[i:i + 8], P(l), ex, cfg, COS[:T], SIN[:T]) for i in range(0, len(Ht), 8)]); del ex
        lin = {n: LBLinear(Wt[f"model.layers.{l}.{m}.weight"].to(dev), a.bpp, gen).to(dev) for n, m in MATS.items()}
        bits += sum(x.bits for x in lin.values()); params += sum(Wt[f"model.layers.{l}.{m}.weight"].numel() for m in MATS.values())
        mods = torch.nn.ModuleDict(lin); opt = torch.optim.Adam(mods.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps); ns = len(Hs); scale = Yt.pow(2).mean()
        def loss_on(idx): return F.mse_loss(block(Hs[idx], P(l), lin, cfg, COS[:T], SIN[:T]), Yt[idx]) / scale
        with torch.no_grad(): l0 = sum(loss_on(torch.arange(i, min(i + 8, ns), device=dev)).item() for i in range(0, ns, 8)) / math.ceil(ns / 8)
        for s in range(a.steps):
            idx = torch.randint(0, ns, (a.bs,), device=dev); loss = loss_on(idx); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            l1 = sum(loss_on(torch.arange(i, min(i + 8, ns), device=dev)).item() for i in range(0, ns, 8)) / math.ceil(ns / 8)
            Hs = torch.cat([block(Hs[i:i + 8], P(l), lin, cfg, COS[:T], SIN[:T]) for i in range(0, ns, 8)]); Ht = Yt
        for p in mods.parameters(): p.requires_grad_(False)
        mods.to("cpu"); students.append(lin); del opt, sched, Yt; torch.cuda.empty_cache()
        print(f"block {l:2d}: relative MSE vs exact block {l0:.4f} (init) -> {l1:.4f} (trained)  [{time.time() - t0:.0f}s]", flush=True)
    # ---- evaluation on the README tokens (same text as E3)
    def run(mk):
        with torch.no_grad():
            x = E[ev]; n = ev.shape[1]
            for l in range(cfg["num_hidden_layers"]):
                if mk is exact or l >= nl: x = block(x, P(l), exact(l), cfg, COS[:n], SIN[:n])
                else:
                    for m in students[l].values(): m.to(dev)
                    x = block(x, P(l), students[l], cfg, COS[:n], SIN[:n])
                    for m in students[l].values(): m.to("cpu")
            lg = rms(x, fnorm, eps) @ E.T
        nll = F.cross_entropy(lg[0, :-1].double(), ev[0, 1:]); return math.exp(nll.item()), lg[0, :-1].argmax(-1)
    p0, a0 = run(exact); p1, a1 = run(None)
    res = dict(bpp_body=bits / params, layers_replaced=nl, seqs=a.seqs, len=a.len, steps=a.steps, lr=a.lr, ppl_exact=p0, ppl_student=p1,
               top1_agree=(a0 == a1).float().mean().item(), minutes=(time.time() - t0) / 60)
    print(json.dumps(res)); out = HERE / "lb_train_results.json"; hist = json.load(open(out)) if out.exists() else []; hist.append(res); json.dump(hist, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
