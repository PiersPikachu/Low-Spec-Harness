#!/usr/bin/env python3
"""
lb_probe.py - what does sub-1-bit compression do to a real model WITHOUT training?

Replaces every transformer-block matrix of Qwen3-0.6B (q/k/v/o/gate/up/down; embedding/lm_head and norms kept exact, as in the papers)
with an approximation, then measures on the same text against the exact (lossless-vault) weights:
  perplexity, top-1 next-token agreement, and per-matrix relative reconstruction error.
Approximations:
  rtn4-g128 / rtn2-g128   plain round-to-nearest int4 / int2, group 128 (4.25 / 2.25 bpp) - the usual "4-bit" reference point
  lb2@B                   LittleBit-2 initialisation (truncated SVD -> Joint-ITQ rotation -> sign -> rank-1 tri-scales, primary + residual path)
                          at B bits per weight, NO quantization-aware training (the papers need ~1B tokens of QAT on 4+ H100s)
usage: lb_probe.py [--tokens 512] [--configs rtn4,rtn2,lb2@1.0,lb2@0.55,lb2@0.3,lb2@0.1]
"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import pack_universal_v3 as pu

MODEL = Path(r"C:\Users\1fred\Desktop\AI\JEPA\_stage_memory\hf_model\Qwen_Qwen3-0.6B")
MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def f32(T, n):
    t = T[n]; return pu.to_f32(pu.load_raw(t), t["dtype"]).reshape(t["shape"] or ())


# ---------------------------------------------------------------- approximations
def rtn(W, bits, g=128):
    o, i = W.shape; X = W.reshape(o, i // g, g); mx = np.abs(X).max(-1, keepdims=True); q = 2 ** (bits - 1) - 1
    s = np.where(mx > 0, mx / q, 1.0); return (np.clip(np.rint(X / s), -q - 1, q) * s).reshape(o, i).astype(np.float32)


def rank1(X):
    U, S, Vt = torch.linalg.svd(X, full_matrices=False); s = torch.sqrt(S[0]); return U[:, :1] * s, Vt[:1, :] * s


def lb2_path(W, r, iters=50, rng=None):
    """torch (CPU, multithreaded) is ~5-10x faster than numpy's LAPACK here; same algorithm"""
    Wt = torch.from_numpy(W); U, S, Vt = torch.linalg.svd(Wt, full_matrices=False); s = torch.sqrt(S[:r])
    Uf = U[:, :r] * s; Vf = Vt[:r, :].T * s                          # (dout, r), (din, r)
    Z = torch.cat([Uf, Vf]); R = torch.linalg.qr(torch.from_numpy(rng.standard_normal((r, r)).astype(np.float32)))[0]
    for _ in range(iters):                                            # Joint-ITQ: align latent rows with the binary hypercube
        B = torch.sign(Z @ R); B[B == 0] = 1; P, _, Qt = torch.linalg.svd(B.T @ Z); R = (P @ Qt).T
    Ua, Va = Uf @ R, Vf @ R
    h, lu = rank1(Ua.abs()); g, lv = rank1(Va.abs())                 # |U| ~ h lu^T, |V| ~ g lv^T
    l = (lu * lv).ravel(); Ub = torch.sign(Ua); Ub[Ub == 0] = 1; Vb = torch.sign(Va); Vb[Vb == 0] = 1
    return (((h * Ub) * l) @ (Vb * g).T).numpy()                     # diag(h) Ub diag(l) Vb^T diag(g)


def lb2(W, bpp, rng):
    o, i = W.shape; r = int((bpp * o * i - 32 * (o + i)) // (2 * (o + i + 16)))
    if r < 1: return np.zeros_like(W), 0
    Wp = lb2_path(W, r, rng=rng); Wr = lb2_path(W - Wp, r, rng=rng)
    return (Wp + Wr).astype(np.float32), r


# ---------------------------------------------------------------- reference forward (all positions)
def rms(x, w, eps):
    return (w * (x / np.sqrt(np.mean(x.astype(np.float64) ** 2, -1, keepdims=True) + eps).astype(np.float32))).astype(np.float32)


def forward_logits(T, cfg, ids, override):
    nl, hid, nh, nkv, hd, eps = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    n = len(ids); half = hd // 2; fr = cfg["rope_theta"] ** (-2.0 * np.arange(half) / hd); a = np.arange(n)[:, None] * fr[None]
    cs, sn = np.cos(a).astype(np.float32)[:, None], np.sin(a).astype(np.float32)[:, None]
    rope = lambda x: np.concatenate([x[..., :half] * cs - x[..., half:] * sn, x[..., half:] * cs + x[..., :half] * sn], -1)
    E = f32(T, "model.embed_tokens.weight"); H = E[np.array(ids)].copy(); mask = np.triu(np.full((n, n), -np.inf, np.float32), 1)
    for l in range(nl):
        g = lambda s: override.get(f"model.layers.{l}.{s}.weight") if f"model.layers.{l}.{s}.weight" in override else f32(T, f"model.layers.{l}.{s}.weight")
        x = rms(H, g("input_layernorm"), eps)
        q = (x @ g("self_attn.q_proj").T).reshape(n, nh, hd); k = (x @ g("self_attn.k_proj").T).reshape(n, nkv, hd); v = (x @ g("self_attn.v_proj").T).reshape(n, nkv, hd)
        q = rope(rms(q, g("self_attn.q_norm"), eps)); k = rope(rms(k, g("self_attn.k_norm"), eps))
        K = np.repeat(k, nh // nkv, 1); V = np.repeat(v, nh // nkv, 1)
        sc = np.einsum("qhd,khd->hqk", q, K) / np.float32(math.sqrt(hd)) + mask; sc -= sc.max(-1, keepdims=True); p = np.exp(sc); p /= p.sum(-1, keepdims=True)
        H = H + np.einsum("hqk,khd->qhd", p, V).reshape(n, nh * hd) @ g("self_attn.o_proj").T
        x = rms(H, g("post_attention_layernorm"), eps); ga = x @ g("mlp.gate_proj").T
        H = H + ((ga / (1 + np.exp(-np.clip(ga, -80, 80)))) * (x @ g("mlp.up_proj").T)) @ g("mlp.down_proj").T
    return rms(H, f32(T, "model.norm.weight"), eps) @ E.T


def score(logits, ids):
    lg = logits[:-1].astype(np.float64); m = lg.max(-1, keepdims=True); lse = (m[:, 0] + np.log(np.exp(lg - m).sum(-1)))
    nll = lse - lg[np.arange(len(ids) - 1), ids[1:]]; return float(np.exp(nll.mean())), logits[:-1].argmax(-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tokens", type=int, default=512); ap.add_argument("--configs", default="rtn4,rtn2,lb2@1.0,lb2@0.55,lb2@0.3,lb2@0.1")
    a = ap.parse_args(); from transformers import AutoTokenizer
    cfg = json.load(open(MODEL / "config.json")); T = pu.scan_model(MODEL); tok = AutoTokenizer.from_pretrained(str(MODEL))
    text = open(MODEL / "README.md", encoding="utf-8").read(); ids = tok(text)["input_ids"][:a.tokens]
    out = HERE / "lb_probe_results.json"; res = json.load(open(out)) if out.exists() else {}
    t0 = time.time(); base = forward_logits(T, cfg, ids, {}); ppl0, top0 = score(base, ids)
    res["exact (lossless vault, ~10.6 bpp)"] = dict(bpp=16 * 0.6646, ppl=ppl0, top1_agree=1.0); print(f"exact: ppl {ppl0:.3f} ({time.time() - t0:.0f}s)", flush=True)
    nparam = sum(int(np.prod(T[f"model.layers.{l}.{m}.weight"]["shape"])) for l in range(cfg["num_hidden_layers"]) for m in MATS)
    for c in a.configs.split(","):
        t0 = time.time(); rng = np.random.default_rng(0); ov = {}; errs = []; bits = 0.0
        for l in range(cfg["num_hidden_layers"]):
            for m in MATS:
                n = f"model.layers.{l}.{m}.weight"; W = f32(T, n)
                if c.startswith("rtn"): b = int(c[3]); Wa = rtn(W, b); bits += W.size * (b + 32 / 128)
                else:
                    B = float(c.split("@")[1]); Wa, r = lb2(W, B, rng); o, i = W.shape; bits += 2 * r * (o + i + 16) + 32 * (o + i)
                ov[n] = Wa; errs.append(float(np.linalg.norm(W - Wa) / np.linalg.norm(W)))
        lg = forward_logits(T, cfg, ids, ov); ppl, top = score(lg, ids)
        res[c] = dict(bpp=bits / nparam, ppl=ppl, top1_agree=float((top == top0).mean()), rel_err_mean=float(np.mean(errs)), rel_err_max=float(np.max(errs)), secs=time.time() - t0)
        print(f"{c:10s} body bpp {bits / nparam:6.3f} | ppl {ppl:10.3f} (exact {ppl0:.3f}) | top-1 agrees with exact {100 * res[c]['top1_agree']:5.1f}% | "
              f"matrix rel. error mean {np.mean(errs):.3f} max {np.max(errs):.3f} | {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
