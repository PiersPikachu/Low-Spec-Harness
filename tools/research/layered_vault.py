#!/usr/bin/env python3
"""layered_vault.py <hf_model_dir> [--layers-sample N] - is a 'layered' lossless vault worth building?

Layered vault = every weight stored as a 4-bit MXFP4 approximation (E2M1 code, one E8M0 scale per 32 weights, round-to-nearest)
plus the exact remainder needed to restore the original bf16 bits. The 4-bit layer alone is a draft model for speculative
decoding, so outputs stay identical to the original. Two unknowns, measured here:

  size        bits/weight of (4-bit layer, entropy-coded) + (original bits given the 4-bit code and scale), against today's
              lossless vault (order-0 entropy of exponent + raw sign/mantissa, ~10.6 bits). Context-conditioned order-0 entropies:
              what a Huffman/rANS coder with one table per context reaches.
  acceptance  greedy: the original model generates 96 tokens for 6 prompts; the 4-bit copy is run over the same text and its
              top token compared at every position. Tokens per verification pass for k drafted tokens are counted by walking the
              agreement sequence exactly as speculative decoding would. Also a 3-bit variant, and (for a larger target) the
              separate Qwen3-0.6B draft used today, for comparison.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
PROMPTS = ["Explain in three sentences how a refrigerator keeps food cold.", "Write a Python function that returns the n-th Fibonacci number iteratively.",
           "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Show the steps.", "List five tips for writing clear technical documentation.",
           "Write the opening two sentences of a short story set on a lighthouse during a storm.", "In one sentence, what is lossless compression?"]
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def H(counts):
    c = counts[counts > 0].astype(np.float64); p = c / c.sum(); return float(-(p * np.log2(p)).sum())


def cond_H(ctx, sym):
    """H(sym | ctx) for integer arrays, in bits per element"""
    key = ctx.astype(np.int64) * 65536 + sym.astype(np.int64)
    _, joint = np.unique(key, return_counts=True); _, marg = np.unique(ctx, return_counts=True)
    n = len(sym); return (H(joint) * 1.0 - H(marg)) if n else 0.0          # H(X,C) - H(C)


def quantize(w, bits=4):
    """MX-style: blocks of 32 along the input dimension, E8M0 power-of-two scale; FP4 E2M1 grid (bits=4) or int levels -3..3 (bits=3)"""
    rows, cols = w.shape; b = w.float().reshape(rows, cols // 32, 32); amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    if bits == 4:
        e = torch.floor(torch.log2(amax)) - 2; s = torch.pow(2.0, e)                # max -> [4, 8): grid top is 6, so clamp
        x = (b / s).clamp(-6, 6); g = FP4.to(w.device); idx = (x.abs().unsqueeze(-1) - g).abs().argmin(-1)
        code = idx + 8 * (x < 0).long(); q = g[idx] * torch.sign(x) * s
    else:
        e = torch.ceil(torch.log2(amax / 3.0)); s = torch.pow(2.0, e); x = torch.round(b / s).clamp(-3, 3)
        code = (x + 3).long(); q = x * s
    return q.reshape(rows, cols).to(w.dtype), code.reshape(rows, cols), e.reshape(rows, cols // 32)


def size_bits(w, code, e):
    """(4-bit layer bits/weight, residual bits/weight, today's lossless bits/weight) for one matrix"""
    u = w.contiguous().view(torch.int16).cpu().numpy().astype(np.int64) & 0xFFFF
    expo = (u >> 7) & 0xFF; sign = u >> 15; mant = u & 0x7F
    c = code.cpu().numpy().astype(np.int64); ee = e.cpu().numpy().astype(np.int64); ee = np.repeat(ee, 32, axis=1)
    ctx = (c * 512 + (ee + 256)).ravel()                                            # the 4-bit code and its block scale
    layer = H(np.bincount(c.ravel())) + H(np.unique(e.cpu().numpy().ravel(), return_counts=True)[1]) / 32
    hi = (sign * 256 + expo).ravel(); res = cond_H(ctx, hi) + cond_H(ctx * 512 + hi, mant.ravel())
    today = H(np.bincount(expo.ravel(), minlength=256)) + 8.0
    return layer, res, today


def tokens_per_pass(agree, k):
    """walk the greedy agreement sequence: a pass accepts drafted tokens until the first miss, plus the target's own token"""
    i, passes, n = 0, 0, len(agree)
    while i < n:
        j = 0
        while j < k and i + j < n and agree[i + j]: j += 1
        i += j + 1; passes += 1
    return n / passes


def main():
    src = Path(sys.argv[1]); sample = int(sys.argv[sys.argv.index("--layers-sample") + 1]) if "--layers-sample" in sys.argv else 1
    big = "4B" in src.name or "4b" in src.name
    tok = AutoTokenizer.from_pretrained(src); t0 = time.time()
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if big: kw.update(device_map="auto", max_memory={0: "4800MiB", "cpu": "9GiB"})
    m = AutoModelForCausalLM.from_pretrained(src, **kw).eval()
    if not big and torch.cuda.is_available(): m = m.cuda()
    dev = next(m.parameters()).device; print(f"{src.name}: loaded in {time.time() - t0:.0f}s on {dev}", flush=True)
    seqs = []
    for p in PROMPTS:                                        # the original model's own greedy continuations
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", enable_thinking=False)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        with torch.no_grad(): out = m.generate(ids.to(dev), max_new_tokens=96, do_sample=False)
        seqs.append((ids.shape[1], out[0].cpu()))
    print(f"generated {sum(len(s) - n for n, s in seqs)} target tokens in {time.time() - t0:.0f}s", flush=True)
    def agreement(model):
        ag = []
        for n, s in seqs:
            with torch.no_grad(): lg = model(s.unsqueeze(0).to(next(model.parameters()).device)).logits[0]
            pred = lg[n - 1:len(s) - 1].argmax(-1).cpu(); ag += (pred == s[n:]).tolist()
        return ag
    res = {"model": src.name}
    if big:                                                  # today's separate draft, for comparison
        dsrc = src.parent / "Qwen_Qwen3-0.6B"
        if dsrc.exists():
            d = AutoModelForCausalLM.from_pretrained(dsrc, dtype=torch.bfloat16).eval()
            if torch.cuda.is_available() and not big: d = d.cuda()
            a = agreement(d); res["draft_0.6B"] = dict(agree=np.mean(a), tpp4=tokens_per_pass(a, 4), tpp8=tokens_per_pass(a, 8)); del d
            print("separate 0.6B draft:", res["draft_0.6B"], flush=True)
    if big:     # offloaded weights live in accelerate's weights_map (CPU tensors): quantize those in place, nothing written to disk
        sz = []
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in LINEARS:
                hook = getattr(mod, "_hf_hook", None); t = mod.weight.data
                if t.device.type == "meta": t = hook.weights_map["weight"]
                q, code, e = quantize(t, 4); layer_no = int(name.split(".layers.")[1].split(".")[0])
                if layer_no % sample == 0: sz.append((t.numel(),) + size_bits(t, code, e))
                t.copy_(q.to(t.device)); del q, code, e
        a = agreement(m)
        n = sum(x[0] for x in sz); L = sum(x[0] * x[1] for x in sz) / n; R = sum(x[0] * x[2] for x in sz) / n; T_ = sum(x[0] * x[3] for x in sz) / n
        r = dict(agree=float(np.mean(a)), tpp4=tokens_per_pass(a, 4), tpp8=tokens_per_pass(a, 8), layer_bits=L, residual_bits=R, layered_total=L + R,
                 today_lossless=T_, overhead=(L + R) / T_ - 1)
        res["mx4"] = r; print("4-bit copy:", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        out = Path(__file__).with_name("layered_vault.json"); allr = json.load(open(out)) if out.exists() else {}
        allr[src.name] = res; json.dump(allr, open(out, "w"), indent=1, default=float); print(f"done in {time.time() - t0:.0f}s"); return
    # quantize in place (originals are not needed afterwards except for the size measurement, done on the way)
    orig = {}
    for bits in ((4,) if big else (4, 3)):           # one pass on the big model: no RAM for a second copy of the originals
        sz = []; li = 0
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in LINEARS:
                if not big and name not in orig: orig[name] = mod.weight.data.to("cpu", copy=True)
                w0 = orig[name].to(mod.weight.device) if not big else mod.weight.data; q, code, e = quantize(w0, bits)
                layer_no = int(name.split(".layers.")[1].split(".")[0])
                if bits == 4 and layer_no % sample == 0: sz.append((w0.numel(),) + size_bits(w0, code, e))
                mod.weight.data.copy_(q); continue
                layer_no = int(name.split(".layers.")[1].split(".")[0])
                if bits == 4 and layer_no % sample == 0: sz.append((w0.numel(),) + size_bits(w0, code, e))
        a = agreement(m); r = dict(agree=float(np.mean(a)), tpp4=tokens_per_pass(a, 4), tpp8=tokens_per_pass(a, 8))
        if bits == 4:
            n = sum(x[0] for x in sz); L = sum(x[0] * x[1] for x in sz) / n; R = sum(x[0] * x[2] for x in sz) / n; T = sum(x[0] * x[3] for x in sz) / n
            r.update(layer_bits=L, residual_bits=R, layered_total=L + R, today_lossless=T, overhead=(L + R) / T - 1)
        res[f"mx{bits}"] = r; print(f"{bits}-bit copy:", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    out = Path(__file__).with_name("layered_vault.json"); allr = json.load(open(out)) if out.exists() else {}
    allr[src.name] = res; json.dump(allr, open(out, "w"), indent=1, default=float); print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
