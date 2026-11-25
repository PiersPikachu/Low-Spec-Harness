#!/usr/bin/env python3
"""mxfp4_entropy.py <model_dir> - can the released MXFP4 expert weights (gpt-oss) be stored losslessly in fewer than 4.25 bits?

MXFP4 = blocks of 32 FP4 (E2M1) codes, two per byte (*_blocks, uint8), plus one E8M0 scale byte per block (*_scales).
Stored size = 4 + 8/32 = 4.25 bits per weight. This measures, on the real tensors, the order-0 entropy of
  the 4-bit codes (per tensor and per expert), the codes conditioned on their block's scale, the scale bytes,
and the byte-pair (two codes) entropy - i.e. the size an entropy coder (Huffman/rANS, as in our vaults) would reach.
"""
import json, math, sys
from pathlib import Path
import numpy as np
from safetensors import safe_open


def H(counts):
    c = counts[counts > 0].astype(np.float64); p = c / c.sum(); return float(-(p * np.log2(p)).sum())


def main():
    d = Path(sys.argv[1]); idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
    names = sorted(k for k in idx if k.endswith("_blocks"))
    tot = dict(w=0, code_bits=0.0, code_bits_e=0.0, code_bits_cs=0.0, scale_bits=0.0, pair_bits=0.0)
    for n in names:
        with safe_open(str(d / idx[n]), "np") as f:
            B = f.get_tensor(n); S = f.get_tensor(n.replace("_blocks", "_scales"))
        E = B.shape[0]; lo = (B & 15).astype(np.uint8); hi = (B >> 4).astype(np.uint8)
        codes = np.stack([lo, hi], -1).reshape(E, -1); nw = codes.size
        h_all = H(np.bincount(codes.ravel(), minlength=16))
        h_e = sum(H(np.bincount(codes[e], minlength=16)) * codes[e].size for e in range(E)) / nw        # per-expert tables
        # codes conditioned on their block scale (scale value picks the table)
        sc = np.repeat(S.reshape(E, -1), 32, axis=1)                                                     # one scale per 32 codes
        joint = np.bincount(sc.ravel().astype(np.int64) * 16 + codes.ravel(), minlength=256 * 16).reshape(256, 16)
        h_cs = sum(H(joint[s]) * joint[s].sum() for s in range(256) if joint[s].sum()) / nw
        h_s = H(np.bincount(S.ravel(), minlength=256)); h_pair = H(np.bincount(B.ravel(), minlength=256)) / 2
        tot["w"] += nw; tot["code_bits"] += h_all * nw; tot["code_bits_e"] += h_e * nw; tot["code_bits_cs"] += h_cs * nw
        tot["scale_bits"] += h_s * S.size; tot["pair_bits"] += h_pair * nw
        print(f"{n:48s} codes H={h_all:.3f} b (per-expert {h_e:.3f}, given scale {h_cs:.3f}, pairs {h_pair:.3f}/code)  scales H={h_s:.2f} of 8 b", flush=True)
    w = tot["w"]; s = tot["scale_bits"] / w
    print(f"\n{w / 1e9:.2f} B MXFP4 weights. Stored: 4.250 bits/weight.")
    for label, key in (("one table per tensor", "code_bits"), ("one table per expert", "code_bits_e"), ("table chosen by block scale", "code_bits_cs"), ("byte-pair table", "pair_bits")):
        b = tot[key] / w + s; print(f"  entropy-coded, {label:28s}: {b:.3f} bits/weight  ({100 * b / 4.25:.1f}% of stored)")


if __name__ == "__main__":
    main()
