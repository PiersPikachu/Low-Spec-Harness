#!/usr/bin/env python3
"""scale_bound.py - what token rates are physically possible on THIS laptop, model by model, before building anything.

Per token the engine must touch every active weight once. Weights that do not fit the fast tiers (VRAM + pinned RAM) come from
the NVMe SSD, so for disk-bound models   s/token = streamed bytes / SSD bandwidth.
Calibration (measured 2026-11-23): Hermes-4-14B huffman4, 18.6 GB/pass, 6.4 GB pinned, 4.8 s/pass -> 12.2 GB / 4.8 s = 2.5 GB/s.
MoE: only the active experts are read; a RAM cache of experts saves reads in proportion to routing locality. Locality model from
arXiv 2608.18261 (Qwen3-30B-A3B: 95% of routed traffic goes through 52.5% of experts); hit(c) is an ASSUMPTION until measured here.
Speculative decoding: dense models get T tokens per pass (3.5 measured, k=8); for MoE it is not applied here (each extra token
routes to other experts, so a pass costs more) - that gain must be measured.
"""
import math

SSD = 2.5e9          # B/s, measured on the contiguous 14B vault
FAST = 5.0e9 + 7.0e9  # B usable: ~5 GB VRAM for weights + ~7 GB pinned RAM (16 GB machine)
CPU_DEC = 14.5e9     # B/s compressed-weight decode, CPU, huffman4 (measured)
SPEC_DENSE = 3.5     # tokens per pass, dense, speculative k=8 (measured mean over 6 prompts)
FMT = {"lossless bf16 (10.6 b)": 10.6, "native MXFP4 / 4-bit (4.5 b)": 4.5, "2-bit (2.6 b)": 2.6}


def hit(c):
    """fraction of expert traffic served from a cache holding fraction c of all experts (assumed concave fit, see doc)"""
    if c <= 0: return 0.0
    if c >= 1: return 1.0
    return 0.95 * (c / 0.525) ** 0.5 if c < 0.525 else 0.95 + 0.05 * (c - 0.525) / 0.475


def dense(P, b):
    total = P * b / 8; streamed = max(0.0, total - FAST)
    t_pass = max(streamed / SSD, total / CPU_DEC)               # disk-bound or decode-bound
    return total, t_pass, SPEC_DENSE / t_pass


def moe(P_total, P_active, P_shared, b):
    """P_shared = always-active part (attention, embeddings, shared experts); experts = the rest"""
    total = P_total * b / 8; shared = P_shared * b / 8; experts_total = total - shared
    per_tok_expert = (P_active - P_shared) * b / 8
    cache_for_experts = max(0.0, FAST - shared); c = min(1.0, cache_for_experts / experts_total) if experts_total > 0 else 1.0
    streamed = (max(0.0, shared - FAST)) + per_tok_expert * (1 - hit(c))
    t = max(streamed / SSD, (shared + per_tok_expert) / CPU_DEC)
    return total, t, 1 / t, c, hit(c)


MODELS = [  # name, kind, total params, active params, always-active params
    ("Hermes-4-14B (dense, measured)", "dense", 14.8e9, None, None),
    ("25B dense", "dense", 25e9, None, None), ("32B dense", "dense", 32e9, None, None),
    ("50B dense", "dense", 50e9, None, None), ("70B dense", "dense", 70e9, None, None), ("100B dense", "dense", 100e9, None, None),
    ("gpt-oss-20b (MoE 20.9B, 3.6B active)", "moe", 20.9e9, 3.61e9, 1.8e9),
    ("Qwen3-30B-A3B (MoE 30.5B, 3.3B active)", "moe", 30.5e9, 3.3e9, 1.5e9),
    ("gpt-oss-120b (MoE 116.8B, 5.1B active)", "moe", 116.8e9, 5.13e9, 2.1e9),
    ("Qwen3-235B-A22B (MoE)", "moe", 235e9, 22e9, 8e9),
    ("DeepSeek-V3 671B-A37B (MoE)", "moe", 671e9, 37e9, 15e9),
    ("1T-A32B MoE (Kimi-K2 class)", "moe", 1000e9, 32e9, 12e9),
    ("2T-A60B MoE (hypothetical)", "moe", 2000e9, 60e9, 20e9),
]

if __name__ == "__main__":
    print(f"machine: SSD {SSD / 1e9:.1f} GB/s, fast tiers {FAST / 1e9:.0f} GB, CPU decode {CPU_DEC / 1e9:.1f} GB/s\n")
    for name, kind, P, A, S in MODELS:
        print(name)
        for f, b in FMT.items():
            if kind == "dense":
                tot, t, tps = dense(P, b); extra = f"pass {t:5.1f} s, x{SPEC_DENSE} tokens/pass (speculative)"
            else:
                tot, t, tps, c, h = moe(P, A, S, b); extra = f"{t:5.2f} s/token, cache holds {100 * c:4.1f}% of experts -> {100 * h:3.0f}% hits"
            print(f"   {f:30s} store {tot / 1e9:7.1f} GB   {tps:6.2f} tok/s   ({extra})")
