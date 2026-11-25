"""how cheaply can the MXFP4 block scales be packed? per-row range of the E8M0 scale bytes (gpt-oss-120b, layers 0, 17, 35)"""
import json, numpy as np
from safetensors import safe_open
d = r"D:\models\gpt-oss-120b"; wm = json.load(open(d + r"\model.safetensors.index.json"))["weight_map"]
tot_rows = 0; fits = {2: 0, 4: 0}; bits_entropy = 0.0; nscale = 0; nweights = 0
for l in (0, 17, 35):
    for t in ("gate_up_proj", "down_proj"):
        n = f"model.layers.{l}.mlp.experts.{t}_scales"
        with safe_open(d + "\\" + wm[n], "np") as f: S = f.get_tensor(n).astype(np.int16)      # [E, rows, nb]
        S = S.reshape(-1, S.shape[-1]); rng = S.max(1) - S.min(1); tot_rows += len(S)
        for b in (2, 4): fits[b] += int((rng < (1 << b)).sum())
        nscale += S.size; nweights += S.size * 32
        c = np.bincount(S.ravel() - S.min(), minlength=1); p = c[c > 0] / c.sum(); bits_entropy += -(p * np.log2(p)).sum() * S.size
print(f"{tot_rows} rows, {nscale} scales: rows whose scale range fits 2 bits {100 * fits[2] / tot_rows:.1f}%, 4 bits {100 * fits[4] / tot_rows:.1f}%")
print(f"order-0 scale entropy {bits_entropy / nscale:.2f} bits of 8")
for b in (2, 4):
    f = fits[b] / tot_rows; per_w = (f * b + (1 - f) * 8) / 32 + 0.0      # + one base byte per row ~ negligible
    print(f"  base + {b}-bit offsets (8-bit for rows that do not fit): {per_w:.3f} bits/weight for scales vs 0.250 -> total {4 + per_w:.3f} of 4.250 ({100 * (4 + per_w) / 4.25:.1f}%)")
