#!/usr/bin/env python3
"""gpu_mv_test.py - G1 proof: GPU fused huffman4 decode+matvec vs float64 numpy on real vault blobs."""
import json, subprocess, sys, tempfile, os
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pack_universal_v3 as pu

EXE = HERE / "mv_bench.exe"
CASES = [
    (HERE.parent / "verify_v3" / "uni", ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.mlp.gate_proj.weight",
                                         "model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight"]),
    (HERE.parent / "vault_llama1b", ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight",
                                     "model.embed_tokens.weight"]),
]


def extract(vault, name):
    V = json.load(open(vault / "vault.json")); t = next(x for x in V["tensors"] if x["name"] == name); S = V["streams"][t["stream"]]
    with open(vault / "vault.bin", "rb") as f:
        f.seek(S["off"] + t["blob_off"]); return np.frombuffer(f.read(t["blob_size"]), np.uint8), t


def main():
    rng = np.random.default_rng(0); ok = True
    print(f"{'tensor':58s} {'shape':>14s} {'relL2':>9s} {'time_us':>8s} {'GB/s':>6s}")
    for vault, names in CASES:
        if not (vault / "vault.json").exists():
            print("skip (missing)", vault); continue
        for name in names:
            blob, t = extract(vault, name); rows, cols = t["rows"], t["cols"]
            x = rng.standard_normal(cols).astype(np.float32)
            with tempfile.TemporaryDirectory() as td:
                bp, xp, yp = (os.path.join(td, n) for n in ("blob.bin", "x.bin", "y.bin"))
                blob.tofile(bp); x.tofile(xp)
                r = subprocess.run([str(EXE), bp, xp, yp, "300", *sys.argv[1:3]], capture_output=True, text=True)
                if r.returncode != 0:
                    print("FAIL", name, r.stderr.strip()); ok = False; continue
                y = np.fromfile(yp, np.float32)
            w = pu.decode_blob(blob); yr = np.empty(rows, np.float64); x64 = x.astype(np.float64)
            for a in range(0, rows, 16384):
                blk = (w[a:a + 16384].astype(np.uint32) << 16).view(np.float32).astype(np.float64); yr[a:a + 16384] = blk @ x64
            rel = float(np.linalg.norm(y - yr) / np.linalg.norm(yr)); good = rel < 1e-5; ok &= good
            us = float(r.stdout.split("time=")[1].split(" us")[0]); gbs = float(r.stdout.split("effective=")[1].split(" GB/s")[0])
            print(f"{'PASS' if good else 'FAIL'} {Path(vault).name[:10]:10s} {name[-45:]:45s} {rows:>6d}x{cols:<6d} {rel:9.2e} {us:8.1f} {gbs:6.0f}")
    print("\nG1", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
