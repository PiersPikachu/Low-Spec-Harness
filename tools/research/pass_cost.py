#!/usr/bin/env python3
"""pass_cost.py - cost of one Hermes-4-14B weight pass vs number of tokens checked (VERIFY n), CPU-only and with GPU layers.
   pass_cost.py [gpu_layers] -> prints s per pass for n = 1, 9, 17, 33, 65"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from spec import Proc
G = int(sys.argv[1]) if len(sys.argv) > 1 else 0
r = json.load(open(HERE / "spec_sweep.json")); ids = r["G"]["fact"]["ids"]; filler = (r["G"]["explain"]["out"] * 3)[:65]
args = [str(_ROOT / "engine" / "nvme_engine_u.exe"), "serve", str(__import__("harness_paths").resolve_vault(os.environ.get("PC_VAULT", "vault_hermes14b_huff"))), "--ctx", "1024", "--batch", "128", "--resident-mb", os.environ.get("PC_RES", "5500"),
        "--log", str(HERE / "pass_cost_log.txt")] + (["--gpu-layers", str(G)] if G else []) + os.environ.get("PC_ARGS", "").split()
T = Proc(args, str(HERE / "pass_cost.err"))
for i in range(0, len(ids), 32): T.verify(ids[i:i + 32])
base = len(ids); out = {}
for n in [int(x) for x in os.environ.get("PC_NS", "1,9,17,33,65,1").split(",")]:
    t0 = time.time(); T.verify(filler[:n]); c = time.time() - t0; T.setpos(base); out.setdefault(n, []).append(c)
    print(f"gpu_layers={G} VERIFY {n:3d}: {c:6.2f} s", flush=True)
T.close()
res = json.load(open(HERE / "pass_cost.json")) if (HERE / "pass_cost.json").exists() else {}
res[os.environ.get("PC_VAULT", "rans") + " " + str(G) + " " + os.environ.get("PC_ARGS", "") + " res" + os.environ.get("PC_RES", "5500")] = out; json.dump(res, open(HERE / "pass_cost.json", "w"), indent=1)
