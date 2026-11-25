#!/usr/bin/env python3
"""park_vault.py <vault name or path> [dest_root] - move a vault to slower storage (default D:\\models\\vaults_parked) safely:
copy with robocopy, verify every stream hash of the COPY with the engine's check, and only then delete the original.
The destination root is added to extra_vault_dirs in harness_config.json so the harness still finds the vault.
Reverse it the same way:  park_vault.py D:\\models\\vaults_parked\\<vault> <harness>\\vaults"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import shutil, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import harness_paths as hp

src = hp.resolve_vault(sys.argv[1]); root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"D:\models\vaults_parked")
dst = root / src.name; root.mkdir(parents=True, exist_ok=True); t0 = time.time()
need = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()); free = shutil.disk_usage(root).free
if free < need + 2e9: sys.exit(f"not enough space on {root}: need {need / 1e9:.1f} GB + 2 GB margin, have {free / 1e9:.1f} GB")
print(f"copy {src} -> {dst} ({need / 1e9:.1f} GB)", flush=True)
r = subprocess.run(["robocopy", str(src), str(dst), "/E", "/J", "/NP", "/NFL", "/NDL", "/R:2", "/W:5"], capture_output=True, text=True)
if r.returncode >= 8: sys.exit(f"robocopy failed ({r.returncode}):\n{r.stdout[-800:]}")
print(f"copied in {(time.time() - t0) / 60:.1f} min; verifying every stream of the copy", flush=True)
c = subprocess.run([str(hp.ENGINE), "check", str(dst)], capture_output=True, text=True); last = (c.stdout.strip().splitlines() or [""])[-1]
print(last, flush=True)
if c.returncode != 0 or "bad=0" not in last: sys.exit("VERIFY FAILED: original kept, copy left for inspection")
shutil.rmtree(src); print(f"original removed; vault now at {dst}")
if root.resolve() not in [d["path"] for d in hp.model_dirs()]: hp.add_model_dir(root); print(f"added {root} to the model folders")
print(f"done in {(time.time() - t0) / 60:.1f} min")
