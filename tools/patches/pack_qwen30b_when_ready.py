"""waits for the Qwen3-30B-A3B download to finish, then packs it (huffman4, every matrix verified bit-exact) into vaults/ and checks it"""
import subprocess, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent.parent; LOGF = Path(r"D:\models\_downloads.log"); OUT = H / "logs" / "pack_qwen30b.txt"
SRC = r"D:\models\Qwen3-30B-A3B-Instruct-2507"; DST = H / "vaults" / "vault_qwen3_30b_a3b_huff"
while "done Qwen/Qwen3-30B-A3B-Instruct-2507" not in LOGF.read_text(encoding="utf-8", errors="ignore"):
    if "FAILED Qwen" in LOGF.read_text(encoding="utf-8", errors="ignore"): open(OUT, "w").write("download failed\n"); sys.exit(1)
    time.sleep(30)
with open(OUT, "w", encoding="utf-8") as f:
    t0 = time.time(); f.write(f"packing {SRC} -> {DST}\n"); f.flush()
    r = subprocess.run([sys.executable, str(H / "pack_universal_v3.py"), "pack", SRC, str(DST)], stdout=f, stderr=subprocess.STDOUT, cwd=str(H))
    f.write(f"\npack exit {r.returncode} after {(time.time() - t0) / 60:.1f} min\n"); f.flush()
    if r.returncode == 0:
        subprocess.run([str(H / "nvme_engine_u.exe"), "check", str(DST)], stdout=f, stderr=subprocess.STDOUT, cwd=str(H))
        subprocess.run([sys.executable, str(H / "vault_defrag.py"), str(DST)], stdout=f, stderr=subprocess.STDOUT, cwd=str(H))
    f.write("ALL DONE\n")
