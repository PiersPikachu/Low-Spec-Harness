import subprocess, sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent.parent
ids = [int(x) for x in open(H / "tests" / "data" / "gptoss_ids.txt").read().split()]
p = subprocess.Popen([str(H / "engine" / "nvme_engine_u.exe"), "serve", sys.argv[1], "--ctx", "512", "--batch", "128", "--resident-mb", __import__("os").environ.get("RES_MB", "4000"), "--log", str(H / "data" / "logs" / "gptoss_engine_log.txt")] + __import__("os").environ.get("MOE_EXTRA", "").split(),
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
assert p.stdout.readline().strip() == "READY", p.stderr.read()[-500:]
p.stdin.write(f"LOGITS {len(ids)}\n{' '.join(map(str, ids))}\n"); p.stdin.flush(); lg = None
while True:
    ln = p.stdout.readline()
    if not ln or ln.startswith("END") or ln.startswith("ERR"): print(ln.strip()); break
    if ln.startswith("L "): lg = np.array(ln[2:].split(), dtype=np.float32)
p.stdin.write("MOESTATS\n"); p.stdin.flush(); print("stats", p.stdout.readline().strip()); p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(30)
ref = np.load(H / "tests" / "data" / "ref_logits.npy"); hf = np.load(H / "tests" / "data" / "gptoss_hf_logits.npy")
print(f"engine vs numpy fp32 reference: relL2 {np.linalg.norm(lg - ref) / np.linalg.norm(ref):.2e}, argmax {lg.argmax()} / {ref.argmax()}, top5 {np.argsort(-lg)[:5].tolist()} / {np.argsort(-ref)[:5].tolist()}")
print(f"engine vs transformers bf16:      relL2 {np.linalg.norm(lg - hf) / np.linalg.norm(hf):.2e}, argmax {lg.argmax()} / {hf.argmax()}")
