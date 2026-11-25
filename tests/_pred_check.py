"""next-layer expert prediction accuracy (NVME_MOE_PREDICT), decode only: prompt prefilled, stats reset, then 64 greedy tokens"""
import os, sys
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_paths as hp
from spec import Proc
from transformers import AutoTokenizer
v = hp.resolve_vault(sys.argv[1]); tok = AutoTokenizer.from_pretrained(str(v)); os.environ["NVME_MOE_PREDICT"] = "1"
T = Proc([str(hp.ENGINE), "serve", str(v), "--ctx", "1024", "--resident-mb", "9000", "--vram-mb", "4800", "--log", str(hp.LOGS / "pred_log.txt")], str(hp.LOGS / "pred.err"))
tot = [0, 0]
for p in ["Explain in three sentences how a refrigerator keeps food cold.", "Write a Python function that returns the n-th Fibonacci number iteratively."]:
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True); ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    T.send("RESET\n"); T.line(); T.send("MOESTATS\n"); a = T.line().split()
    T.send(f"GEN 64 0 1 1 1 {len(ids)}\n{' '.join(map(str, ids))}\n")
    while not T.line().startswith("END"): pass
    T.send("MOESTATS\n"); b = T.line().split(); h, n = int(b[11]) - int(a[11]), int(b[12]) - int(a[12]); tot[0] += h; tot[1] += n
    print(f"{p[:40]:40s} next-layer prediction: {h}/{n} = {100 * h / max(n, 1):.1f}% of used experts predicted (incl. prefill)", flush=True)
T.close(); print(f"overall {100 * tot[0] / max(tot[1], 1):.1f}%")
