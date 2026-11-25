"""greedy 64 tokens on two vaults, CPU only (no VRAM, no GPU experts): token-for-token identical?"""
import os, sys
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_paths as hp
from spec import Proc
from transformers import AutoTokenizer
os.environ["NVME_NO_GPUX"] = "1"; out = {}
for name in sys.argv[1:3]:
    v = hp.resolve_vault(name); tok = AutoTokenizer.from_pretrained(str(v)); res = []
    T = Proc([str(hp.ENGINE), "serve", str(v), "--ctx", "1024", "--resident-mb", "9000", "--log", os.devnull], os.devnull)
    for p in ["Explain in three sentences how a refrigerator keeps food cold.", "Write a Python function that returns the n-th Fibonacci number iteratively."]:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True); ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        T.send("RESET\n"); T.line(); o, _ = T.gen(ids, 64); res.append(o)
    T.close(); out[name] = res
a, b = out[sys.argv[1]], out[sys.argv[2]]
for i in range(len(a)): print(f"prompt {i}: {'IDENTICAL' if a[i] == b[i] else 'DIFFERENT'} ({len(a[i])} tokens)")
