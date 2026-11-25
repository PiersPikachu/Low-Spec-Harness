#!/usr/bin/env python3
"""run_14b.py - generate with the full Hermes-4-14B vault (19.7 GB compressed) on this machine through the planner's chosen layout.
  run_14b.py split [max_new]   first N layers on the GPU worker, 13 in RAM, rest streamed from NVMe
  run_14b.py cpu   [max_new]   CPU engine only (RAM-resident + NVMe streaming)
Greedy decoding, ChatML prompt from the model's own tokenizer. Prints tokens as they arrive and the engine's timing line."""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
from transformers import AutoTokenizer
VAULT = sys.argv[3] if len(sys.argv) > 3 else str(HERE / "vault_hermes14b"); TOKDIR = r"D:\models\Hermes-4-14B"
mode = sys.argv[1] if len(sys.argv) > 1 else "split"; max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 32
tok = AutoTokenizer.from_pretrained(TOKDIR)
msgs = [{"role": "user", "content": "In one sentence, what is lossless compression?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
if isinstance(ids, dict) or hasattr(ids, "keys"): ids = ids["input_ids"]
print(f"prompt: {len(ids)} tokens; mode={mode}; max_new={max_new}", flush=True)
args = [str(_ROOT / "engine" / "nvme_engine_u.exe"), "serve", VAULT, "--ctx", "512", "--log", str(HERE / "run14b_log.txt")]
args += ["--gpu-layers", "10", "--resident-mb", "6200"] if mode == "split" else ["--resident-mb", "6200"]
t0 = time.time(); p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
assert p.stdout.readline().strip() == "READY", "engine did not start"; print(f"engine ready after {time.time() - t0:.0f}s (weights pinned/loaded)", flush=True)
p.stdin.write(f"GEN {max_new} 0 1 1 1 {len(ids)}\n{' '.join(map(str, ids))}\n"); p.stdin.flush()
out = []; t1 = time.time(); stats = end = ""
while True:
    ln = p.stdout.readline()
    if not ln: break
    if ln.startswith("T "):
        out.append(int(ln.split()[1])); print(f"  token {len(out):2d} at {time.time() - t1:6.1f}s: {tok.decode([out[-1]])!r}", flush=True)
    elif ln.startswith("S "): stats = ln.strip()
    elif ln.startswith("END") or ln.startswith("ERR"): end = ln.strip(); break
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=60)
print("\nEND:", end); print("STATS:", stats)
print("\nREPLY (ascii-escaped):", tok.decode(out).encode("unicode_escape").decode("ascii"))
Path(HERE / f"run14b_{mode}_{Path(VAULT).name}_tokens.txt").write_text(" ".join(map(str, out)))
