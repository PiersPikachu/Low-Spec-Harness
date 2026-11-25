"""was the 'The user' vs 'User' split a near-tie? logits after the prompt + 'analysis' token, engine CPU-only vs GPU-expert path"""
import os, subprocess, sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_paths as hp
from transformers import AutoTokenizer
v = hp.resolve_vault("vault_gptoss_120b"); tok = AutoTokenizer.from_pretrained(str(v))
ids = tok.apply_chat_template([{"role": "user", "content": "In one sentence, what is lossless compression?"}], add_generation_prompt=True, tokenize=True)
ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids) + tok.encode("analysis<|message|>", add_special_tokens=False, allowed_special="all") if False else tok.encode("analysis", add_special_tokens=False) + [tok.convert_tokens_to_ids("<|message|>")]
for name, extra, env in [("CPU only", [], {"NVME_NO_GPUX": "1"}), ("GPU experts", ["--vram-mb", "4800"], {})]:
    e = dict(os.environ, **env)
    p = subprocess.Popen([str(hp.ENGINE), "serve", str(v), "--ctx", "512", "--resident-mb", "9000", "--log", os.devnull] + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=e)
    assert p.stdout.readline().strip() == "READY"
    if name != "CPU only":       # warm the caches so VRAM holds experts and the GPU path is used
        w = tok.apply_chat_template([{"role": "user", "content": "Explain in three sentences how a refrigerator keeps food cold."}], add_generation_prompt=True, tokenize=True); w = list(w["input_ids"] if hasattr(w, "keys") else w)
        p.stdin.write(f"GEN 48 0 1 1 1 {len(w)}\n{chr(32).join(map(str, w))}\n"); p.stdin.flush()
        while not p.stdout.readline().startswith("END"): pass
        p.stdin.write("RESET\n"); p.stdin.flush(); p.stdout.readline()
    p.stdin.write(f"LOGITS {len(ids)}\n{' '.join(map(str, ids))}\n"); p.stdin.flush(); lg = None
    while True:
        l = p.stdout.readline()
        if not l or l.startswith("END"): break
        if l.startswith("L "): lg = np.array(l[2:].split(), np.float32)
    p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(60)
    o = np.argsort(-lg)[:3]
    print(f"{name:12s}: top3 " + ", ".join(f"{tok.decode([int(i)])!r} {lg[i]:.4f}" for i in o) + f"   margin {lg[o[0]] - lg[o[1]]:.5f}", flush=True)
    if name == "CPU only": ref = lg
    else: print(f"rel. L2 between the two paths: {np.linalg.norm(lg - ref) / np.linalg.norm(ref):.1e}")
