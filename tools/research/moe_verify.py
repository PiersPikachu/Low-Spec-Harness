#!/usr/bin/env python3
"""moe_verify.py <hf_model_dir> <vault_dir> [--resident-mb N] - the MoE engine against Hugging Face transformers in fp32.
  1 last-token logits for 3 prompts (relative L2, argmax, top-5)          2 greedy generation, 24 tokens, token for token
The model is loaded in fp32 on the CPU, so use it for small checkpoints (the synthetic one, or real ones with enough RAM)."""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from verify_engine import run_engine
from spec import Proc
from transformers import AutoModelForCausalLM


def main():
    hfd, vault = sys.argv[1], sys.argv[2]; res = sys.argv[sys.argv.index("--resident-mb") + 1] if "--resident-mb" in sys.argv else "200"
    os.environ["ENGINE_EXTRA_ARGS"] = f"--resident-mb {res} " + os.environ.get("MOE_EXTRA", "")
    m = AutoModelForCausalLM.from_pretrained(hfd, torch_dtype=torch.float32).eval(); V = m.config.vocab_size; ok = True
    rng = np.random.default_rng(1)
    for i in range(3):
        toks = [int(x) for x in rng.integers(0, V, 6 + 5 * i)]
        with torch.no_grad(): ref = m(torch.tensor([toks])).logits[0, -1].numpy()
        got, _ = run_engine(_ROOT / "engine" / "nvme_engine_u.exe", vault, toks, m.config.num_hidden_layers, m.config.hidden_size, want_dump=False)
        rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref)); same = int(got.argmax()) == int(ref.argmax())
        top5 = list(np.argsort(-got)[:5]) == list(np.argsort(-ref)[:5]); ok &= rel < 1e-4 and same
        print(f"{'PASS' if rel < 1e-4 and same else 'FAIL'} logits, {len(toks)} tokens: relL2 {rel:.2e}, argmax {int(got.argmax())}/{int(ref.argmax())}, top5 {'same' if top5 else 'differ'}")
    prompt = [int(x) for x in rng.integers(0, V, 8)]
    with torch.no_grad(): ref = m.generate(torch.tensor([prompt]), max_new_tokens=24, do_sample=False, min_new_tokens=24)[0, len(prompt):].tolist()
    T = Proc([str(_ROOT / "engine" / "nvme_engine_u.exe"), "serve", vault, "--ctx", "128", "--resident-mb", res, "--log", os.devnull] + os.environ.get("MOE_EXTRA", "").split(), os.devnull)
    out, _ = T.gen(prompt, 24); T.send("MOESTATS\n"); st = T.line().split(); T.close()
    same = out == ref; ok &= same
    print(f"{'PASS' if same else 'FAIL'} greedy 24 tokens vs HF fp32: {'identical' if same else f'first difference at {next(i for i in range(min(len(out), len(ref))) if out[i] != ref[i])}'}"
          f"  | expert cache: {st[1]} hits, {st[2]} misses, {st[4]} slots, {st[3]} ms reading")
    print("MOE VERIFY", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
