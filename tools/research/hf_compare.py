#!/usr/bin/env python3
"""
hf_compare.py - token-for-token comparison of the compressed-vault engines against Hugging Face transformers (the original, uncompressed weights).

For each model and prompt (greedy decoding, same token ids everywhere):
  HF fp32   transformers, weights upcast bf16->fp32 (exact), fp32 math      <- closest thing to the original model's "true" output
  HF bf16   transformers, bf16 math (how people usually run the model)
  CPU eng   nvme_engine_u.exe on the vault        GPU eng   gpu_engine.exe on the vault
Reported: first index where two token streams differ, and at that step the top-1 vs top-2 logit margin of HF fp32 (a tiny margin = a near-tie, i.e. rounding noise).
Also the last-prompt-token logits: relative L2 of each system against HF fp32.
Because HF bf16 vs HF fp32 are both *uncompressed*, their disagreement is the noise floor any engine has to be judged against.
"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_compare as gc
import verify_engine as ve

PROMPTS = ["The capital of France is", "Explain in one sentence why the sky is blue.", 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
           "Q: What is 17 * 23?\nA:", "Once upon a time in a small village by the sea,", "List three advantages of compressing neural network weights losslessly:\n1.",
           "The three laws of thermodynamics state that", "Translate to French: 'Where is the train station?'\n"]
NEW = 48


def hf_generate(model, ids, n):
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False, return_dict_in_generate=True, output_scores=True, pad_token_id=0,
                             temperature=None, top_p=None, top_k=None)
    toks = out.sequences[0, len(ids):].tolist(); margins = []
    for s in out.scores:
        t2 = torch.topk(s[0].float(), 2).values; margins.append(float(t2[0] - t2[1]))
    return toks, margins


def hf_last_logits(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0, -1].float().numpy()


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return None


def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare_model(name, model_dir, vault, with_gpu=True):
    cfg = json.load(open(Path(model_dir) / "config.json")); nl = cfg["num_hidden_layers"]; hid = cfg["hidden_size"]; tok = AutoTokenizer.from_pretrained(model_dir)
    ids_list = [tok(p)["input_ids"] for p in PROMPTS]
    keep = [i for i, ids in enumerate(ids_list) if max(ids) < cfg["vocab_size"]]
    if len(keep) < len(ids_list): print(f"note: {len(ids_list) - len(keep)} prompt(s) skipped (token id >= vocab_size {cfg['vocab_size']})")
    PR = [PROMPTS[i] for i in keep]; ids_list = [ids_list[i] for i in keep]; print(f"\n=== {name}: {len(PR)} prompts x {NEW} greedy tokens ===", flush=True)
    t0 = time.time(); m32 = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    ref = [hf_generate(m32, ids, NEW) for ids in ids_list]; l32 = [hf_last_logits(m32, ids) for ids in ids_list]; del m32
    print(f"HF fp32 done ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time(); m16 = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
    b16 = [hf_generate(m16, ids, NEW)[0] for ids in ids_list]; lb16 = [hf_last_logits(m16, ids) for ids in ids_list]; del m16
    print(f"HF bf16 done ({time.time() - t0:.0f}s)", flush=True)
    systems = {"HF bf16": (b16, lb16)}
    for label, exe, extra, on in (("CPU engine", gc.CPU, ("--cached", "--threads", "8", "--resident-mb", "3500"), True), ("GPU engine", gc.GPU, (), with_gpu)):
        if not on: continue
        toks = []; lg = []
        for ids in ids_list:
            t, end, st = gc.run(exe, vault, ids, NEW, 0.0, 1.0, 1, 1, extra); toks.append(t)
            l, _ = ve.run_engine(exe if label == "GPU engine" else gc.CPU, vault, ids, nl, hid, want_dump=False); lg.append(l)
        systems[label] = (toks, lg)
    rep = []; summary = {k: dict(identical=0, diverged=0, first=[]) for k in systems}; lrel = {k: [] for k in systems}
    for pi, (p, ids) in enumerate(zip(PR, ids_list)):
        ref_t, margins = ref[pi]; line = f"[{pi}] {p[:34]!r:38s}"
        for k, (toks, lg) in systems.items():
            t = toks[pi]; d = first_diff(t, ref_t)
            if d is None and len(t) == len(ref_t): summary[k]["identical"] += 1; line += f" | {k}: identical ({len(t)})"
            else:
                summary[k]["diverged"] += 1; d = d if d is not None else min(len(t), len(ref_t)); summary[k]["first"].append(d)
                mg = f"{margins[d]:.3f}" if d < len(margins) else "n/a"; line += f" | {k}: differs @{d} (HF fp32 top1-top2 margin {mg}, len {len(t)}/{len(ref_t)})"
            lrel[k].append(rel(np.asarray(lg[pi]), l32[pi]))
        print(line, flush=True); rep.append(line)
    print(f"\n--- {name} summary vs HF fp32 (greedy, {NEW} tokens x {len(PR)} prompts) ---")
    for k in systems:
        s = summary[k]; print(f"{k:11s}: {s['identical']}/{len(PR)} prompts identical, {s['diverged']} diverge (first-diff indices {s['first']}) | last-token logits relL2 vs HF fp32: mean {np.mean(lrel[k]):.2e} max {np.max(lrel[k]):.2e}")
    print("sample (prompt 0):", ascii(tok.decode(ref[0][0][:24])))
    return summary


if __name__ == "__main__":
    which = sys.argv[1:] or ["0.6b-rans", "1b-rans"]
    jobs = {"0.6b": ("Qwen3-0.6B", r"C:\Users\1fred\Desktop\AI\JEPA\_stage_memory\hf_model\Qwen_Qwen3-0.6B", r"D:\models\cube-harness_archive_20261121\verify_v3\uni"),
            "1b": ("Llama-3.2-1B", r"D:\models\cube-harness_archive_20261121\models\Llama-3.2-1B", r"D:\models\cube-harness_archive_20261121\vault_llama1b"),
            "0.6b-rans": ("Qwen3-0.6B [rans4 vault]", r"C:\Users\1fred\Desktop\AI\JEPA\_stage_memory\hf_model\Qwen_Qwen3-0.6B", _ROOT / "data" / "vaults" / "vault_06b_rans"),
            "1b-rans": ("Llama-3.2-1B [rans4 vault]", r"D:\models\cube-harness_archive_20261121\models\Llama-3.2-1B", _ROOT / "data" / "vaults" / "vault_llama1b_rans"),
            "14b-mini": ("Hermes-4-14B first 3 layers (real weights, vocab 32768)", r"D:\models\hermes14b_mini3", r"D:\models\vault_hermes14b_mini3")}
    for w in which: compare_model(*jobs[w])
