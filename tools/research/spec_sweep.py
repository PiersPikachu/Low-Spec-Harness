#!/usr/bin/env python3
"""
spec_sweep.py - choose k for speculative decoding from measurements, without paying a full 14B run per k.

Why this is exact: with greedy decoding the target's output G does not depend on k, and the draft's k-token proposal at any point
depends only on the tokens before it. So one real run per prompt gives G; then, for every k, the draft alone replays the rounds
(SETPOS to the round start, GEN k) and counts how many target passes that k would have needed. The cost of one target pass of
n = k+1 tokens is measured directly (VERIFY n). Predicted s/token = (passes x pass_cost(k+1) + draft time) / tokens.
The best k is then confirmed with a real end-to-end run.

  spec_sweep.py run      phase A (real k=4 runs per prompt -> G) + phase B (pass cost for n = 1..9)   [slow: target passes]
  spec_sweep.py replay   phase C (replay k = 1..8 with the draft)                                     [fast]
  spec_sweep.py confirm K  real run at k=K on two prompts, compared with the prediction
results accumulate in spec_sweep.json
"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from spec import Proc
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import harness_paths as hp
TARGET = hp.resolve_vault("vault_hermes14b_huff"); DRAFT = next((hp.resolve_vault(n) for n in ("vault_06b_huff", "uni") if n in hp.find_vaults()), None)
OUT = HERE / "spec_sweep.json"; MAXNEW = 48; CTX = 1024
PROMPTS = {
    "fact":    "In one sentence, what is lossless compression?",
    "explain": "Explain in three sentences how a refrigerator keeps food cold.",
    "code":    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "list":    "List five tips for writing clear technical documentation.",
    "math":    "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Show the steps.",
    "story":   "Write the opening two sentences of a short story set on a lighthouse during a storm.",
}


def load(): return json.load(open(OUT)) if OUT.exists() else {}
def save(r): json.dump(r, open(OUT, "w"), indent=1)


def prompt_ids(tok, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def eos_set():
    kv = dict(l.split()[:2] for l in open(TARGET / "vault.idx") if len(l.split()) == 2); return {int(kv["eos"]), int(kv["eos2"])}


def target_proc():
    return Proc([str(_ROOT / "engine" / "nvme_engine_u.exe"), "serve", str(TARGET), "--ctx", str(CTX), "--batch", "32", "--resident-mb", os.environ.get("SPEC_RES_MB", "5500"),] + (["--gpu-layers", os.environ["SPEC_GPU_LAYERS"]] if os.environ.get("SPEC_GPU_LAYERS") else []) + (["--gpu-exe", os.environ["SPEC_GPU_EXE"]] if os.environ.get("SPEC_GPU_EXE") else []) + [
                 "--log", str(HERE / "spec_target_log.txt")], str(HERE / "spec_target.err"))


def draft_proc():
    return Proc([str(_ROOT / "engine" / "gpu" / "gpu_engine.exe"), "serve", str(DRAFT), "--ctx", str(CTX)], str(HERE / "spec_draft.err"))


def real_run(T, D, ids, k, eos, maxnew=MAXNEW):
    T.send("RESET\n"); T.line(); D.send("RESET\n"); D.line()
    t0 = time.time(); a = None
    for i in range(0, len(ids), 32): a, _ = T.verify(ids[i:i + 32])
    D.gen(ids, 1); D.setpos(len(ids)); t_pre = time.time() - t0
    gen = [a[-1]]; S = ids + gen; passes = drafted = kept = 0; t_draft = t_ver = 0.0; t1 = time.time()
    while gen[-1] not in eos and len(gen) < maxnew:
        x = S[-1]; kk = min(k, maxnew - len(gen))
        td = time.time(); d, _ = D.gen([x], kk); t_draft += time.time() - td
        tv = time.time(); av, _ = T.verify([x] + d); t_ver += time.time() - tv
        j = 0
        while j < len(d) and d[j] == av[j]: j += 1
        add = d[:j] + [av[j]]
        cut = next((i for i, t in enumerate(add) if t in eos), None)
        if cut is not None: add = add[:cut + 1]
        gen += add; S += add; passes += 1; drafted += len(d); kept += j
        T.setpos(len(S) - 1); D.setpos(len(S) - 1)
    gen = gen[:maxnew]
    return dict(k=k, tokens=gen, n=len(gen), prefill_s=t_pre, decode_s=time.time() - t1, passes=passes, drafted=drafted, kept=kept, draft_s=t_draft, verify_s=t_ver)


def run():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TARGET)); eos = eos_set(); r = load(); r.setdefault("G", {}); r.setdefault("realk4", {})
    T = target_proc(); D = draft_proc(); print("engines ready", flush=True)
    for name, text in PROMPTS.items():
        if name in r["G"]: continue
        ids = prompt_ids(tok, text); res = real_run(T, D, ids, 4, eos)
        r["G"][name] = dict(ids=ids, out=res["tokens"]); r["realk4"][name] = {k: v for k, v in res.items() if k != "tokens"}; save(r)
        print(f"[A] {name}: {res['n']} tokens, {res['passes']} passes after prefill, {res['decode_s'] / max(res['n'] - 1, 1):.2f} s/token decode | {ascii(tok.decode(res['tokens'], skip_special_tokens=True))[:110]}", flush=True)
    if "pass_cost" not in r:                                         # phase B: cost of one VERIFY of n tokens
        ids = r["G"]["fact"]["ids"]; T.send("RESET\n"); T.line()
        for i in range(0, len(ids), 32): T.verify(ids[i:i + 32])
        base = len(ids); cost = {}
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 5, 9]:                 # 1, 5, 9 repeated to see the noise
            t0 = time.time(); T.verify(r["G"]["fact"]["out"][:n] if n <= len(r["G"]["fact"]["out"]) else [ids[-1]] * n); c = time.time() - t0
            T.setpos(base); cost.setdefault(str(n), []).append(c); print(f"[B] VERIFY {n}: {c:.2f}s", flush=True)
        r["pass_cost"] = cost; save(r)
    T.close(); D.close()


def replay():
    r = load(); eos = eos_set(); D = draft_proc(); r["replay"] = {}
    for k in range(1, 9):
        tot = dict(passes=0, tokens=0, drafted=0, kept=0, draft_s=0.0)
        for name, g in r["G"].items():
            ids, G = g["ids"], g["out"]; D.send("RESET\n"); D.line(); D.gen(ids, 1); D.setpos(len(ids))
            S = ids + G[:1]; produced = 1; passes = 0
            while produced < len(G) and G[produced - 1] not in eos:
                kk = min(k, MAXNEW - produced); td = time.time(); d, _ = D.gen([S[-1]], kk); tot["draft_s"] += time.time() - td
                j = 0
                while j < len(d) and produced + j < len(G) and d[j] == G[produced + j]: j += 1
                step = min(j + 1, len(G) - produced); S += G[produced:produced + step]; produced += step; passes += 1
                tot["drafted"] += len(d); tot["kept"] += j; D.setpos(len(S) - 1)
            tot["passes"] += passes; tot["tokens"] += len(G) - 1                       # the first token comes from the prefill pass
        r["replay"][str(k)] = tot; print(f"[C] k={k}: {tot['tokens'] / tot['passes']:.2f} tokens per target pass, drafts kept {tot['kept']}/{tot['drafted']} "
                                         f"({100 * tot['kept'] / max(tot['drafted'], 1):.0f}%), draft time {tot['draft_s']:.1f}s", flush=True)
    D.close(); save(r)
    cost = {int(n): sum(v) / len(v) for n, v in r["pass_cost"].items()}
    print("\n k | tokens/pass | pass cost (s, n=k+1) | predicted s/token | vs plain (n=1)")
    best = None
    for k in range(1, 9):
        t = r["replay"][str(k)]; c = cost[k + 1]; spt = (t["passes"] * c + t["draft_s"]) / t["tokens"]
        print(f" {k} | {t['tokens'] / t['passes']:11.2f} | {c:20.2f} | {spt:17.2f} | {cost[1] / spt:6.2f}x")
        if best is None or spt < best[1]: best = (k, spt)
    r["best_k"] = best[0]; save(r); print(f"\nbest k = {best[0]} ({best[1]:.2f} s/token predicted; plain {cost[1]:.2f} s/token)")


def confirm(k):
    from transformers import AutoTokenizer
    r = load(); eos = eos_set(); T = target_proc(); D = draft_proc(); r.setdefault("confirm", {})
    cost = {int(n): sum(v) / len(v) for n, v in r["pass_cost"].items()}
    tag = os.environ.get("SPEC_GPU_LAYERS", "0")
    T.send("RESET\n"); T.line(); ids0 = r["G"]["fact"]["ids"]
    for i in range(0, len(ids0), 32): T.verify(ids0[i:i + 32])
    for n in (1, 5, 9):
        t0 = time.time(); T.verify(r["G"]["fact"]["out"][:n]); c = time.time() - t0; T.setpos(len(ids0)); print(f"[cost gpu_layers={tag}] VERIFY {n}: {c:.2f}s", flush=True)
        r.setdefault("pass_cost_gpu", {}).setdefault(tag, {}).setdefault(str(n), []).append(c)
    for name in ("explain", "code"):
        g = r["G"][name]; res = real_run(T, D, g["ids"], k, eos)
        same = res["tokens"] == g["out"][:len(res["tokens"])]
        rp = sum(1 for _ in [0])
        print(f"[confirm k={k}] {name}: {res['n']} tokens, {res['passes']} passes, {res['decode_s'] / max(res['n'] - 1, 1):.2f} s/token decode, "
              f"output {'IDENTICAL to the k=4 run' if same else 'DIFFERENT'}", flush=True)
        r["confirm"][f"{name}@{k}@gpu{tag}"] = {kk: v for kk, v in res.items() if kk != "tokens"} | {"identical": same}; save(r)
    T.close(); D.close()


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "run"
    run() if c == "run" else replay() if c == "replay" else confirm(int(sys.argv[2]))
