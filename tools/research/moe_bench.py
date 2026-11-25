#!/usr/bin/env python3
"""moe_bench.py <vault> <resident_mb> [--tokens 48] [--trace file] - real decode speed and expert-cache behaviour of a MoE vault.
Six prompts (the same six as spec_sweep), each answered greedily in a fresh conversation; reports tok/s per prompt from the
engine's own stats, expert cache hits/misses and read time (MOESTATS), and optionally writes the routing trace for moe_locality.py."""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import harness_paths as hp
from spec import Proc
PROMPTS = ["In one sentence, what is lossless compression?", "Explain in three sentences how a refrigerator keeps food cold.",
           "Write a Python function that returns the n-th Fibonacci number iteratively.", "List five tips for writing clear technical documentation.",
           "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Show the steps.",
           "Write the opening two sentences of a short story set on a lighthouse during a storm."]


def main():
    vault = hp.resolve_vault(sys.argv[1]); R = sys.argv[2]; N = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 48
    env_trace = sys.argv[sys.argv.index("--trace") + 1] if "--trace" in sys.argv else None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(vault))
    if env_trace:
        os.environ["NVME_MOE_TRACE"] = env_trace
        if os.path.exists(env_trace): os.remove(env_trace)
    T = Proc([str(hp.ENGINE), "serve", str(vault), "--ctx", "1024", "--batch", "128", "--resident-mb", R, "--log", str(hp.LOGS / "moe_bench_log.txt")] + os.environ.get("MOE_EXTRA", "").split(), str(hp.LOGS / "moe_bench.err"))
    res = []
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        T.send("RESET\n"); T.line(); T.send("MOESTATS\n"); m0 = T.line().split()
        T.send(f"GEN {N} 0 1 1 1 {len(ids)}\n{' '.join(map(str, ids))}\n"); out = []; stats = ""; t0 = time.time()
        while True:
            l = T.line()
            if l.startswith("T "): out.append(int(l.split()[1]))
            elif l.startswith("S "): stats = l[2:].strip()
            elif l.startswith("END"): break
        T.send("MOESTATS\n"); m1 = T.line().split()
        kv = dict(x.split("=", 1) for x in stats.split() if "=" in x)
        hits, miss = int(m1[1]) - int(m0[1]), int(m1[2]) - int(m0[2]); rd = float(m1[3]) - float(m0[3])
        r = dict(prompt=p[:40], tokens=len(out), tok_s=float(kv.get("tok_s", 0)), prefill_ms=float(kv.get("prefill_ms", 0)), wall_s=time.time() - t0,
                 hits=hits, misses=miss, hit_rate=hits / max(hits + miss, 1), read_ms=rd, text=tok.decode(out, skip_special_tokens=True)[:120])
        res.append(r)
        print(f"{r['tok_s']:6.2f} tok/s decode | prefill {r['prefill_ms'] / 1e3:5.1f}s | cache hit {100 * r['hit_rate']:5.1f}% ({miss} misses, {rd / 1e3:5.1f}s reading) | {ascii(r['text'][:70])}", flush=True)
    T.send("MOESTATS\n"); ms = T.line().split(); slots = ms[4]; T.close()
    if len(ms) > 6: print(f"VRAM tier: {ms[6]} slots, {ms[5]} RAM misses served from VRAM, {ms[7]} saved to VRAM, {float(ms[8]) / 1e3:.1f} s copying" + (f", {ms[9]} experts computed on the GPU in {float(ms[10]) / 1e3:.1f} s" if len(ms) > 10 else ""))
    tot_t = sum(r["tokens"] for r in res); print(f"\nresident {R} MB, {slots} expert slots: mean decode {sum(r['tok_s'] for r in res) / len(res):.2f} tok/s, "
                                               f"hit rate {100 * sum(r['hits'] for r in res) / max(sum(r['hits'] + r['misses'] for r in res), 1):.1f}%")
    out = HERE / "moe_bench.json"; allr = json.load(open(out)) if out.exists() else {}
    allr[f"{vault.name}@{R}"] = res; json.dump(allr, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
