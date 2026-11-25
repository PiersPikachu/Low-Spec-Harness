#!/usr/bin/env python3
"""
tree_sim.py - how many tokens per 14B pass would SpecExec-style draft TREES give, before building them into the engines?

Exact for greedy decoding: the 14B's output G for each of the 6 prompts is already recorded (spec_sweep.json). Each round, a draft
tree of N nodes is grown best-first by cumulative draft probability (SpecExec: Svirschevski et al., NeurIPS 2024), children = the
draft's top-8 next tokens. The 14B would verify the whole tree in one pass; the tokens gained = longest root path matching G, + 1.
The draft is Qwen3-0.6B in fp32 (same weights as the huffman4 draft vault).
  tree_sim.py [--budgets 1,4,8,16,32,64,128] [--device cpu|cuda]
"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import argparse, heapq, json, math, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
MODEL = Path(r"C:\Users\1fred\Desktop\AI\JEPA\_stage_memory\hf_model\Qwen_Qwen3-0.6B")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budgets", default="1,4,8,16,32,64,128"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--chain", type=int, default=8); a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False); m = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.float32).to(a.device).eval()
    r = json.load(open(HERE / "spec_sweep.json")); eos = {151643, 151645}; cache = {}

    def children(seq):                                            # draft's top-8 next tokens after seq, as (logprob, token)
        key = tuple(seq)
        if key not in cache:
            lg = m(torch.tensor([seq], device=a.device)).logits[0, -1].float().log_softmax(-1); v, i = lg.topk(8)
            cache[key] = list(zip(v.tolist(), i.tolist()))
        return cache[key]

    def tree_gain(prefix, target, budget):
        """grow a best-first tree of `budget` nodes below prefix; return matched depth along target"""
        tree = {}; heap = [(-lp, (t,)) for lp, t in children(prefix)]; heapq.heapify(heap); n = 0
        while heap and n < budget:
            nlp, path = heapq.heappop(heap); tree[path] = -nlp; n += 1
            if path[-1] in eos or len(path) >= 24: continue
            for lp, t in children(prefix + list(path)): heapq.heappush(heap, (nlp - lp, path + (t,)))
        j = 0
        while j < len(target) and tuple(target[:j + 1]) in tree: j += 1
        return j

    def chain_gain(prefix, target, k):                            # today's draft: greedy chain of k tokens
        seq = list(prefix); j = 0
        for _ in range(k):
            t = children(seq)[0][1]
            if j == len([x for x in seq[len(prefix):]]) and j < len(target) and t == target[j]: j += 1
            seq.append(t)
        return j

    out = {}
    for label, B in [(f"chain k={a.chain}", None)] + [(f"tree {b}", int(b)) for b in a.budgets.split(",")]:
        t0 = time.time(); tokens = passes = 0
        for name, g in r["G"].items():
            ids, G = g["ids"], g["out"]; p = 1
            while p < len(G) and G[p - 1] not in eos:
                pre = ids + G[:p]; tgt = G[p:]
                j = chain_gain(pre, tgt, a.chain) if B is None else tree_gain(pre, tgt, B)
                step = min(j + 1, len(G) - p); p += step; passes += 1
            tokens += len(G) - 1
        out[label] = dict(tokens=tokens, passes=passes, tokens_per_pass=tokens / passes)
        print(f"{label:12s}: {tokens / passes:5.2f} tokens per 14B pass ({passes} passes for {tokens} tokens)  [{time.time() - t0:.0f}s]", flush=True)
    json.dump(out, open(HERE / "tree_sim.json", "w"), indent=1)


if __name__ == "__main__":
    main()
