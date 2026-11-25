#!/usr/bin/env python3
"""moe_policy_sim.py <trace.bin> <layers> <experts> <top_k> <cache_slots> - replay a routing trace through candidate expert-cache policies.
  LRU       evict least recently used (engine today)
  LFU-d     evict lowest decayed use count (count *= d every token, +1 per use), ties by recency
  SLRU      segmented LRU: new experts enter a probation segment; a second use promotes to the protected segment (80% of slots)
  Belady    optimal (needs the future): upper bound
Accesses are replayed per token-layer as the engine sees them (all of a layer's experts are needed together)."""
import sys, heapq
from collections import OrderedDict
import numpy as np


def main():
    path, L, E, K, C = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    a = np.fromfile(path, np.int16).reshape(-1, 2 + K); keys = (a[:, 0].astype(np.int64) * E)[:, None] + a[:, 2:]
    seq = [int(x) for x in keys.ravel()]; warm = 32 * L * K; res = {}

    def lru():
        c = OrderedDict(); h = n = 0
        for i, k in enumerate(seq):
            hit = k in c
            if hit: c.move_to_end(k)
            else:
                c[k] = 1
                if len(c) > C: c.popitem(last=False)
            if i >= warm: h += hit; n += 1
        return h / n

    def lfu(d):
        cnt = {}; cache = set(); last = {}; h = n = 0; step = 0
        for i, k in enumerate(seq):
            if i % (L * K) == 0: step += 1
            cnt[k] = cnt.get(k, 0.0) * (d ** (step - last.get(("s", k), step))) + 1.0; last[("s", k)] = step; last[k] = i
            hit = k in cache
            if not hit:
                if len(cache) >= C:
                    v = min(cache, key=lambda x: (cnt[x] * (d ** (step - last[("s", x)])), last[x])); cache.remove(v)
                cache.add(k)
            if i >= warm: h += hit; n += 1
        return h / n

    def slru(frac=0.8):
        prot, prob = OrderedDict(), OrderedDict(); P = int(C * frac); h = n = 0
        for i, k in enumerate(seq):
            hit = k in prot or k in prob
            if k in prot: prot.move_to_end(k)
            elif k in prob:
                del prob[k]; prot[k] = 1
                if len(prot) > P: kk, _ = prot.popitem(last=False); prob[kk] = 1
            else: prob[k] = 1
            while len(prot) + len(prob) > C: prob.popitem(last=False) if prob else prot.popitem(last=False)
            if i >= warm: h += hit; n += 1
        return h / n

    def belady():
        nxt = [0] * len(seq); last = {}
        for i in range(len(seq) - 1, -1, -1): nxt[i] = last.get(seq[i], 1 << 60); last[seq[i]] = i
        bel = {}; heap = []; h = n = 0
        for i, k in enumerate(seq):
            hit = k in bel
            if not hit and len(bel) >= C:
                while True:
                    nn, kk = heapq.heappop(heap)
                    if bel.get(kk) == -nn: del bel[kk]; break
            bel[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
            if i >= warm: h += hit; n += 1
        return h / n

    print(f"{len(seq) // (L * K)} tokens, cache {C} of {L * E} experts ({100 * C / (L * E):.1f}%)")
    for name, f in [("LRU", lru), ("SLRU 80%", slru), ("LFU d=0.99", lambda: lfu(0.99)), ("LFU d=0.95", lambda: lfu(0.95)), ("LFU d=0.8", lambda: lfu(0.8)), ("Belady", belady)]:
        print(f"  {name:12s} hit {100 * f():5.1f}%", flush=True)


if __name__ == "__main__":
    main()
