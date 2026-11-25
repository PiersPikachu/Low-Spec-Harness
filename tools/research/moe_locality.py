#!/usr/bin/env python3
"""moe_locality.py <trace.bin> <layers> <experts> <top_k> [--warm N] - expert cache hit rate vs cache size, from a real routing trace.

The engine writes one record per (token, layer) when NVME_MOE_TRACE=<file> is set: int16 layer, int16 position, top_k int16 expert ids.
For cache sizes from 1% to 100% of all experts this replays the trace through
  LRU     what the engine does today
  static  the experts most used in the first --warm tokens pinned (a frozen hot set)
  Belady  the optimal policy (evict the expert needed furthest in the future): an upper bound for any cache
and reports the hit rate after the warm-up tokens, plus the bytes per token streamed from disk at a given expert size.
"""
import sys
from collections import OrderedDict, Counter
import numpy as np


def load(path, K):
    a = np.fromfile(path, np.int16).reshape(-1, 2 + K); return a


def main():
    path, L, E, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    warm = int(sys.argv[sys.argv.index("--warm") + 1]) if "--warm" in sys.argv else 32
    xmb = float(sys.argv[sys.argv.index("--expert-mb") + 1]) if "--expert-mb" in sys.argv else 6.3
    a = load(path, K); keys = (a[:, 0].astype(np.int64) * E)[:, None] + a[:, 2:]          # global expert id per access
    order = [int(x) for x in keys.ravel()]; ntok = len(a) // L
    print(f"trace: {len(a)} token-layer records = {ntok} tokens x {L} layers, {len(order)} expert accesses, {len(set(order))} distinct of {L * E}")
    warm_acc = warm * L * K
    cnt = Counter(order); top = [k for k, _ in cnt.most_common()]
    cov = np.cumsum(sorted(cnt.values(), reverse=True)) / len(order)
    for f in (0.1, 0.25, 0.5):
        print(f"  {100 * f:.0f}% of experts carry {100 * cov[int(f * L * E) - 1]:.1f}% of accesses" if int(f * L * E) - 1 < len(cov) else "")
    nxt = [0] * len(order); last = {}
    for i in range(len(order) - 1, -1, -1): nxt[i] = last.get(order[i], 1 << 60); last[order[i]] = i
    print(f"\n cache (% of experts) | LRU hit | static hot-set hit | Belady (optimal) hit | MB/token from disk (LRU)")
    for f in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0):
        C = max(1, int(f * L * E)); lru = OrderedDict(); h_l = h_s = h_b = n = 0
        hot = set(Counter(order[:warm_acc]).most_common(C)); hot = {k for k, _ in hot}
        import heapq
        bel = {}; heap = []
        for i, k in enumerate(order):
            post = i >= warm_acc
            if k in lru: lru.move_to_end(k); h_l += post
            else:
                lru[k] = 1
                if len(lru) > C: lru.popitem(last=False)
            h_s += post and k in hot
            if k in bel: h_b += post
            elif len(bel) >= C:
                while True:
                    nn, kk = heapq.heappop(heap)
                    if bel.get(kk) == -nn: del bel[kk]; break
            bel[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k)); n += post
        miss_per_tok = (1 - h_l / max(n, 1)) * L * K
        print(f"  {100 * f:5.0f}%  ({C:5d})        | {100 * h_l / max(n, 1):6.1f}% | {100 * h_s / max(n, 1):17.1f}% | {100 * h_b / max(n, 1):19.1f}% | {miss_per_tok * xmb:7.0f}")


if __name__ == "__main__":
    main()
