#!/usr/bin/env python3
"""test_specp.py - is speculative sampling exact?
 1 logic: a 6-token Markov 'model' (target P, draft Q with different and partly disjoint supports) generated through the full
   draft/verify/accept/rollback loop at k=3; the joint law of 3-token continuations must equal the target chain's (chi-square)
 2 real rows: the target's and the draft's actual distributions for the next token (Qwen3-4B target, Qwen3-0.6B draft, temp 0.7,
   top-p 0.8, top-k 20); accept_step run 200k times; the emitted token's law must equal the target row
 3 draft fidelity: 1500 DRAFTP calls with fresh seeds from one context; their tokens must follow the distribution DRAFTP reports
 4 greedy: temp 0 through the same path must reproduce the recorded 48-token Qwen3-4B greedy reply token for token"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import itertools, math, random, subprocess, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from spec import Proc, accept_step, sample_dist

SM = Path(r"C:\Users\1fred\Desktop\AI\JEPA\_stage_memory"); T4 = SM / "vault3"; TOK4 = SM / "hf_model" / "Quen3-4B"
DR = Path(r"D:\models\cube-harness_archive_20261121\verify_v3\uni")
ok = True


def chi2(obs, exp_p, n):
    """chi-square statistic, dof, and an upper-tail p-value (Wilson-Hilferty normal approximation)"""
    cells = [(obs.get(k, 0), n * p) for k, p in exp_p.items() if n * p >= 5]
    rest_o = n - sum(o for o, _ in cells); rest_e = n - sum(e for _, e in cells)
    if rest_e >= 5: cells.append((rest_o, rest_e))
    x2 = sum((o - e) ** 2 / e for o, e in cells); dof = max(len(cells) - 1, 1)
    z = ((x2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return x2, dof, 0.5 * math.erfc(z / math.sqrt(2))


def report(name, obs, exp_p, n):
    global ok
    x2, dof, pv = chi2(obs, exp_p, n); tv = 0.5 * sum(abs(obs.get(k, 0) / n - exp_p.get(k, 0)) for k in set(obs) | set(exp_p))
    good = pv > 0.001; ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name}: n={n} total-variation {tv:.4f}, chi2 {x2:.1f} on {dof} dof, p={pv:.3f}")


# ---------------------------------------------------------------- 1 logic, synthetic Markov chain
rng = random.Random(7); V = 6
def rowgen(zero_frac):
    w = [rng.random() if rng.random() > zero_frac else 0.0 for _ in range(V)]
    if sum(w) == 0: w[rng.randrange(V)] = 1.0
    s = sum(w); return [(t, x / s) for t, x in enumerate(w) if x > 0]
P = {a: rowgen(0.3) for a in range(V)}; Q = {a: rowgen(0.3) for a in range(V)}
exact = {}
for seq in itertools.product(range(V), repeat=3):
    pr, last = 1.0, 0
    for t in seq: pr *= dict(P[last]).get(t, 0.0); last = t
    if pr > 0: exact[seq] = pr
N = 200000; cnt = Counter(); r = random.Random(11)
for _ in range(N):
    S = [0]
    while len(S) < 4:
        d, qs, last = [], [], S[-1]
        for _ in range(3): qs.append(Q[last]); t = sample_dist(Q[last], r); d.append(t); last = t
        ctx = [S[-1]] + d; ps = [P[c] for c in ctx]
        new, _ = accept_step(d, qs, ps, r); S += new
    cnt[tuple(S[1:4])] += 1
report("logic: 3-token joint law equals the target chain (k=3, disjoint supports)", cnt, exact, N)

# ---------------------------------------------------------------- engines
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(TOK4))
ids = tok.apply_chat_template([{"role": "user", "content": "Name one fruit and say why you like it."}], add_generation_prompt=True, tokenize=True)
ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
T = Proc([str(_ROOT / "engine" / "nvme_engine_u.exe"), "serve", str(T4), "--ctx", "512", "--batch", "32", "--resident-mb", "4500", "--log", str(HERE / "test_specp_log.txt")], str(HERE / "test_specp_t.err"))
D = Proc([str(_ROOT / "engine" / "gpu" / "gpu_engine.exe"), "serve", str(DR), "--ctx", "512"], str(HERE / "test_specp_d.err"))
sp = (0.7, 0.8, 20)
# pick a high-entropy position: the reply is forced to start with a list opener, then the target chooses an item
ctx = ids + tok("Here are three fruits I like: apples,", add_special_tokens=False)["input_ids"]
rows = []
for i in range(0, len(ctx), 32): rows += T.verifyp(ctx[i:i + 32], *sp)
T.setpos(0)
best = max(range(len(ids), len(ctx) + 1), key=lambda i: len(rows[i - 1]))   # row i-1 = distribution of token i
prow = rows[best - 1]; c = ctx[:best]                                  # target's distribution for the token after c
D.gen(c[:-1], 1); D.setpos(len(c) - 1)
_, qs = D.draftp([c[-1]], 1, *sp, 12345); qrow = qs[0]; D.setpos(len(c) - 1)
print(f"context ends {tok.decode(c[-6:])!r}: target row {len(prow)} tokens (top {tok.decode([prow[0][0]])!r} {prow[0][1]:.3f}), "
      f"draft row {len(qrow)} tokens (top {tok.decode([qrow[0][0]])!r} {qrow[0][1]:.3f}), overlap {len(set(dict(prow)) & set(dict(qrow)))}")

# 2 real rows
N = 200000; cnt = Counter(); r = random.Random(3)
for _ in range(N):
    d = sample_dist(qrow, r); new, _ = accept_step([d], [qrow], [prow, [(0, 1.0)]], r); cnt[new[0]] += 1
report("real rows: emitted token ~ target distribution (Qwen3-4B / Qwen3-0.6B, temp 0.7)", cnt, dict(prow), N)

# 3 draft fidelity
n3 = 1500; cnt = Counter(); rep = None
for i in range(n3):
    toks, q = D.draftp([c[-1]], 1, *sp, 1000003 * (i + 1)); cnt[toks[0]] += 1; rep = q[0]; D.setpos(len(c) - 1)
report("draft samples follow the distribution DRAFTP reports", cnt, dict(rep), n3)
T.close(); D.close()

# 4 greedy through the sampled path
ref = "785 12884 7952 6303 1576 34615 304 279 9237 594 16566 44477 23327 6303 92859 315 39020 803 1091 5021 92859 11 1075 2518 476 13753 13 1096 71816 11 3881 438 13255 62969 71816 11 13666 979 39020 83161 448 3720 34615 11 323 6303 3100 374"
out = subprocess.run([sys.executable, str(_ROOT / "app" / "spec.py"), str(T4), str(DR), "--tokenizer", str(TOK4), "--resident-mb", "4500", "--max-new", "48", "--k", "8",
                      "--prompt", "Explain in two sentences why the sky is blue.", "--temp", "0", "--expect", ref], capture_output=True, text=True, encoding="utf-8").stdout
m = [l for l in out.splitlines() if l.startswith("MATCH") or l.startswith("SPEC")]
good = any("IDENTICAL" in l for l in m); ok &= good
print(f"{'PASS' if good else 'FAIL'} greedy (temp 0) through the sampling path: " + " | ".join(m))
print("\nSPECULATIVE SAMPLING TESTS", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
