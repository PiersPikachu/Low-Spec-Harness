#!/usr/bin/env python3
"""
spec.py - greedy speculative decoding across two harness engines. Output is the TARGET model's own greedy output:
the draft only proposes tokens; every token kept is one the target itself would pick at that position.

  spec.py <target_vault> <draft_vault> [--k 8] [--max-new 48] [--prompt "..."] [--target-gpu-layers 0] [--resident-mb N]
          [--draft-engine gpu|cpu] [--plain] [--expect "id id id ..."]

Round: the draft continues greedily for k tokens from the last committed token x; the target runs [x, d1..dk] in ONE weight pass
(VERIFY) and returns its own argmax after each; the longest matching prefix d1..dj is kept plus the target's token at position j
(a correction, or a free extra token when all k matched). Both KV caches are rolled back to the committed length (SETPOS).
--plain runs the target alone (normal greedy GEN) for the same prompt as a reference.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import harness_paths as hp
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


class Proc:
    def __init__(self, args, log):
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "w"), text=True, bufsize=1)
        if self.p.stdout.readline().strip() != "READY": raise SystemExit(f"engine failed to start: {' '.join(args)} (see {log})")
    def send(self, s): self.p.stdin.write(s); self.p.stdin.flush()
    def line(self):
        l = self.p.stdout.readline()
        if not l: raise SystemExit("engine stopped")
        return l
    def setpos(self, n):
        self.send(f"SETPOS {n}\n"); l = self.line()
        if not l.startswith("OK"): raise SystemExit(f"SETPOS failed: {l.strip()}")
    def gen(self, ids, k):
        """greedy GEN: feeds ids, returns (new tokens, engine position after)"""
        self.send(f"GEN {k} 0 1 1 1 {len(ids)}\n{' '.join(map(str, ids))}\n"); out = []
        while True:
            l = self.line()
            if l.startswith("T "): out.append(int(l.split()[1]))
            elif l.startswith("END"): return out, int(l.split()[3])
            elif l.startswith("ERR"): raise SystemExit(l.strip())
    def verify(self, ids):
        self.send(f"VERIFY {len(ids)}\n{' '.join(map(str, ids))}\n"); a = None
        while True:
            l = self.line()
            if l.startswith("A"): a = [int(x) for x in l.split()[1:]]
            elif l.startswith("END"): return a, int(l.split()[3])
            elif l.startswith("ERR"): raise SystemExit(l.strip())
    def verifyp(self, ids, temp, top_p, top_k):
        """target: one weight pass; per position the exact sampling distribution [(token, prob), ...]"""
        self.send(f"VERIFYP {len(ids)} {temp} {top_p} {top_k}\n{' '.join(map(str, ids))}\n"); rows = []
        while True:
            l = self.line()
            if l.startswith("P "): t = l.split(); n = int(t[1]); rows.append([(int(t[2 + 2 * j]), float(t[3 + 2 * j])) for j in range(n)])
            elif l.startswith("END"): return rows
            elif l.startswith("ERR"): raise SystemExit(l.strip())
    def draftp(self, ids, k, temp, top_p, top_k, seed):
        """draft: feed ids, sample k tokens; returns the tokens and the distribution each was drawn from"""
        self.send(f"DRAFTP {k} {temp} {top_p} {top_k} {seed} {len(ids)}\n{' '.join(map(str, ids))}\n"); toks, qs = [], []
        while True:
            l = self.line()
            if l.startswith("D "): t = l.split(); toks.append(int(t[1])); n = int(t[2]); qs.append([(int(t[3 + 2 * j]), float(t[4 + 2 * j])) for j in range(n)])
            elif l.startswith("END"): return toks, qs
            elif l.startswith("ERR"): raise SystemExit(l.strip())
    def close(self):
        try: self.send("QUIT\n"); self.p.wait(timeout=30)
        except Exception: self.p.kill()


def sample_dist(dist, rng):
    r = rng.random(); c = 0.0
    for t, pr in dist:
        c += pr
        if r < c: return t
    return dist[-1][0]


def accept_step(d, qs, ps, rng):
    """speculative sampling (Leviathan et al. 2023; Chen et al. 2023). d[i] ~ qs[i]; ps has len(d)+1 rows (target distributions).
    Accept d[i] with probability min(1, p(d)/q(d)); at the first rejection draw from norm(max(0, p - q)) and stop;
    if all are accepted draw one more token from the last target row. Every emitted token is distributed exactly as the target's
    own sampler at that position. With temp 0 (one-hot rows) this reduces to the greedy rule. Returns (tokens, accepted count)."""
    out = []
    for i, tok in enumerate(d):
        p = dict(ps[i]); q = dict(qs[i]); pd, qd = p.get(tok, 0.0), q.get(tok, 0.0)
        if qd > 0 and rng.random() * qd < pd:
            out.append(tok); continue
        res = [(t, pv - q.get(t, 0.0)) for t, pv in ps[i] if pv > q.get(t, 0.0)]; Z = sum(v for _, v in res)
        out.append(sample_dist([(t, v / Z) for t, v in res], rng) if Z > 0 else sample_dist(ps[i], rng))
        return out, i
    out.append(sample_dist(ps[len(d)], rng)); return out, len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target"); ap.add_argument("draft"); ap.add_argument("--k", type=int, default=8); ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--prompt", default="In one sentence, what is lossless compression?"); ap.add_argument("--target-gpu-layers", type=int, default=0)
    ap.add_argument("--resident-mb", type=int, default=6000); ap.add_argument("--draft-engine", default="gpu", choices=["gpu", "cpu"]); ap.add_argument("--plain", action="store_true")
    ap.add_argument("--expect", default=None); ap.add_argument("--ctx", type=int, default=1024); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--top-p", type=float, default=0.8); ap.add_argument("--top-k", type=int, default=20); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    import harness_paths as hp                                     # names or paths, found wherever the vaults live
    a.target = str(hp.resolve_vault(a.target)) if a.target != 'x' else a.target
    a.draft = str(hp.resolve_vault(a.draft)) if a.draft != 'x' else a.draft
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.target)
    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"): ids = ids["input_ids"]
    ids = list(ids)
    eos = set(); idx = dict(l.split()[:2] for l in open(Path(a.target) / "vault.idx") if len(l.split()) == 2)
    eos |= {int(idx.get("eos", -1)), int(idx.get("eos2", -1))}
    targs = [str(hp.ENGINE), "serve", a.target, "--ctx", str(a.ctx), "--batch", str(a.batch), "--resident-mb", str(a.resident_mb), "--log", str(HERE / "spec_target_log.txt")]
    if a.target_gpu_layers: targs += ["--gpu-layers", str(a.target_gpu_layers)]
    t0 = time.time(); T = Proc(targs, str(hp.LOGS / "spec_target.err")); print(f"target ready ({time.time() - t0:.0f}s): {Path(a.target).name}", flush=True)
    if a.plain:
        t1 = time.time(); out, _ = T.gen(ids, a.max_new); dt = time.time() - t1; T.close()
        print(f"PLAIN greedy: {len(out)} tokens in {dt:.1f}s = {dt / max(len(out), 1):.2f} s/token (includes prefill of {len(ids)} prompt tokens)")
        print("TOKENS", " ".join(map(str, out))); print("TEXT", ascii(tok.decode(out, skip_special_tokens=True))); return
    dargs = ([str(hp.GPU_ENGINE), "serve", a.draft, "--ctx", str(a.ctx)] if a.draft_engine == "gpu"
             else [str(hp.ENGINE), "serve", a.draft, "--ctx", str(a.ctx), "--threads", "2", "--cached", "--log", os.devnull])
    t0 = time.time(); D = Proc(dargs, str(hp.LOGS / "spec_draft.err")); print(f"draft ready ({time.time() - t0:.0f}s): {Path(a.draft).name} on {a.draft_engine}", flush=True)
    t_start = time.time()
    # prefill: target sees the whole prompt (in chunks of --batch), its argmax after the last prompt token is the first generated token
    import random
    rng = random.Random(a.seed); sp = (a.temp, a.top_p, a.top_k)
    tv = time.time(); rows = None
    for i in range(0, len(ids), a.batch): rows = T.verifyp(ids[i:i + a.batch], *sp)
    gen = [sample_dist(rows[-1], rng)]; S = ids + gen; t_prefill = time.time() - tv
    D.gen(ids, 1); D.setpos(len(ids))                                   # draft: prompt into its KV, drop its own sampled token
    rounds = drafted = accepted = 0; t_draft = t_verify = 0.0
    while len(gen) < a.max_new and gen[-1] not in eos:
        x = S[-1]; k = min(a.k, a.max_new - len(gen))
        td = time.time(); d, qs = D.draftp([x], k, *sp, rng.getrandbits(62) | 1); t_draft += time.time() - td
        tv = time.time(); ps = T.verifyp([x] + d, *sp); t_verify += time.time() - tv
        new, j = accept_step(d, qs, ps, rng)
        cut = next((i for i, t in enumerate(new) if t in eos), None)
        if cut is not None: new = new[:cut + 1]
        gen += new; S += new; rounds += 1; drafted += len(d); accepted += j
        T.setpos(len(S) - 1); D.setpos(len(S) - 1)
        print(f"round {rounds:3d}: draft {len(d)} kept {j} +1 -> {len(gen)} tokens  ({time.time() - t_start:6.1f}s)  {ascii(tok.decode(new))}", flush=True)
    dt = time.time() - t_start; gen = gen[:a.max_new]
    print(f"\nSPEC k={a.k} temp={a.temp}: {len(gen)} tokens in {dt:.1f}s = {dt / len(gen):.2f} s/token | prefill {t_prefill:.1f}s | {rounds} target passes, "
          f"{accepted}/{drafted} drafted tokens accepted ({100 * accepted / max(drafted, 1):.0f}%), {len(gen) / max(rounds, 1):.2f} tokens per target pass "
          f"| time in target {t_verify:.1f}s, in draft {t_draft:.1f}s")
    print("TOKENS", " ".join(map(str, gen))); print("TEXT", ascii(tok.decode(gen, skip_special_tokens=True)))
    if a.expect:
        ref = [int(x) for x in a.expect.split()]; n = min(len(ref), len(gen)); diff = next((i for i in range(n) if ref[i] != gen[i]), None)
        print(f"MATCH vs reference: first {n} tokens {'IDENTICAL' if diff is None else 'DIFFER at index %d' % diff}")
    T.close(); D.close()


if __name__ == "__main__":
    main()
