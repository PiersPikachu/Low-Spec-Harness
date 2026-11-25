#!/usr/bin/env python3
"""gen_compare.py - GEN protocol parity: the GPU engine vs the CPU engine, same vault, same prompt, same sampler settings.
Greedy (temp 0) must agree token for token until float noise flips a near-tie; reports the first divergence if any."""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import os, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
CPU = _ROOT / "engine" / "nvme_engine_u.exe"; GPU = _ROOT / "engine" / "gpu" / "gpu_engine.exe"


def run(exe, vault, prompt, max_new, temp, topp, topk, seed, extra=()):
    td = tempfile.mkdtemp()
    p = subprocess.Popen([str(exe), "serve", str(vault), "--ctx", "512", *extra, "--log", os.path.join(td, "l.txt")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    assert p.stdout.readline().strip() == "READY"
    p.stdin.write(f"GEN {max_new} {temp} {topp} {topk} {seed} {len(prompt)}\n{' '.join(map(str, prompt))}\n"); p.stdin.flush()
    toks, end, stats = [], "", ""
    while True:
        ln = p.stdout.readline()
        if not ln: break
        if ln.startswith("T "): toks.append(int(ln.split()[1]))
        elif ln.startswith("S "): stats = ln.strip()
        elif ln.startswith("END") or ln.startswith("ERR"): end = ln.strip(); break
    p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=30)
    return toks, end, stats


def main(vault, vocab, split=None, resident_mb=None, cpu_resident_mb=None):
    ok = True
    prompt = [int((1000 + 7919 * i) % vocab) for i in range(12)]
    for label, (temp, topp, topk, seed) in {"greedy": (0.0, 1.0, 1, 1), "sampled t=0.7 p=0.8 k=20 seed=7": (0.7, 0.8, 20, 7)}.items():
        res = ("--resident-mb", str(resident_mb)) if resident_mb else ()
        cres = ("--resident-mb", str(cpu_resident_mb)) if cpu_resident_mb else res
        a, ea, sa = run(CPU, vault, prompt, 48, temp, topp, topk, seed, ("--cached", "--threads", "2") + cres)
        if split:
            b, eb, sb = run(CPU, vault, prompt, 48, temp, topp, topk, seed, ("--cached", "--threads", "2", "--gpu-layers", str(split)) + res)
        else:
            b, eb, sb = run(GPU, vault, prompt, 48, temp, topp, topk, seed)
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        same = a == b
        print(f"{'PASS' if same else 'DIVERGES'} {label}: {len(a)} vs {len(b)} tokens, first difference at index {n}; end cpu='{ea}' gpu='{eb}'")
        print(f"      cpu {sa}\n      gpu {sb}")
        ok &= (same or (n is not None and n >= 24))     # float-noise tie flips late in a sequence are tolerated, early ones are not
    print("GEN PARITY", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]; split = int(a[a.index('--split') + 1]) if '--split' in a else None; res = int(a[a.index('--resident-mb') + 1]) if '--resident-mb' in a else None
    cres = int(a[a.index('--cpu-resident-mb') + 1]) if '--cpu-resident-mb' in a else None
    sys.exit(0 if main(a[0], int(a[1]), split, res, cres) else 1)
