#!/usr/bin/env python3
"""parity_vs_c.py <vault> [engine options] - the Rust engine (R_EXE, default target/release/cube) against nvme_engine_u (C_EXE) on the
serve protocol: LOGITS / SETPOS / GEN / VERIFY / VERIFYP must match line for line, logits and per-layer hidden states bit for bit.
Env: NT=prompt tokens (45), NVME_KV_MARGIN to force sliding-window rings."""
import subprocess, sys, os, random, struct, time, tempfile
C = os.environ.get("C_EXE", "/home/claude/refbuild/nvme_engine_u"); R = os.environ.get("R_EXE", "target/release/cube")
def run(exe, vault, cmds, extra=(), env=None):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run([exe, "serve", vault, "--ctx", "256", "--threads", "2", "--log", os.path.join(tempfile.gettempdir(), "plog.txt")] + list(extra),
                       input="".join(cmds) + "QUIT\n", capture_output=True, text=True, env=e, timeout=1800)
    if p.returncode: print(p.stderr[-2000:]); raise SystemExit(f"{exe} failed")
    return [l for l in p.stdout.splitlines() if not l.startswith("S ") and l != "READY"], p.stdout
def bits(line):
    return [struct.unpack("<I", struct.pack("<f", float(x)))[0] for x in line.split()[1:]]
vault = sys.argv[1]; extra = sys.argv[2:]
random.seed(1)
V = int([l.split()[1] for l in open(os.path.join(vault, "vault.idx")) if l.startswith("vocab ")][0]) if os.path.exists(os.path.join(vault,"vault.idx")) else 1000
toks = [random.randrange(min(V, 50000)) for _ in range(int(os.environ.get("NT","45")))]
cmds = [f"LOGITS {len(toks)}\n{' '.join(map(str,toks))}\n", f"LOGITS 3\n5 6 7\n", "SETPOS 20\n", "LOGITS 1\n9\n", "RESET\n",
        f"GEN 12 0.8 0.9 30 1234 {len(toks)}\n{' '.join(map(str,toks))}\n", "RESET\n",
        f"VERIFY 4\n1 2 3 4\n", f"VERIFYP 3 0.7 0.8 20\n4 5 6\n", "RESET\n", f"GEN 8 0 1 1 0 5\n10 11 12 13 14\n"]
res = {}
for name, exe in (("C", C), ("R", R)):
    t = time.time(); out, raw = run(exe, vault, cmds, extra, {"NVME_DUMP_H": os.path.join(tempfile.gettempdir(), f"dump_{name}.bin")}); res[name] = out
    print(f"{name}: {time.time()-t:.1f}s, {len(out)} lines")
a, b = res["C"], res["R"]
ok = len(a) == len(b)
for i, (x, y) in enumerate(zip(a, b)):
    if x.startswith("L ") and y.startswith("L "):
        bx, by = bits(x), bits(y); nd = sum(1 for p, q in zip(bx, by) if p != q)
        if nd or len(bx) != len(by): ok = False; print(f"line {i}: logits differ in {nd}/{len(bx)}"); 
        else: print(f"line {i}: {len(bx)} logits bit-identical")
    elif x.split()[:3] != y.split()[:3] if x.startswith("END verify") else x != y:
        ok = False; print(f"line {i}: C={x[:120]!r}\n         R={y[:120]!r}")
dc, dr = open(os.path.join(tempfile.gettempdir(), "dump_C.bin"),"rb").read(), open(os.path.join(tempfile.gettempdir(), "dump_R.bin"),"rb").read()
print("hidden dumps:", len(dc), len(dr), "identical" if dc == dr else "DIFFER")
ok &= dc == dr
print("PARITY", "PASS" if ok else "FAIL")
