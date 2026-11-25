#!/usr/bin/env python3
"""
rans4.py - "cube-rANS": the cube ladder with frequency-weighted bases, as a lossless codec for bf16 weight exponents.

Positional view (same as cube_core): a state x is a number; encoding symbol s multiplies the state by M/f_s and adds a digit
(x' = (x // f_s) * M + c_s + x % f_s). With all f_s equal this is exactly the fixed-base ladder; with f_s = frequency it reaches the entropy.
Decoding extracts digits last-in-first-out (most compressed -> uncompressed), like the ladder's unpack.

Per matrix:  exponent byte -> rANS (static per-matrix table, precision M = 2**mbits), sign+mantissa byte -> raw (incompressible, measured).
Chains: a chain = R consecutive rows x one of 4 interleaved column streams (column j goes to stream j % 4), so a row block decodes as 4 independent chains.
Blob 'rans4' layout (little endian):
  u32 rows, u32 cols, u8 mbits, u8 R, u16 0 | u16 freq[256] | u32 off[4*nblk+1] (chain k of block b at 4*b+k) | chain bytes | 8 pad | sm[rows*cols]
Each chain starts with its 4-byte final state, then renormalisation bytes in decode order.

  rans4.py measure <model_dir> [--max-tensors N]     bytes per config vs huffman4 and the order-0 ideal; bit-exact round trip
  rans4.py selftest
"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import argparse, struct, sys, time
from pathlib import Path
import numpy as np
from numba import njit
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pack_universal_v3 as pu

RANS_L = 1 << 23


def quantize(cnt, mbits):
    """frequencies summing to exactly 2**mbits, every present symbol >= 1"""
    M = 1 << mbits; n = int(cnt.sum()); f = np.zeros(256, np.int64)
    nz = np.flatnonzero(cnt)
    if nz.size == 1:
        f[nz[0]] = M; return f
    f[nz] = np.maximum(1, np.floor(cnt[nz] * M / n + 0.5)).astype(np.int64)
    while f.sum() != M:                      # fix rounding drift on the symbols where it costs the fewest bits
        d = M - int(f.sum())
        if d > 0:
            i = nz[np.argmax(cnt[nz] / f[nz])]; f[i] += 1
        else:
            cand = nz[f[nz] > 1]; i = cand[np.argmin(cnt[cand] / f[cand])]; f[i] -= 1
    return f


@njit(cache=True)
def _enc_chain(e, r0, r1, k, freq, cum, mbits, tmp):
    L = 1 << 23; p = tmp.size; x = L; cols = e.shape[1]
    for r in range(r1 - 1, r0 - 1, -1):
        jl = k + ((cols - 1 - k) // 4) * 4 if k < cols else -1
        for j in range(jl, k - 1, -4):
            s = e[r, j]; f = freq[s]
            xmax = ((L >> mbits) << 8) * f
            while x >= xmax:
                p -= 1; tmp[p] = x & 255; x >>= 8
            x = ((x // f) << mbits) + (x % f) + cum[s]
    for b in range(4):
        p -= 1; tmp[p] = (x >> (8 * (3 - b))) & 255
    return p


@njit(cache=True)
def _enc_matrix(e, freq, cum, mbits, R, out, off):
    rows = e.shape[0]; nblk = (rows + R - 1) // R; tmp = np.empty(out.size, np.uint8); pos = 0
    for b in range(nblk):
        r0 = b * R; r1 = min(rows, r0 + R)
        for k in range(4):
            p = _enc_chain(e, r0, r1, k, freq, cum, mbits, tmp); n = tmp.size - p
            off[4 * b + k] = pos; out[pos:pos + n] = tmp[p:]; pos += n
    off[4 * nblk] = pos
    return pos


@njit(cache=True)
def _dec_matrix(buf, off, freq, cum, s2s, mbits, R, rows, cols, out):
    L = 1 << 23; M1 = (1 << mbits) - 1; nblk = (rows + R - 1) // R
    for b in range(nblk):
        r0 = b * R; r1 = min(rows, r0 + R)
        for k in range(4):
            p = off[4 * b + k]
            x = np.int64(buf[p]) | (np.int64(buf[p + 1]) << 8) | (np.int64(buf[p + 2]) << 16) | (np.int64(buf[p + 3]) << 24); p += 4
            for r in range(r0, r1):
                for j in range(k, cols, 4):
                    slot = x & M1; s = s2s[slot]
                    x = freq[s] * (x >> mbits) + slot - cum[s]
                    while x < L:
                        x = (x << 8) | buf[p]; p += 1
                    out[r, j] = s


def encode(w, mbits=14, R=16):
    """w: uint16 (rows, cols) bf16 bits -> rans4 blob (np.uint8)"""
    rows, cols = w.shape; e = ((w >> 7) & 255).astype(np.uint8); sm = (((w >> 8) & 0x80) | (w & 0x7F)).astype(np.uint8)
    freq = quantize(np.bincount(e.ravel(), minlength=256), mbits); cum = np.concatenate([[0], np.cumsum(freq)[:-1]]).astype(np.int64)
    nblk = (rows + R - 1) // R; off = np.zeros(4 * nblk + 1, np.int64); buf = np.empty(rows * cols * 2 + 16 * nblk + 64, np.uint8)
    n = _enc_matrix(e, freq, cum, mbits, R, buf, off)
    if n >= 1 << 32:
        raise SystemExit("matrix too large for 32-bit chain offsets")
    hdr = struct.pack('<IIBBH', rows, cols, mbits, R, 0) + freq.astype('<u2').tobytes() + off.astype('<u4').tobytes()
    return np.concatenate([np.frombuffer(hdr, np.uint8), buf[:n], np.zeros(8, np.uint8), sm.ravel()])


def parse(blob):
    rows, cols, mbits, R, _ = struct.unpack('<IIBBH', bytes(blob[:12])); freq = blob[12:524].view('<u2').astype(np.int64)
    nblk = (rows + R - 1) // R; o0 = 524; off = blob[o0:o0 + 4 * (4 * nblk + 1)].view('<u4').astype(np.int64); c0 = o0 + 4 * (4 * nblk + 1)
    n = int(off[-1]); sm = blob[c0 + n + 8:c0 + n + 8 + rows * cols].reshape(rows, cols)
    return rows, cols, mbits, R, freq, off, blob[c0:c0 + n + 8], sm


def decode(blob):
    rows, cols, mbits, R, freq, off, buf, sm = parse(blob)
    cum = np.concatenate([[0], np.cumsum(freq)[:-1]]).astype(np.int64); s2s = np.repeat(np.arange(256, dtype=np.int64), freq)
    e = np.empty((rows, cols), np.uint8); _dec_matrix(buf, off, freq, cum, s2s, mbits, R, rows, cols, e)
    sm = sm.astype(np.uint16); return ((sm & 0x80) << 8) | (e.astype(np.uint16) << 7) | (sm & 0x7F)


def measure(model_dir, max_tensors=0):
    T = pu.scan_model(model_dir); names = [k for k, t in T.items() if len(t["shape"]) >= 2 and t["dtype"] == "BF16"]
    if max_tensors: names = names[::max(1, len(names) // max_tensors)]
    cfgs = [(m, R) for m in (12, 14) for R in (1, 4, 16, 64)]; tot = {c: 0 for c in cfgs}; huff = ideal = raw = 0; bad = 0; t0 = time.time()
    for i, name in enumerate(names):
        t = T[name]; rows = t["shape"][0]; cols = int(np.prod(t["shape"][1:])); w = np.ascontiguousarray(pu.load_raw(t).reshape(rows, cols)); raw += w.nbytes
        huff += pu.compress_matrix(w).size
        cnt = np.bincount(((w >> 7) & 255).ravel(), minlength=256).astype(np.float64); p = cnt[cnt > 0] / cnt.sum(); ideal += -(cnt[cnt > 0] * np.log2(p)).sum() / 8 + w.size
        for c in cfgs:
            b = encode(w, *c); tot[c] += b.size
            if c == (14, 16) and not np.array_equal(decode(b), w): bad += 1; print("ROUNDTRIP MISMATCH", name)
        if i % 25 == 0: print(f"\r{i + 1}/{len(names)} {time.time() - t0:.0f}s", end="", flush=True)
    print(f"\n{Path(model_dir).name}: {len(names)} matrices, raw {raw / 1e9:.3f} GB  (round trip of config M=2^14,R=16 on every matrix: {'BIT-EXACT' if not bad else str(bad) + ' MISMATCHES'})")
    print(f"{'codec':28s} {'GB':>8s} {'ratio':>8s} {'vs huffman4':>12s}")
    print(f"{'huffman4 (vault today)':28s} {huff / 1e9:8.4f} {huff / raw:8.4f}")
    for c in cfgs: print(f"{'rans4 M=2^%d R=%d' % c:28s} {tot[c] / 1e9:8.4f} {tot[c] / raw:8.4f} {100 * (tot[c] - huff) / raw:+11.3f}%")
    print(f"{'order-0 ideal (entropy)':28s} {ideal / 1e9:8.4f} {ideal / raw:8.4f} {100 * (ideal - huff) / raw:+11.3f}%")
    return bad == 0


def selftest():
    rng = np.random.default_rng(0); ok = True
    for shape, kind in (((37, 101), "normal"), ((5, 3), "normal"), ((1, 1), "normal"), ((64, 4096), "normal"), ((9, 2050), "normal"), ((6, 17), "single"), ((23, 130), "random16")):
        if kind == "single": w = np.full(shape, 0x3F80, np.uint16)
        elif kind == "random16": w = rng.integers(0, 65536, shape, dtype=np.uint16)
        else: w = pu._bf16(rng.standard_normal(shape) * 0.02)
        for m in (12, 14):
            for R in (1, 3, 16):
                g = np.array_equal(decode(encode(w, m, R)), w); ok &= g
                if not g: print("FAIL", shape, kind, m, R)
    print("RANS4 SELFTEST", "PASSED" if ok else "FAILED"); return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("measure"); a.add_argument("model_dir"); a.add_argument("--max-tensors", type=int, default=0); sp.add_parser("selftest")
    x = ap.parse_args(); sys.exit(0 if (selftest() if x.cmd == "selftest" else measure(x.model_dir, x.max_tensors)) else 1)
