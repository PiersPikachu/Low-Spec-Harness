"""time gpu_experts.dll on one real gpt-oss expert in isolation (no engine threads running) and check it against numpy"""
import ctypes as C, json, os, struct, sys, time
import numpy as np
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); V = os.path.join(H, "data", "vaults", sys.argv[1] if len(sys.argv) > 1 else "vault_gptoss_20b")
cr = C.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\cudart64_12.dll"); gx = C.CDLL(os.path.join(H, "engine", "gpu_experts.dll"))
idx = open(os.path.join(V, "vault.idx")).read().split("\n"); kv = {l.split()[0]: l.split()[1:] for l in idx if l.strip()}
e0 = int(kv["expert_stream0"][0]); st = [l.split() for l in idx if l.startswith(f"stream {e0} ")][0]
off, size, padded = int(st[2]), int(st[3]), int(st[4]); hid, I = int(kv["hidden"][0]), int(kv["moe_inter"][0])
blob = np.fromfile(os.path.join(V, "vault.bin"), np.uint8, count=padded, offset=off)
subs = [struct.unpack_from("<Q", blob, 8 + 8 * i)[0] for i in range(4)]
d = C.c_void_p(); assert cr.cudaMalloc(C.byref(d), C.c_size_t(padded)) == 0
assert cr.cudaMemcpy(d, blob.ctypes.data_as(C.c_void_p), C.c_size_t(padded), 1) == 0
assert gx.gx_init(128, hid, I) == 0
def hdr(o): return struct.unpack_from("<II", blob, o + 4)
rgu, cgu = hdr(subs[0]); rd, cd = hdr(subs[2])
f = gx.gx_mxf4_expert; f.argtypes = [C.c_void_p] + [C.c_uint64] * 4 + [C.c_int] * 4 + [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_void_p]
for m in (1, 4, 8):
    X = np.random.default_rng(0).standard_normal((m, hid)).astype(np.float32); Y = np.zeros((m, rd), np.float32)
    for _ in range(3): f(d, *subs, rgu, cgu, rd, cd, X.ctypes.data, m, 7.0, 1.702, Y.ctypes.data)
    t = time.perf_counter(); n = 200
    for _ in range(n): f(d, *subs, rgu, cgu, rd, cd, X.ctypes.data, m, 7.0, 1.702, Y.ctypes.data)
    dt = (time.perf_counter() - t) / n * 1e3
    # numpy reference for this expert
    FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    def deq(o):
        r, c = hdr(o); nb = c // 32; S = blob[o + 12:o + 12 + r * nb].reshape(r, nb); Bk = blob[o + 12 + r * nb:o + 12 + r * nb + r * c // 2].reshape(r, nb, 16)
        v = np.stack([FP4[Bk & 15], FP4[Bk >> 4]], -1).reshape(r, nb, 32) * np.ldexp(np.float32(1), S.astype(np.int32) - 127)[..., None]; return v.reshape(r, c)
    bgu = np.frombuffer(blob[subs[1] + 8:subs[1] + 8 + 4 * rgu].tobytes(), np.float32); bd = np.frombuffer(blob[subs[3] + 8:subs[3] + 8 + 4 * rd].tobytes(), np.float32)
    gu = X @ deq(subs[0]).T + bgu; g, u = np.minimum(gu[:, ::2], 7), np.clip(gu[:, 1::2], -7, 7)
    ref = ((u + 1) * g / (1 + np.exp(-1.702 * g))) @ deq(subs[2]).T + bd
    print(f"m={m}: {dt:.3f} ms per expert call ({padded / 1e6:.1f} MB expert -> {padded / dt / 1e6:.0f} GB/s), relL2 vs numpy {np.linalg.norm(Y - ref) / np.linalg.norm(ref):.1e}")
