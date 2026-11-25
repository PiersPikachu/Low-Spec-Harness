"""make_gpu_patch.py - rank-1 weight patches (--patch file.wpat, same format as the CPU engine) for gpu/gpu_engine.cu, in place."""
import os, re
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "gpu_engine.cu")
s = open(P, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "rank1_k" in s:
    print("already patched"); raise SystemExit(0)

def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt} match(es), got {n}: {old[:70]!r}"; s = s.replace(old, new)

KERN = r'''
// ---------------- rank-1 weight patch: y -= u * (w . x)  (one block) ----------------
__global__ void rank1_k(const float *__restrict__ u, const float *__restrict__ w, const float *__restrict__ x, float *__restrict__ y, int od, int id) {
    __shared__ float red[32]; float d = 0.f;
    for (int j = threadIdx.x; j < id; j += blockDim.x) d += w[j] * x[j];
    d = block_sum(d, red);
    for (int r = threadIdx.x; r < od; r += blockDim.x) y[r] -= d * u[r];
}
struct GPatch { int l, k, od, id; float *u, *w; };
static std::vector<GPatch> G_PATCHES;
static void patch_load(const char *path, const Cfg &c) {
    FILE *f = fopen(path, "rb"); char mg[4]; uint32_t n;
    if (!f || fread(mg, 1, 4, f) != 4 || memcmp(mg, "WPAT", 4) || fread(&n, 4, 1, f) != 1) { fprintf(stderr, "cannot read patch file %s\n", path); exit(1); }
    int q = c.nh * c.hd, kv = c.nkv * c.hd; int od_k[7] = {q, kv, kv, c.hid, c.inter, c.inter, c.hid}, id_k[7] = {c.hid, c.hid, c.hid, q, c.hid, c.hid, c.inter};
    for (uint32_t i = 0; i < n; i++) {
        int32_t h[4]; if (fread(h, 4, 4, f) != 4) { fprintf(stderr, "patch file truncated\n"); exit(1); }
        GPatch p{h[0], h[1], h[2], h[3], nullptr, nullptr};
        if (p.l < 0 || p.l >= c.nl || p.k < 0 || p.k > 6 || p.od != od_k[p.k] || p.id != id_k[p.k]) { fprintf(stderr, "patch does not match this model's shapes\n"); exit(1); }
        std::vector<float> u(p.od), w(p.id);
        if (fread(u.data(), 4, p.od, f) != (size_t)p.od || fread(w.data(), 4, p.id, f) != (size_t)p.id) { fprintf(stderr, "patch file truncated\n"); exit(1); }
        CK(cudaMalloc(&p.u, 4 * (size_t)p.od)); CK(cudaMalloc(&p.w, 4 * (size_t)p.id));
        CK(cudaMemcpy(p.u, u.data(), 4 * (size_t)p.od, cudaMemcpyHostToDevice)); CK(cudaMemcpy(p.w, w.data(), 4 * (size_t)p.id, cudaMemcpyHostToDevice));
        G_PATCHES.push_back(p);
    }
    fclose(f); fprintf(stderr, "gpu engine: applying %u rank-1 weight patches from %s\n", n, path);
}
static void apply_patches(int l, int k, const float *x, float *y) {
    for (auto &p : G_PATCHES) if (p.l == l && p.k == k) rank1_k<<<1, 256>>>(p.u, p.w, x, y, p.od, p.id);
}
'''
rep("\n// ---------------------------------------------------------------- host: vault loading", KERN + "\n// ---------------------------------------------------------------- host: vault loading")
# hook every matvec of a layer: mv(S[k], X, Y, ...) -> mv(...); apply_patches(l, k, X, Y);
body_a = s.index("static void layer_step(Eng *E, int l, int pos) {"); body_b = s.index("static void forward(Eng *E", body_a)
body = s[body_a:body_b]
body2, n = re.subn(r"mv\(S\[(\d)\], ([^,]+), ([^,]+), ([^;]+)\);", lambda m: f"mv(S[{m.group(1)}], {m.group(2)}, {m.group(3)}, {m.group(4)}); apply_patches(l, {m.group(1)}, {m.group(2)}, {m.group(3)});", body)
assert n == 7, n
s = s[:body_a] + body2 + s[body_b:]
rep('else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);',
    'else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]); else if (!strcmp(argv[i], "--patch") && i + 1 < argc) patchf = argv[++i];')
rep("std::string cmd = argv[1]; int ctx = 256, ntok = 64, layers = -1;", "std::string cmd = argv[1]; int ctx = 256, ntok = 64, layers = -1; const char *patchf = nullptr;")
rep("Eng E; E.only_layers = (cmd == \"worker\") ? layers : -1; double t0 = now(); if (eng_open(&E, argv[2], ctx)) return 1;",
    "Eng E; E.only_layers = (cmd == \"worker\") ? layers : -1; double t0 = now(); if (eng_open(&E, argv[2], ctx)) return 1;\n    if (patchf) patch_load(patchf, E.c);")
open(P, "w", encoding="utf-8", newline="").write(s)
print("rank-1 patch support added to", P)
