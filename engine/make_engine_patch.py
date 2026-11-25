"""make_engine_patch.py - rank-1 weight patches for nvme_engine_u.c:  --patch file.wpat
A patch (layer l, matrix k, u, w) turns that matrix W into W - u w^T without touching the vault: after the matvec the engine
subtracts u * (w . x). k: 0 q, 1 k, 2 v, 3 o, 4 gate, 5 up, 6 down. File: 'WPAT' u32 count, then per patch
i32 layer, i32 k, i32 out_dim, i32 in_dim, f32 u[out_dim], f32 w[in_dim]. Layers run by the GPU worker get the same file."""
import os, re
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")

def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt} match(es), got {n}: {old[:70]!r}"; s = s.replace(old, new)

CODE = r'''/* ---------- rank-1 weight patches (W' = W - u w^T applied after the matvec; the vault stays bit-exact) ---------- */
typedef struct { int l, k, od, id; float *u, *w; } WPatch;
static WPatch *G_PATCH = NULL; static int G_NPATCH = 0; static char G_PATCHF[600] = "";
static void patch_load(Eng *E) {
    if (!G_PATCHF[0]) return;
    FILE *f = fopen(G_PATCHF, "rb"); char mg[4]; uint32_t n;
    if (!f || fread(mg, 1, 4, f) != 4 || memcmp(mg, "WPAT", 4) || fread(&n, 4, 1, f) != 1) die("cannot read patch file");
    Cfg *c = &E->c; int q = c->nh * c->hd, kv = c->nkv * c->hd; int od_k[7] = {q, kv, kv, c->hid, c->inter, c->inter, c->hid}, id_k[7] = {c->hid, c->hid, c->hid, q, c->hid, c->hid, c->inter};
    G_PATCH = calloc(n, sizeof(WPatch)); G_NPATCH = (int)n;
    for (uint32_t i = 0; i < n; i++) {
        WPatch *p = &G_PATCH[i]; int32_t h[4];
        if (fread(h, 4, 4, f) != 4) die("patch file truncated");
        p->l = h[0]; p->k = h[1]; p->od = h[2]; p->id = h[3];
        if (p->l < 0 || p->l >= c->nl || p->k < 0 || p->k > 6 || p->od != od_k[p->k] || p->id != id_k[p->k]) die("patch does not match this model's shapes");
        p->u = malloc(4 * (size_t)p->od); p->w = malloc(4 * (size_t)p->id);
        if (fread(p->u, 4, p->od, f) != (size_t)p->od || fread(p->w, 4, p->id, f) != (size_t)p->id) die("patch file truncated");
    }
    fclose(f); lg("[patch] %d rank-1 patches from %s", G_NPATCH, G_PATCHF); fprintf(stderr, "applying %d rank-1 weight patches from %s\n", G_NPATCH, G_PATCHF);
}
static void mv_p(int l, int k, const uint8_t *blob, const float *X, float *Y, int n) {
    mv_batch(blob, X, Y, n);
    for (int i = 0; i < G_NPATCH; i++) {
        const WPatch *p = &G_PATCH[i]; if (p->l != l || p->k != k) continue;
        for (int t = 0; t < n; t++) {
            const float *x = X + (size_t)t * p->id; float *y = Y + (size_t)t * p->od; double d = 0;
            for (int j = 0; j < p->id; j++) d += (double)p->w[j] * x[j];
            float sf = (float)d; for (int r = 0; r < p->od; r++) y[r] -= sf * p->u[r];
        }
    }
}
'''
rep("static FILE *G_DUMPH = NULL;\n", CODE + "static FILE *G_DUMPH = NULL;\n")
s2 = re.sub(r"mv_batch\(sub\[(\d)\], ", lambda m: f"mv_p(l, {m.group(1)}, sub[{m.group(1)}], ", s)
assert s2.count("mv_p(l, ") == 7, s2.count("mv_p(l, "); s = s2
rep('else if (!strcmp(argv[i], "--gpu-exe") && i + 1 < argc) snprintf(E->gpu_exe, sizeof E->gpu_exe, "%s", argv[++i]);',
    'else if (!strcmp(argv[i], "--gpu-exe") && i + 1 < argc) snprintf(E->gpu_exe, sizeof E->gpu_exe, "%s", argv[++i]);\n'
    '        else if (!strcmp(argv[i], "--patch") && i + 1 < argc) snprintf(G_PATCHF, sizeof G_PATCHF, "%s", argv[++i]);')
rep("eng_init_model(&E, ctx); eng_attach_gpu(&E, ctx);", "eng_init_model(&E, ctx); patch_load(&E); eng_attach_gpu(&E, ctx);", 2)
rep('snprintf(cmd, sizeof cmd, "\\"%s\\" worker \\"%s\\" --ctx %d --layers %d", exe, E->vdir, ctx, E->gpu_layers);',
    'snprintf(cmd, sizeof cmd, "\\"%s\\" worker \\"%s\\" --ctx %d --layers %d%s%s%s", exe, E->vdir, ctx, E->gpu_layers, G_PATCHF[0] ? " --patch \\"" : "", G_PATCHF, G_PATCHF[0] ? "\\"" : "");')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("rank-1 patch support added to", P)
