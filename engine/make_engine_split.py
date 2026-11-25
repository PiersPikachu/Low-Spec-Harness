"""make_engine_split.py - adds split mode to nvme_engine_u.c:  --gpu-layers N [--gpu-exe path]
Layers 0..N-1 run inside gpu_engine.exe (worker, weights compressed in VRAM); embedding, layers N.., final norm, lm_head and sampling stay here.
Only the hidden state (hid floats per token) crosses the pipe."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")

def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, f"expected {cnt} match(es), got {n}: {old[:70]!r}"
    s = s.replace(old, new)

rep("uint8_t *resident[40]; int first_streamed;", "uint8_t *resident[128]; int first_streamed; int gpu_layers; char gpu_exe[600]; double t_gpu_ms;")
# lift the 40-stream limit (Hermes-4-14B / Qwen3-14B has 40 layers -> 42 streams, 41 timing records)
rep("SRec rec[40];", "SRec rec[128];")
rep("int order[40], no = 0;", "int order[128], no = 0;")
rep('if (E->nst > 40) die("too many streams");', 'if (E->nst > 120) die("too many streams");')
rep("E->st = calloc(64, sizeof(Stream));", "E->st = calloc(128, sizeof(Stream));")
rep("if (idx < 64) {", "if (idx < 128) {")
rep("double lw_sum[40] = {0}, lr_sum[40] = {0}, lc_sum[40] = {0};", "double lw_sum[128] = {0}, lr_sum[128] = {0}, lc_sum[128] = {0};")
rep("(size_t)(max_new + 2) * 40);", "(size_t)(max_new + 2) * 128);")
rep('else if (!strcmp(argv[i], "--cached")) E->direct = 0;',
    'else if (!strcmp(argv[i], "--gpu-layers") && i + 1 < argc) E->gpu_layers = atoi(argv[++i]);\n'
    '        else if (!strcmp(argv[i], "--gpu-exe") && i + 1 < argc) snprintf(E->gpu_exe, sizeof E->gpu_exe, "%s", argv[++i]);\n'
    '        else if (!strcmp(argv[i], "--cached")) E->direct = 0;')
# streams for the GPU layers are never pinned, loaded or prefetched here
rep("if (r < E->c.nl) order[no++] = r; } }", "if (r < E->c.nl && r >= E->gpu_layers) order[no++] = r; } }")
rep("E->first_streamed = -1; for (int i = 0; i < E->c.nl; i++) if (!E->resident[i])", "E->first_streamed = -1; for (int i = E->gpu_layers; i < E->c.nl; i++) if (!E->resident[i])")
rep("if (!E->resident[i] && E->st[i].padded > E->bufsz)", "if (i >= E->gpu_layers && !E->resident[i] && E->st[i].padded > E->bufsz)")

WORKER = r'''/* ---------- split mode: GPU worker process (layers 0..gpu_layers-1) ---------- */
static HANDLE w_in = NULL, w_out = NULL, w_proc = NULL; static float *w_dump = NULL;
static void wr_all(const void *p, size_t n) { const char *b = (const char *)p; while (n) { DWORD w = 0; if (!WriteFile(w_in, b, (DWORD)(n > (1u << 20) ? (1u << 20) : n), &w, NULL) || !w) die("GPU worker pipe closed (write)"); b += w; n -= w; } }
static void rd_all(void *p, size_t n) { char *b = (char *)p; while (n) { DWORD r = 0; if (!ReadFile(w_out, b, (DWORD)(n > (1u << 20) ? (1u << 20) : n), &r, NULL) || !r) die("GPU worker pipe closed (read)"); b += r; n -= r; } }
static void worker_quit(void) { if (!w_proc) return; uint32_t h[4] = {0x54495551u, 0, 0, 0}; DWORD w; WriteFile(w_in, h, 16, &w, NULL); if (WaitForSingleObject(w_proc, 3000) != WAIT_OBJECT_0) TerminateProcess(w_proc, 0); w_proc = NULL; }
static void eng_attach_gpu(Eng *E, int ctx) {
    if (E->gpu_layers <= 0) return;
    if (E->gpu_layers > E->c.nl) die("--gpu-layers exceeds the layer count");
    char exe[600]; if (E->gpu_exe[0]) snprintf(exe, sizeof exe, "%s", E->gpu_exe); else { char me[600]; GetModuleFileNameA(NULL, me, sizeof me); char *sl = strrchr(me, '\\'); if (sl) *sl = 0; snprintf(exe, sizeof exe, "%s\\gpu\\gpu_engine.exe", me); }
    SECURITY_ATTRIBUTES sa; sa.nLength = sizeof sa; sa.lpSecurityDescriptor = NULL; sa.bInheritHandle = TRUE; HANDLE ir, iw, orr, ow;
    if (!CreatePipe(&ir, &iw, &sa, 1 << 20) || !CreatePipe(&orr, &ow, &sa, 1 << 20)) die("cannot create pipes");
    SetHandleInformation(iw, HANDLE_FLAG_INHERIT, 0); SetHandleInformation(orr, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si; memset(&si, 0, sizeof si); si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES; si.hStdInput = ir; si.hStdOutput = ow; si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi; char cmd[1500]; snprintf(cmd, sizeof cmd, "\"%s\" worker \"%s\" --ctx %d --layers %d", exe, E->vdir, ctx, E->gpu_layers);
    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) { fprintf(stderr, "cannot start GPU worker: %s\n", exe); exit(1); }
    CloseHandle(ir); CloseHandle(ow); CloseHandle(pi.hThread); w_in = iw; w_out = orr; w_proc = pi.hProcess;
    char line[64]; int n = 0; for (;;) { char ch; DWORD got = 0; if (!ReadFile(w_out, &ch, 1, &got, NULL) || !got) { fprintf(stderr, "GPU worker failed to start (its message is above)\n"); exit(1); } if (ch == '\n') break; if (n < 63) line[n++] = ch; } line[n] = 0;
    if (strncmp(line, "READY", 5)) die("GPU worker handshake failed");
    w_dump = malloc(4 * (size_t)E->gpu_layers * E->c.hid); atexit(worker_quit);
    lg("[split] GPU worker runs layers 0..%d (%s); this process runs layers %d..%d + final norm + lm_head", E->gpu_layers - 1, exe, E->gpu_layers, E->c.nl - 1);
}
static void gpu_layers_call(Eng *E, int n, int pos0) {
    int hid = E->c.hid; double t = now(); uint32_t h[4] = {0x31574446u, (uint32_t)n, (uint32_t)pos0, G_DUMPH ? 1u : 0u};
    wr_all(h, 16); wr_all(E->H, 4 * (size_t)n * hid); rd_all(E->H, 4 * (size_t)n * hid);
    if (G_DUMPH) { rd_all(w_dump, 4 * (size_t)E->gpu_layers * hid); fwrite(w_dump, 4, (size_t)E->gpu_layers * hid, G_DUMPH); }
    E->t_gpu_ms = (now() - t) * 1e3;
}

'''
rep("static void forward(Eng *E, const int *tok, int n, int want_logits) {", WORKER + "static void forward(Eng *E, const int *tok, int n, int want_logits) {")
rep("E->t_embed = (now() - t0) * 1e3; E->t_attn = 0;", "E->t_embed = (now() - t0) * 1e3; E->t_attn = 0;\n    if (E->gpu_layers > 0) gpu_layers_call(E, n, pos0);")
rep("for (int i = 0; i < ns; i++) {\n        int s = (i < c->nl) ? i : E->lm_s; const uint8_t *blob; int ub = -1;", "for (int i = E->gpu_layers; i < ns; i++) {\n        int s = (i < c->nl) ? i : E->lm_s; const uint8_t *blob; int ub = -1;")
rep("eng_init_model(&E, ctx);", "eng_init_model(&E, ctx); eng_attach_gpu(&E, ctx);", 2)
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("split mode added to", P)
