/* nvme_engine.c - lossless NVMe weight-streaming inference for Qwen3-style models (bf16 weights, weights never change).
 *   pack   <model_dir> <vault_dir>            safetensors -> vault (raw sign+mantissa bytes + Huffman-coded exponents)
 *   serve  <vault_dir> [options]              stream layers from disk (double buffer), fused decode+matvec, stdin/stdout protocol
 *   check  <vault_dir> [options]              read every stream, verify hashes, log sequential read speed
 *   selftest
 * options: --ctx N --threads N --batch N --io-threads N --io-chunk MB --cached --log FILE --log-detail N
 * Build: gcc -O3 -march=native -ffp-contract=off -fopenmp -static nvme_engine.c -o nvme_engine.exe -lm -lpsapi */
#ifndef _WIN32
#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <malloc.h>
#define FSEEK64(f, o) _fseeki64(f, (__int64)(o), SEEK_SET)
#define FTELL64(f) ((uint64_t)_ftelli64(f))
#else
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#define FSEEK64(f, o) fseeko(f, (off_t)(o), SEEK_SET)
#define FTELL64(f) ((uint64_t)ftello(f))
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#define NOINLINE __attribute__((noinline))
#define LMAX 12
#define LMASK ((1u << LMAX) - 1)
#define SECT 4096

static double now(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}
static FILE *G_LOG = NULL;
static void lg(const char *fmt, ...) {
    va_list a; va_start(a, fmt);
    if (G_LOG) { vfprintf(G_LOG, fmt, a); fputc('\n', G_LOG); fflush(G_LOG); }
    va_end(a);
}
static void stamp(char *b, size_t n) { time_t t = time(NULL); struct tm *tm = localtime(&t); strftime(b, n, "%Y-%m-%d %H:%M:%S", tm); }
static void die(const char *m) { fprintf(stderr, "ERROR: %s\n", m); lg("[error] %s", m); exit(1); }

/* ---------- threads ---------- */
typedef struct { void (*fn)(void *); void *arg;
#ifdef _WIN32
    HANDLE h;
#else
    pthread_t h;
#endif
} Thr;
#ifdef _WIN32
static DWORD WINAPI tramp(LPVOID p) {
    Thr *t = p;
    /* optional: boost IO/loader threads so busy compute threads cannot starve them (env NVME_IO_PRIO: 0=off(default), 1=above normal, 2=highest) */
    { const char *e = getenv("NVME_IO_PRIO"); int lvl = e ? atoi(e) : 0;
      if (lvl == 1) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
      else if (lvl >= 2) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); }
    t->fn(t->arg); return 0;
}
static void thr_start(Thr *t, void (*fn)(void *), void *a) { t->fn = fn; t->arg = a; t->h = CreateThread(NULL, 0, tramp, t, 0, NULL); }
static void thr_join(Thr *t) { WaitForSingleObject(t->h, INFINITE); CloseHandle(t->h); }
#else
static void *tramp(void *p) { Thr *t = p; t->fn(t->arg); return NULL; }
static void thr_start(Thr *t, void (*fn)(void *), void *a) { t->fn = fn; t->arg = a; pthread_create(&t->h, NULL, tramp, t); }
static void thr_join(Thr *t) { pthread_join(t->h, NULL); }
#endif

/* ---------- aligned memory, system info ---------- */
static void *aalloc(size_t n) {
    n = (n + SECT - 1) & ~(size_t)(SECT - 1);
#ifdef _WIN32
    return _aligned_malloc(n, SECT);
#else
    void *p = NULL; if (posix_memalign(&p, SECT, n)) return NULL; return p;
#endif
}
#ifndef _WIN32
static double proc_kb(const char *file, const char *key) {
    FILE *f = fopen(file, "r"); char line[256]; double v = -1; size_t kl = strlen(key); if (!f) return -1;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, key, kl)) { v = atof(line + kl); break; }
    fclose(f); return v;
}
#endif
static void mem_info(double *tot, double *avail, double *ws, double *peak) {
#ifdef _WIN32
    MEMORYSTATUSEX ms; ms.dwLength = sizeof ms; GlobalMemoryStatusEx(&ms);
    *tot = ms.ullTotalPhys / 1048576.0; *avail = ms.ullAvailPhys / 1048576.0;
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) { *ws = pmc.WorkingSetSize / 1048576.0; *peak = pmc.PeakWorkingSetSize / 1048576.0; } else *ws = *peak = -1;
#else
    *tot = proc_kb("/proc/meminfo", "MemTotal:") / 1024.0; *avail = proc_kb("/proc/meminfo", "MemAvailable:") / 1024.0;
    *ws = proc_kb("/proc/self/status", "VmRSS:") / 1024.0; *peak = proc_kb("/proc/self/status", "VmHWM:") / 1024.0;
#endif
}

/* ---------- random-access file with optional direct (unbuffered) I/O ---------- */
typedef struct {
#ifdef _WIN32
    HANDLE h;
#else
    int fd;
#endif
    int direct;
} RFile;
static int rf_open(RFile *f, const char *path, int want_direct) {
#ifdef _WIN32
    DWORD fl = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN | (want_direct ? FILE_FLAG_NO_BUFFERING : 0);
    f->h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, fl, NULL);
    if (f->h == INVALID_HANDLE_VALUE) { if (want_direct) return rf_open(f, path, 0); return -1; }
    f->direct = want_direct; return 0;
#else
    int fl = O_RDONLY;
#ifdef O_DIRECT
    if (want_direct) fl |= O_DIRECT;
#endif
    f->fd = open(path, fl);
    if (f->fd < 0 && want_direct) { f->fd = open(path, O_RDONLY); f->direct = 0; return f->fd < 0 ? -1 : 0; }
    f->direct = want_direct; return f->fd < 0 ? -1 : 0;
#endif
}
static int rf_pread(RFile *f, void *buf, uint64_t off, size_t len) {
    uint8_t *p = buf;
    while (len) {
#ifdef _WIN32
        DWORD n = (DWORD)(len > (1u << 30) ? (1u << 30) : len), got = 0; OVERLAPPED ov; memset(&ov, 0, sizeof ov);
        ov.Offset = (DWORD)(off & 0xFFFFFFFFu); ov.OffsetHigh = (DWORD)(off >> 32); ov.hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
        BOOL ok = ReadFile(f->h, p, n, NULL, &ov);
        if (!ok && GetLastError() != ERROR_IO_PENDING) { CloseHandle(ov.hEvent); return -1; }
        ok = GetOverlappedResult(f->h, &ov, &got, TRUE); CloseHandle(ov.hEvent);
        if (!ok || !got) return -1;
        p += got; off += got; len -= got;
#else
        ssize_t n = pread(f->fd, p, len, (off_t)off);
        if (n < 0) { if (f->direct && errno == EINVAL) { fcntl(f->fd, F_SETFL, fcntl(f->fd, F_GETFL) & ~O_DIRECT); f->direct = 0; continue; } return -1; }
        if (!n) return -1;
        p += n; off += (uint64_t)n; len -= (size_t)n;
#endif
    }
    return 0;
}
typedef struct { RFile *f; uint8_t *buf; uint64_t off; size_t len, chunk; int err; } Rd;
static void rd_fn(void *a) {
    Rd *r = a; size_t done = 0; r->err = 0;
    while (done < r->len) { size_t n = r->len - done; if (n > r->chunk) n = r->chunk;
        if (rf_pread(r->f, r->buf + done, r->off + done, n)) { r->err = 1; return; } done += n; }
}
static int read_stream(RFile *f, uint8_t *buf, uint64_t off, size_t len, int nt, size_t chunk) {
    Rd r[8]; Thr th[8]; int n = 0; size_t pos = 0; memset(r, 0, sizeof r);
    if (nt < 1) nt = 1; if (nt > 8) nt = 8;
    size_t per = ((len / nt) + SECT - 1) & ~(size_t)(SECT - 1);
    for (int i = 0; i < nt && pos < len; i++) { size_t l = per; if (pos + l > len) l = len - pos;
        r[i].f = f; r[i].buf = buf + pos; r[i].off = off + pos; r[i].len = l; r[i].chunk = chunk; r[i].err = 0; pos += l; n++; }
    for (int i = 1; i < n; i++) thr_start(&th[i], rd_fn, &r[i]);
    rd_fn(&r[0]);
    int err = r[0].err; for (int i = 1; i < n; i++) { thr_join(&th[i]); err |= r[i].err; }
    return err;
}

/* ---------- canonical Huffman over exponents (max code length 12) ---------- */
static void huff_lengths(const uint64_t *freq, uint8_t *len) {
    uint64_t f[256]; memcpy(f, freq, sizeof f);
    for (;;) {
        memset(len, 0, 256);
        int idx[256], n = 0;
        for (int i = 0; i < 256; i++) if (f[i]) idx[n++] = i;
        if (n == 0) return;
        if (n == 1) { len[idx[0]] = 1; return; }
        uint64_t w[512]; int par[512], al[512], na = n, m = n;
        for (int k = 0; k < n; k++) { w[k] = f[idx[k]]; par[k] = -1; al[k] = k; }
        while (na > 1) {
            int a = 0, b = -1;
            for (int k = 1; k < na; k++) if (w[al[k]] < w[al[a]]) a = k;
            for (int k = 0; k < na; k++) { if (k == a) continue; if (b < 0 || w[al[k]] < w[al[b]]) b = k; }
            w[m] = w[al[a]] + w[al[b]]; par[m] = -1; par[al[a]] = m; par[al[b]] = m;
            al[a] = m; al[b] = al[na - 1]; na--; m++;
        }
        int mx = 0;
        for (int k = 0; k < n; k++) { int d = 0; for (int p = par[k]; p >= 0; p = par[p]) d++; len[idx[k]] = (uint8_t)d; if (d > mx) mx = d; }
        if (mx <= LMAX) return;
        for (int i = 0; i < 256; i++) if (f[i]) f[i] = (f[i] + 1) >> 1;
    }
}
static void make_rcodes(const uint8_t *len, uint32_t *rc) {
    int cnt[LMAX + 2] = {0}; uint32_t next[LMAX + 2], c = 0;
    for (int s = 0; s < 256; s++) if (len[s]) cnt[len[s]]++;
    for (int l = 1; l <= LMAX; l++) { c = (c + cnt[l - 1]) << 1; next[l] = c; }
    for (int s = 0; s < 256; s++) {
        int l = len[s]; if (!l) { rc[s] = 0; continue; }
        uint32_t code = next[l]++, r = 0;
        for (int i = 0; i < l; i++) r |= ((code >> i) & 1u) << (l - 1 - i);
        rc[s] = r;
    }
}
static void build_lut(const uint8_t *len, uint16_t *lut) {
    uint32_t rc[256]; make_rcodes(len, rc); memset(lut, 0, sizeof(uint16_t) << LMAX);
    for (int s = 0; s < 256; s++) { int l = len[s]; if (!l) continue;
        for (uint32_t k = 0; k < (1u << (LMAX - l)); k++) lut[rc[s] | (k << l)] = (uint16_t)(s | (l << 8)); }
}
/* blob v3: rows u32 | cols u32 | len[256] | stream_off u32[4*rows+1] | exponent streams (+8 pad) | sign+mantissa bytes rows*cols
 * each row's exponents are split into 4 interleaved bitstreams (symbols j%4==k) so 4 decode chains run in parallel */
static uint8_t *compress_layer(const uint16_t *w, int rows, int cols, size_t *outsz, double *ent_bits) {
    uint64_t freq[256] = {0}; size_t n = (size_t)rows * cols;
    for (size_t i = 0; i < n; i++) freq[(w[i] >> 7) & 255]++;
    if (ent_bits) for (int s = 0; s < 256; s++) if (freq[s]) *ent_bits -= (double)freq[s] * log2((double)freq[s] / (double)n);
    uint8_t len[256]; huff_lengths(freq, len); uint32_t rc[256]; make_rcodes(len, rc);
    size_t rowcap = ((size_t)cols * LMAX + 7) / 8 + 64;
    uint8_t *tmp = malloc(rowcap * rows); size_t no4 = 4 * (size_t)rows + 1; uint32_t *off = malloc(4 * no4); size_t pos = 0;
    if (!tmp || !off) die("out of memory in compress_layer");
    for (int r = 0; r < rows; r++) for (int k = 0; k < 4; k++) {
        off[4 * (size_t)r + k] = (uint32_t)pos; uint64_t acc = 0; int nb = 0; const uint16_t *row = w + (size_t)r * cols;
        for (int j = k; j < cols; j += 4) { int e = (row[j] >> 7) & 255; acc |= (uint64_t)rc[e] << nb; nb += len[e];
            while (nb >= 8) { tmp[pos++] = (uint8_t)acc; acc >>= 8; nb -= 8; } }
        if (nb > 0) tmp[pos++] = (uint8_t)acc;
    }
    if (pos > 0xFFFFFFF0u) die("matrix too large for 32-bit stream offsets");
    off[4 * (size_t)rows] = (uint32_t)pos;
    size_t p_off = 264, p_es = p_off + 4 * no4, p_sm = p_es + pos + 8, total = p_sm + n;
    uint8_t *b = calloc(total, 1); uint32_t R = rows, C = cols;
    if (!b) die("out of memory in compress_layer");
    memcpy(b, &R, 4); memcpy(b + 4, &C, 4); memcpy(b + 8, len, 256);
    memcpy(b + p_off, off, 4 * no4); memcpy(b + p_es, tmp, pos);
    for (size_t i = 0; i < n; i++) b[p_sm + i] = (uint8_t)(((w[i] >> 8) & 0x80) | (w[i] & 0x7F));
    free(tmp); free(off); *outsz = total; return b;
}
static inline void decode_row(const uint16_t *lut, const uint8_t *es, const uint32_t *o4, const uint8_t *sm, int cols, float *out) {
    const uint8_t *p0 = es + o4[0], *p1 = es + o4[1], *p2 = es + o4[2], *p3 = es + o4[3];
    uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0; int c0 = 0, c1 = 0, c2 = 0, c3 = 0, j = 0;
#define REF(b, c, p) if (c < 24) { uint32_t v; memcpy(&v, p, 4); p += 4; b |= (uint64_t)v << c; c += 32; }
#define DEC(b, c, J) { uint32_t e = lut[b & LMASK]; int l = (int)(e >> 8); b >>= l; c -= l; e &= 255; uint32_t s = sm[J]; \
        uint32_t bits = ((s & 0x80u) << 24) | (e << 23) | ((s & 0x7Fu) << 16); memcpy(&out[J], &bits, 4); }
    for (; j + 8 <= cols; j += 8) {   /* 4 independent chains, 2 symbols each per refill (bc >= 24, codes <= 12 bits) */
        REF(b0, c0, p0) REF(b1, c1, p1) REF(b2, c2, p2) REF(b3, c3, p3)
        DEC(b0, c0, j) DEC(b1, c1, j + 1) DEC(b2, c2, j + 2) DEC(b3, c3, j + 3)
        DEC(b0, c0, j + 4) DEC(b1, c1, j + 5) DEC(b2, c2, j + 6) DEC(b3, c3, j + 7)
    }
    for (; j < cols; j++) {
        int k = j & 3; uint64_t *b = k == 0 ? &b0 : k == 1 ? &b1 : k == 2 ? &b2 : &b3; int *c = k == 0 ? &c0 : k == 1 ? &c1 : k == 2 ? &c2 : &c3;
        const uint8_t **p = k == 0 ? &p0 : k == 1 ? &p1 : k == 2 ? &p2 : &p3;
        if (*c < LMAX) { uint32_t v; memcpy(&v, *p, 4); *p += 4; *b |= (uint64_t)v << *c; *c += 32; }
        DEC(*b, *c, j)
    }
#undef REF
#undef DEC
}
static inline void raw_row(const uint16_t *w, int cols, float *out) {
    for (int j = 0; j < cols; j++) { uint32_t b = (uint32_t)w[j] << 16; memcpy(&out[j], &b, 4); }
}
static NOINLINE float dot_row(const float *w, const float *x, int n) {
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0; int j = 0;
    for (; j + 8 <= n; j += 8) {
        a0 += w[j] * x[j]; a1 += w[j + 1] * x[j + 1]; a2 += w[j + 2] * x[j + 2]; a3 += w[j + 3] * x[j + 3];
        a4 += w[j + 4] * x[j + 4]; a5 += w[j + 5] * x[j + 5]; a6 += w[j + 6] * x[j + 6]; a7 += w[j + 7] * x[j + 7];
    }
    for (; j < n; j++) a0 += w[j] * x[j];
    return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}
#if defined(__AVX2__)
/* multi-symbol table: 12-bit window -> up to 3 exponent symbols. entry = syms(24) | bits_used(4)<<24 | count(2)<<28 */
static uint32_t lut3_entry(const uint16_t *lut, uint32_t v) {
    uint32_t bits = v, used = 0, cnt = 0, syms = 0;
    while (cnt < 3) { uint32_t e = lut[bits & LMASK]; uint32_t l = e >> 8; if (l == 0 || used + l > LMAX) break; syms |= (e & 255u) << (8 * cnt); cnt++; used += l; bits >>= l; }
    if (cnt == 0) { syms = 0; used = 1; cnt = 1; }
    return syms | (used << 24) | (cnt << 28);
}
#define LK(b, c, pos, e) { uint32_t t = l3[b & LMASK]; memcpy(e + pos, &t, 4); int u = (t >> 24) & 15; b >>= u; c -= u; pos += (t >> 28) & 3; }
#define RF(b, c, p) if (c < 24) { uint32_t v; memcpy(&v, p, 4); p += 4; b |= (uint64_t)v << c; c += 32; }
/* decode the 4 interleaved exponent streams of one row into 4 byte arrays (4 independent chains) */
static inline void dec4(const uint32_t *l3, const uint8_t *es, const uint32_t *o4, int cols, uint8_t *e0, uint8_t *e1, uint8_t *e2, uint8_t *e3) {
    const uint8_t *p0 = es + o4[0], *p1 = es + o4[1], *p2 = es + o4[2], *p3 = es + o4[3];
    int n0 = (cols + 3) >> 2, n1 = (cols + 2) >> 2, n2 = (cols + 1) >> 2, n3 = cols >> 2, q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0; int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    while (q0 < n0 && q1 < n1 && q2 < n2 && q3 < n3) {
        RF(b0, c0, p0) RF(b1, c1, p1) RF(b2, c2, p2) RF(b3, c3, p3)
        LK(b0, c0, q0, e0) LK(b1, c1, q1, e1) LK(b2, c2, q2, e2) LK(b3, c3, q3, e3)
        LK(b0, c0, q0, e0) LK(b1, c1, q1, e1) LK(b2, c2, q2, e2) LK(b3, c3, q3, e3)
    }
    while (q0 < n0) { RF(b0, c0, p0) LK(b0, c0, q0, e0) }
    while (q1 < n1) { RF(b1, c1, p1) LK(b1, c1, q1, e1) }
    while (q2 < n2) { RF(b2, c2, p2) LK(b2, c2, q2, e2) }
    while (q3 < n3) { RF(b3, c3, p3) LK(b3, c3, q3, e3) }
}
#undef LK
#undef RF
static inline __m256 asm8(__m128i e8, const uint8_t *sm) {
    __m128i s8 = _mm_loadl_epi64((const __m128i *)sm); __m256i e32 = _mm256_cvtepu8_epi32(e8), s32 = _mm256_cvtepu8_epi32(s8);
    __m256i bits = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(s32, _mm256_set1_epi32(0x80)), 24), _mm256_slli_epi32(e32, 23)),
                                   _mm256_slli_epi32(_mm256_and_si256(s32, _mm256_set1_epi32(0x7F)), 16));
    return _mm256_castsi256_ps(bits);
}
/* exponents of weights j..j+15 (element i..i+3 of each stream array), in weight order */
static inline void ev16(const uint8_t *e0, const uint8_t *e1, const uint8_t *e2, const uint8_t *e3, int i, __m128i *lo, __m128i *hi) {
    int a, b, c, d; memcpy(&a, e0 + i, 4); memcpy(&b, e1 + i, 4); memcpy(&c, e2 + i, 4); memcpy(&d, e3 + i, 4);
    __m128i ab = _mm_unpacklo_epi8(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b)), cd = _mm_unpacklo_epi8(_mm_cvtsi32_si128(c), _mm_cvtsi32_si128(d));
    __m128i v = _mm_unpacklo_epi16(ab, cd); *lo = v; *hi = _mm_srli_si128(v, 8);
}
/* one row dot x, same summation order as dot_row (8 lanes, mul then add, tail into lane 0, same final tree) => bit-identical */
static inline float dot_decode3(const uint32_t *l3, const uint8_t *es, const uint32_t *o4, const uint8_t *sm, int cols, const float *x, uint8_t *eb) {
    int cap = (cols >> 2) + 16; uint8_t *e0 = eb, *e1 = eb + cap, *e2 = eb + 2 * cap, *e3 = eb + 3 * cap;
    dec4(l3, es, o4, cols, e0, e1, e2, e3);
    __m256 acc = _mm256_setzero_ps(); int j = 0;
    for (; j + 16 <= cols; j += 16) {
        __m128i lo, hi; ev16(e0, e1, e2, e3, j >> 2, &lo, &hi);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(asm8(lo, sm + j), _mm256_loadu_ps(x + j)));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(asm8(hi, sm + j + 8), _mm256_loadu_ps(x + j + 8)));
    }
    if (j + 8 <= cols) { __m128i lo, hi; ev16(e0, e1, e2, e3, j >> 2, &lo, &hi); acc = _mm256_add_ps(acc, _mm256_mul_ps(asm8(lo, sm + j), _mm256_loadu_ps(x + j))); j += 8; }
    float a[8]; _mm256_storeu_ps(a, acc);
    for (; j < cols; j++) { const uint8_t *ek = (j & 3) == 0 ? e0 : (j & 3) == 1 ? e1 : (j & 3) == 2 ? e2 : e3; uint32_t e = ek[j >> 2], sv = sm[j];
        uint32_t bits = ((sv & 0x80u) << 24) | (e << 23) | ((sv & 0x7Fu) << 16); float w; memcpy(&w, &bits, 4); a[0] += w * x[j]; }
    return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
}
static inline void decode_row3(const uint32_t *l3, const uint8_t *es, const uint32_t *o4, const uint8_t *sm, int cols, float *out, uint8_t *eb) {
    int cap = (cols >> 2) + 16; uint8_t *e0 = eb, *e1 = eb + cap, *e2 = eb + 2 * cap, *e3 = eb + 3 * cap;
    dec4(l3, es, o4, cols, e0, e1, e2, e3);
    int j = 0;
    for (; j + 8 <= cols; j += 8) { __m128i lo, hi; ev16(e0, e1, e2, e3, j >> 2, &lo, &hi); _mm256_storeu_ps(out + j, asm8(lo, sm + j)); }
    for (; j < cols; j++) { const uint8_t *ek = (j & 3) == 0 ? e0 : (j & 3) == 1 ? e1 : (j & 3) == 2 ? e2 : e3; uint32_t e = ek[j >> 2], sv = sm[j];
        uint32_t bits = ((sv & 0x80u) << 24) | (e << 23) | ((sv & 0x7Fu) << 16); memcpy(&out[j], &bits, 4); }
}
#endif
/* Y[t][r] = sum_j W[r][j] * X[t][j] for n vectors; each row is decoded ONCE and reused for all n vectors */
static void mv_batch(const uint8_t *blob, const float *X, float *Y, int n) {
    uint32_t rows, cols; memcpy(&rows, blob, 4); memcpy(&cols, blob + 4, 4);
    const uint8_t *len = blob + 8; const uint32_t *off = (const uint32_t *)(blob + 264);
    const uint8_t *es = blob + 264 + 4 * (4 * (size_t)rows + 1); const uint8_t *sm = es + off[4 * (size_t)rows] + 8;
    uint16_t *lut = malloc(sizeof(uint16_t) << LMAX); build_lut(len, lut);
#if defined(__AVX2__)
    uint32_t *l3 = malloc(4 << LMAX); size_t ebcap = 4 * (((size_t)cols >> 2) + 16) + 64;
#pragma omp parallel
    { float *buf = malloc(4 * (size_t)cols + 64); uint8_t *eb = malloc(ebcap);
#pragma omp for schedule(static)
      for (int v = 0; v < (1 << LMAX); v++) l3[v] = lut3_entry(lut, (uint32_t)v);
#pragma omp for schedule(dynamic, 8) /* dynamic: a thread slowed by IO interrupts/preemption just takes fewer rows (kbench: -6..-20% under IO load) */
      for (int r = 0; r < (int)rows; r++) {
          const uint32_t *o4 = off + 4 * (size_t)r; const uint8_t *smr = sm + (size_t)r * cols;
          if (n == 1) Y[r] = dot_decode3(l3, es, o4, smr, (int)cols, X, eb);
          else { decode_row3(l3, es, o4, smr, (int)cols, buf, eb); for (int t = 0; t < n; t++) Y[(size_t)t * rows + r] = dot_row(buf, X + (size_t)t * cols, (int)cols); }
      }
      free(buf); free(eb); }
    free(l3);
#else
#pragma omp parallel
    { float *buf = malloc(4 * (size_t)cols);
#pragma omp for schedule(static)
      for (int r = 0; r < (int)rows; r++) {
          decode_row(lut, es, off + 4 * (size_t)r, sm + (size_t)r * cols, (int)cols, buf);
          for (int t = 0; t < n; t++) Y[(size_t)t * rows + r] = dot_row(buf, X + (size_t)t * cols, (int)cols);
      }
      free(buf); }
#endif
    free(lut);
}
static int verify_blob(const uint8_t *blob, const uint16_t *w) {
    uint32_t rows, cols; memcpy(&rows, blob, 4); memcpy(&cols, blob + 4, 4);
    const uint8_t *len = blob + 8; const uint32_t *off = (const uint32_t *)(blob + 264);
    const uint8_t *es = blob + 264 + 4 * (4 * (size_t)rows + 1); const uint8_t *sm = es + off[4 * (size_t)rows] + 8;
    uint16_t *lut = malloc(sizeof(uint16_t) << LMAX); build_lut(len, lut);
    float *a = malloc(4 * (size_t)cols), *b = malloc(4 * (size_t)cols); int ok = 1;
    for (uint32_t r = 0; r < rows && ok; r++) {
        decode_row(lut, es, off + 4 * (size_t)r, sm + (size_t)r * cols, (int)cols, a); raw_row(w + (size_t)r * cols, (int)cols, b);
        if (memcmp(a, b, 4 * (size_t)cols)) ok = 0;
    }
    free(lut); free(a); free(b); return ok;
}
static uint64_t hash64(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ull; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); h ^= w; h *= 1099511628211ull; h ^= h >> 29; }
    for (; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* ---------- model config / vault index ---------- */
typedef struct { int hid, inter, nl, nh, nkv, hd, vocab, tie, eos, eos2; double eps, theta; } Cfg;
static int cfg_kv(Cfg *c, const char *k, double v) {
    if (!strcmp(k, "hidden")) c->hid = (int)v; else if (!strcmp(k, "inter")) c->inter = (int)v;
    else if (!strcmp(k, "layers")) c->nl = (int)v; else if (!strcmp(k, "heads")) c->nh = (int)v;
    else if (!strcmp(k, "kv_heads")) c->nkv = (int)v; else if (!strcmp(k, "head_dim")) c->hd = (int)v;
    else if (!strcmp(k, "vocab")) c->vocab = (int)v; else if (!strcmp(k, "tie")) c->tie = (int)v;
    else if (!strcmp(k, "eos")) c->eos = (int)v; else if (!strcmp(k, "eos2")) c->eos2 = (int)v;
    else if (!strcmp(k, "eps")) c->eps = v; else if (!strcmp(k, "theta")) c->theta = v; else return 0;
    return 1;
}
typedef struct { uint64_t off, size, padded, raw, hash; } Stream;
static void path_join(char *o, size_t n, const char *a, const char *b) { snprintf(o, n, "%s/%s", a, b); }

/* ---------- safetensors reader ---------- */
typedef struct { char name[192]; int dt, nd; int64_t shape[4]; uint64_t b, e; int sh; } Tens;
typedef struct { FILE *f; uint64_t data0; } Shard;
static const char *skipws(const char *p) { while (*p && isspace((unsigned char)*p)) p++; return p; }
static const char *parse_str(const char *p, char *out, size_t cap) {
    size_t n = 0; p++;
    while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; if (n + 1 < cap) out[n++] = *p; p++; }
    out[n] = 0; return *p ? p + 1 : p;
}
static const char *skip_val(const char *p) {
    p = skipws(p);
    if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; p++; } return *p ? p + 1 : p; }
    if (*p == '{' || *p == '[') { int d = 0;
        do { if (*p == '"') { p = skip_val(p); continue; } if (*p == '{' || *p == '[') d++; else if (*p == '}' || *p == ']') d--; p++; } while (*p && d > 0);
        return p; }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}
static const char *parse_tensor(const char *p, Tens *t) {
    p = skipws(p); p++;
    for (;;) {
        p = skipws(p); if (!*p) break; if (*p == '}') { p++; break; } if (*p == ',') { p++; continue; }
        char key[32]; p = parse_str(p, key, sizeof key); p = skipws(p); if (*p == ':') p++; p = skipws(p);
        if (!strcmp(key, "dtype")) { char v[16]; p = parse_str(p, v, sizeof v); t->dt = !strcmp(v, "BF16") ? 0 : !strcmp(v, "F16") ? 1 : !strcmp(v, "F32") ? 2 : 9; }
        else if (!strcmp(key, "shape")) { p++; t->nd = 0;
            for (;;) { p = skipws(p); if (!*p || *p == ']') { if (*p) p++; break; } if (*p == ',') { p++; continue; }
                char *e; long long v = strtoll(p, &e, 10); if (e == p) break; if (t->nd < 4) t->shape[t->nd++] = v; p = e; } }
        else if (!strcmp(key, "data_offsets")) { char *e; p++; p = skipws(p); t->b = strtoull(p, &e, 10); p = skipws(e); if (*p == ',') p++;
            p = skipws(p); t->e = strtoull(p, &e, 10); p = skipws(e); if (*p == ']') p++; }
        else p = skip_val(p);
    }
    return p;
}
static int parse_header(const char *s, int sh, Tens *arr, int *n, int max) {
    const char *p = skipws(s); if (*p != '{') return -1; p++;
    for (;;) {
        p = skipws(p); if (!*p || *p == '}') break; if (*p == ',') { p++; continue; }
        char key[192]; p = parse_str(p, key, sizeof key); p = skipws(p); if (*p == ':') p++; p = skipws(p);
        if (!strcmp(key, "__metadata__")) { p = skip_val(p); continue; }
        if (*n >= max) return -2;
        Tens *t = &arr[*n]; memset(t, 0, sizeof *t); strncpy(t->name, key, 191); t->sh = sh;
        p = parse_tensor(p, t); (*n)++;
    }
    return 0;
}

/* ---------- pack ---------- */
static const char *KIND[8] = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed"};
static uint8_t *make_super(uint8_t **subs, size_t *szs, int nsub, size_t *out) {
    size_t off[8], pos = 128;
    for (int i = 0; i < nsub; i++) { pos = (pos + 63) & ~(size_t)63; off[i] = pos; pos += szs[i]; }
    uint8_t *b = calloc(pos, 1); uint32_t magic = 0x534C5657, ns = (uint32_t)nsub;
    if (!b) die("out of memory (super blob)");
    memcpy(b, &magic, 4); memcpy(b + 4, &ns, 4);
    for (int i = 0; i < nsub; i++) { uint64_t o = off[i]; memcpy(b + 8 + 8 * i, &o, 8); memcpy(b + off[i], subs[i], szs[i]); }
    *out = pos; return b;
}
static int cmd_pack(const char *mdir, const char *odir) {
    char p[600], line[600]; Cfg c; memset(&c, 0, sizeof c); c.eos = 151645; c.eos2 = 151643;
    path_join(p, sizeof p, mdir, "config.txt"); FILE *f = fopen(p, "r"); if (!f) die("config.txt not found in model dir (run prepare.py)");
    while (fgets(line, sizeof line, f)) { char k[64]; double v; if (sscanf(line, "%63s %lf", k, &v) == 2) cfg_kv(&c, k, v); }
    fclose(f);
    if (!c.hid || !c.nl || !c.nh || !c.nkv || !c.hd || !c.vocab || !c.inter) die("config.txt incomplete");
    path_join(p, sizeof p, mdir, "shards.txt"); f = fopen(p, "r"); if (!f) die("shards.txt not found (run prepare.py)");
    Shard sh[64]; int nsh = 0; Tens *T = malloc(sizeof(Tens) * 4096); int nt = 0;
    while (fgets(line, sizeof line, f) && nsh < 64) {
        size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r' || line[L - 1] == ' ')) line[--L] = 0;
        if (!L) continue; path_join(p, sizeof p, mdir, line);
        sh[nsh].f = fopen(p, "rb"); if (!sh[nsh].f) { fprintf(stderr, "cannot open %s\n", p); return 1; }
        uint64_t hl; if (fread(&hl, 8, 1, sh[nsh].f) != 1 || hl > (1u << 28)) die("bad safetensors header");
        char *hdr = malloc(hl + 1); if (fread(hdr, 1, hl, sh[nsh].f) != hl) die("short safetensors header"); hdr[hl] = 0;
        sh[nsh].data0 = 8 + hl; if (parse_header(hdr, nsh, T, &nt, 4096)) die("cannot parse safetensors header"); free(hdr); nsh++;
    }
    fclose(f); printf("shards=%d tensors=%d\n", nsh, nt);
    path_join(p, sizeof p, odir, "vault.bin"); FILE *vf = fopen(p, "wb"); if (!vf) die("cannot create vault.bin (does the vault dir exist?)");
    Stream *st = calloc(c.nl + 2, sizeof(Stream)); int nst = 0;
    int hid = c.hid, qd = c.nh * c.hd, kvd = c.nkv * c.hd;
    float *res = malloc(sizeof(float) * ((size_t)c.nl * (2 * hid + 2 * c.hd) + hid)); size_t rp = 0;
    double kraw[8] = {0}, kcomp[8] = {0}, kent[8] = {0}; double t0 = now();
    int rowsv[7] = {qd, kvd, kvd, hid, c.inter, c.inter, hid}, colsv[7] = {hid, hid, hid, qd, hid, hid, c.inter};
    const char *sfx[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
#define FINDT(var, nm) do { var = NULL; for (int _i = 0; _i < nt; _i++) if (!strcmp(T[_i].name, nm)) { var = &T[_i]; break; } if (!var) { fprintf(stderr, "missing tensor %s\n", nm); return 1; } } while (0)
#define LOADT(w, tt, cnt) do { if ((tt)->dt != 0) { fprintf(stderr, "%s is not BF16 (this engine packs bf16 only)\n", (tt)->name); return 1; } \
        size_t _n = ((tt)->e - (tt)->b) / 2; if (_n != (size_t)(cnt)) { fprintf(stderr, "%s: size mismatch\n", (tt)->name); return 1; } \
        w = malloc(_n * 2); FSEEK64(sh[(tt)->sh].f, sh[(tt)->sh].data0 + (tt)->b); if (fread(w, 2, _n, sh[(tt)->sh].f) != _n) die("short read of tensor"); } while (0)
    for (int l = 0; l <= c.nl + (c.tie ? 0 : 1); l++) {
        uint8_t *subs[8]; size_t szs[8]; int nsub = 0;
        if (l < c.nl) {
            char nm[256]; Tens *tt; uint16_t *w;
            for (int i = 0; i < 7; i++) {
                snprintf(nm, sizeof nm, "model.layers.%d.%s.weight", l, sfx[i]); FINDT(tt, nm); LOADT(w, tt, (size_t)rowsv[i] * colsv[i]);
                subs[nsub] = compress_layer(w, rowsv[i], colsv[i], &szs[nsub], &kent[i]);
                if (!verify_blob(subs[nsub], w)) die("VERIFY FAILED: compressed matrix does not decode bit-exact");
                kraw[i] += 2.0 * rowsv[i] * colsv[i]; kcomp[i] += szs[nsub]; nsub++; free(w);
            }
            const char *nn[4] = {"input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"}; int ln[4] = {hid, hid, c.hd, c.hd};
            for (int i = 0; i < 4; i++) { snprintf(nm, sizeof nm, "model.layers.%d.%s.weight", l, nn[i]); FINDT(tt, nm); LOADT(w, tt, ln[i]);
                for (int j = 0; j < ln[i]; j++) { uint32_t b = (uint32_t)w[j] << 16; memcpy(&res[rp++], &b, 4); } free(w); }
        } else {
            const char *nm = (l == c.nl) ? "model.embed_tokens.weight" : "lm_head.weight"; Tens *tt; uint16_t *w;
            FINDT(tt, nm); LOADT(w, tt, (size_t)c.vocab * hid);
            subs[0] = compress_layer(w, c.vocab, hid, &szs[0], &kent[7]); nsub = 1;
            if (!verify_blob(subs[0], w)) die("VERIFY FAILED: embedding does not decode bit-exact");
            kraw[7] += 2.0 * c.vocab * hid; kcomp[7] += szs[0]; free(w);
        }
        size_t tot; uint8_t *sb = make_super(subs, szs, nsub, &tot); for (int i = 0; i < nsub; i++) free(subs[i]);
        size_t padded = (tot + SECT - 1) & ~(size_t)(SECT - 1); uint8_t *pb = calloc(padded, 1); memcpy(pb, sb, tot); free(sb);
        st[nst].off = FTELL64(vf); st[nst].size = tot; st[nst].padded = padded; st[nst].hash = hash64(pb, padded);
        st[nst].raw = 0; for (int i = 0; i < nsub && l < c.nl; i++) st[nst].raw += 2ull * rowsv[i] * colsv[i];
        if (l >= c.nl) st[nst].raw = 2ull * c.vocab * hid;
        if (fwrite(pb, 1, padded, vf) != padded) die("write error (disk full?)"); free(pb); nst++;
        printf("\rpacked stream %d/%d", nst, c.nl + 1 + (c.tie ? 0 : 1)); fflush(stdout);
    }
    {   Tens *tt; uint16_t *w; FINDT(tt, "model.norm.weight"); LOADT(w, tt, hid);
        for (int j = 0; j < hid; j++) { uint32_t b = (uint32_t)w[j] << 16; memcpy(&res[rp++], &b, 4); } free(w); }
    fclose(vf); printf("\n");
    path_join(p, sizeof p, odir, "resident.bin"); f = fopen(p, "wb"); fwrite(res, 4, rp, f); fclose(f);
    path_join(p, sizeof p, odir, "vault.idx"); f = fopen(p, "w");
    fprintf(f, "WVAULT 3\nhidden %d\ninter %d\nlayers %d\nheads %d\nkv_heads %d\nhead_dim %d\nvocab %d\ntie %d\neos %d\neos2 %d\neps %.10g\ntheta %.10g\nnstreams %d\n",
            c.hid, c.inter, c.nl, c.nh, c.nkv, c.hd, c.vocab, c.tie, c.eos, c.eos2, c.eps, c.theta, nst);
    uint64_t sum_raw = 0, sum_c = 0;
    for (int i = 0; i < nst; i++) { fprintf(f, "stream %d %llu %llu %llu %llu %llx\n", i, (unsigned long long)st[i].off, (unsigned long long)st[i].size,
                                            (unsigned long long)st[i].padded, (unsigned long long)st[i].raw, (unsigned long long)st[i].hash); sum_raw += st[i].raw; sum_c += st[i].size; }
    fclose(f);
    char ts[64]; stamp(ts, sizeof ts);
    lg("=== PACK %s ===", ts); lg("[pack] model_dir=%s vault_dir=%s time_s=%.1f", mdir, odir, now() - t0);
    lg("[pack] streams=%d raw_bytes=%llu stored_bytes=%llu ratio=%.4f (all matrices verified bit-exact)", nst, (unsigned long long)sum_raw, (unsigned long long)sum_c, (double)sum_c / (double)sum_raw);
    for (int i = 0; i < 8; i++) if (kraw[i] > 0) lg("[pack] kind=%-9s raw_MB=%.1f stored_MB=%.1f ratio=%.4f exponent_entropy_bits=%.3f", KIND[i], kraw[i] / 1e6, kcomp[i] / 1e6, kcomp[i] / kraw[i], kent[i] / (kraw[i] / 2));
    printf("vault: %.2f GB raw -> %.2f GB stored (ratio %.4f), all matrices bit-exact\n", sum_raw / 1e9, sum_c / 1e9, (double)sum_c / (double)sum_raw);
    return 0;
}

/* ---------- engine ---------- */
typedef struct Eng Eng;
typedef struct { Eng *E; int s; uint8_t *buf; double read_ms; } Ld;
typedef struct { int s; double read, wait, comp; uint64_t bytes; } SRec;
struct Eng {
    Cfg c; int ctx, B, io_threads, direct, detail; size_t io_chunk, bufsz;
    Stream *st; int nst, lm_s; RFile vf, ef; char vdir[512];
    uint8_t *resident[40]; int first_streamed; double resident_mb, resident_used;
    uint8_t *bufs[2]; Thr pth; Ld ld; int pend_s, pend_buf;
    float *kc, *vc, *cosT, *sinT, *ln_in, *ln_post, *qn, *kn, *fnrm;
    float *H, *XN, *Q, *K, *V, *ATT, *O, *G, *logits;
    uint32_t *emb_off, emb_rows, emb_cols; uint16_t *emb_lut; uint8_t *emb_es, *emb_sm; uint64_t emb_es0, emb_sm0;
    int pos; SRec rec[40]; double t_attn, t_embed;
    void (*gfwd)(Eng *, const int *, int, int);
};
static void loader_fn(void *a) {
    Ld *l = a; Eng *E = l->E; Stream *s = &E->st[l->s]; double t0 = now();
    if (read_stream(&E->vf, l->buf, s->off, (size_t)s->padded, E->io_threads, E->io_chunk)) die("vault read failed");
    l->read_ms = (now() - t0) * 1e3;
}
static int eng_open(Eng *E, const char *vdir) {
    char p[600], line[600]; memset(&E->c, 0, sizeof E->c); snprintf(E->vdir, sizeof E->vdir, "%s", vdir);
    path_join(p, sizeof p, vdir, "vault.idx"); FILE *f = fopen(p, "r"); if (!f) { fprintf(stderr, "no vault.idx in %s (run pack)\n", vdir); return -1; }
    E->nst = 0; E->st = calloc(64, sizeof(Stream));
    while (fgets(line, sizeof line, f)) {
        char k[64]; double v; unsigned long long a, b, c2, d, h; int idx;
        if (!strncmp(line, "WVAULT", 6)) { if (atoi(line + 7) != 3) { fprintf(stderr, "vault format %d is old: pack it again into a new folder (upgrade_v3.bat)\n", atoi(line + 7)); return -1; } continue; }
        if (sscanf(line, "stream %d %llu %llu %llu %llu %llx", &idx, &a, &b, &c2, &d, &h) == 6) { if (idx < 64) { E->st[idx].off = a; E->st[idx].size = b; E->st[idx].padded = c2; E->st[idx].raw = d; E->st[idx].hash = h; if (idx + 1 > E->nst) E->nst = idx + 1; } }
        else if (sscanf(line, "%63s %lf", k, &v) == 2) cfg_kv(&E->c, k, v);
    }
    fclose(f); E->lm_s = E->c.tie ? E->c.nl : E->c.nl + 1;
    path_join(p, sizeof p, vdir, "vault.bin");
    if (rf_open(&E->vf, p, E->direct) || rf_open(&E->ef, p, 0)) { fprintf(stderr, "cannot open %s\n", p); return -1; }
    E->pend_s = -1; return 0;
}
/* pin streams in RAM (lm_head first, then layers 0..), then size the two stream buffers for what is still streamed */
static void eng_buffers(Eng *E, double resident_mb) {
    double budget = resident_mb * 1e6; int order[40], no = 0; E->resident_used = 0;
    if (E->nst > 40) die("too many streams");
    if (resident_mb > 0) {
        double mt, ma, mw, mp; mem_info(&mt, &ma, &mw, &mp); double lim = (ma - 1500.0) * 1048576.0; if (lim < 0) lim = 0;
        if (budget > lim) { lg("[resident] requested %.0f MB exceeds free RAM minus 1.5 GB reserve; clamped to %.0f MB", resident_mb, lim / 1e6); budget = lim; }
    }
    order[no++] = E->lm_s;   /* then layers in van der Corput order: any prefix is spread evenly through the network */
    { int nb = 1; while ((1 << nb) < E->c.nl) nb++;
      for (int m = 0; m < (1 << nb); m++) { int r = 0; for (int q = 0; q < nb; q++) if (m & (1 << q)) r |= 1 << (nb - 1 - q); if (r < E->c.nl) order[no++] = r; } }
    for (int k = 0; k < no && resident_mb > 0; k++) {
        int s = order[k]; Stream *st = &E->st[s];
        if (E->resident_used + (double)st->padded > budget) continue;
        uint8_t *b = aalloc((size_t)st->padded); if (!b) continue;
        if (read_stream(&E->vf, b, st->off, (size_t)st->padded, E->io_threads, E->io_chunk)) die("vault read failed while pinning");
        E->resident[s] = b; E->resident_used += (double)st->padded;
    }
    E->first_streamed = -1; for (int i = 0; i < E->c.nl; i++) if (!E->resident[i]) { E->first_streamed = i; break; }
    if (E->first_streamed < 0 && !E->resident[E->lm_s]) E->first_streamed = E->lm_s;
    E->bufsz = 0; for (int i = 0; i < E->nst; i++) if (!E->resident[i] && E->st[i].padded > E->bufsz) E->bufsz = (size_t)E->st[i].padded;
    if (!E->bufsz) E->bufsz = SECT;
    E->bufs[0] = aalloc(E->bufsz); E->bufs[1] = aalloc(E->bufsz); if (!E->bufs[0] || !E->bufs[1]) die("cannot allocate stream buffers");
}
static void embed_init(Eng *E) {
    uint8_t hdr[264]; uint64_t o0; Stream *s = &E->st[E->c.nl];
    if (rf_pread(&E->ef, &o0, s->off + 8, 8)) die("cannot read embedding header");
    uint64_t sub = s->off + o0; if (rf_pread(&E->ef, hdr, sub, 264)) die("cannot read embedding sub-header");
    memcpy(&E->emb_rows, hdr, 4); memcpy(&E->emb_cols, hdr + 4, 4);
    size_t no4 = 4 * (size_t)E->emb_rows + 1;
    E->emb_off = malloc(4 * no4); if (rf_pread(&E->ef, E->emb_off, sub + 264, 4 * no4)) die("cannot read embedding row table");
    E->emb_lut = malloc(sizeof(uint16_t) << LMAX); build_lut(hdr + 8, E->emb_lut);
    E->emb_es0 = sub + 264 + 4 * no4; E->emb_sm0 = E->emb_es0 + E->emb_off[4 * (size_t)E->emb_rows] + 8;
    E->emb_es = malloc((size_t)E->emb_cols * 2 + 64); E->emb_sm = malloc(E->emb_cols + 8);
}
static void embed_row(Eng *E, int tok, float *out) {
    if (tok < 0 || (uint32_t)tok >= E->emb_rows) die("token id out of range");
    const uint32_t *o = E->emb_off + 4 * (size_t)tok; uint32_t a = o[0], b = o[4], o4[4] = {0, o[1] - a, o[2] - a, o[3] - a};
    if (rf_pread(&E->ef, E->emb_es, E->emb_es0 + a, (size_t)(b - a) + 8)) die("embedding read failed");
    if (rf_pread(&E->ef, E->emb_sm, E->emb_sm0 + (uint64_t)tok * E->emb_cols, E->emb_cols)) die("embedding read failed");
    decode_row(E->emb_lut, E->emb_es, o4, E->emb_sm, (int)E->emb_cols, out);
}
static void eng_init_model(Eng *E, int ctx) {
    Cfg *c = &E->c; E->ctx = ctx; char p[600]; int hid = c->hid, qd = c->nh * c->hd, kvd = c->nkv * c->hd, B = E->B;
    size_t nres = (size_t)c->nl * (2 * hid + 2 * c->hd) + hid; float *res = malloc(4 * nres);
    path_join(p, sizeof p, E->vdir, "resident.bin"); FILE *f = fopen(p, "rb"); if (!f || fread(res, 4, nres, f) != nres) die("resident.bin missing or wrong size"); fclose(f);
    E->ln_in = malloc(4 * (size_t)c->nl * hid); E->ln_post = malloc(4 * (size_t)c->nl * hid); E->qn = malloc(4 * (size_t)c->nl * c->hd); E->kn = malloc(4 * (size_t)c->nl * c->hd); E->fnrm = malloc(4 * (size_t)hid);
    size_t rp = 0;
    for (int l = 0; l < c->nl; l++) { memcpy(E->ln_in + (size_t)l * hid, res + rp, 4 * hid); rp += hid; memcpy(E->ln_post + (size_t)l * hid, res + rp, 4 * hid); rp += hid;
        memcpy(E->qn + (size_t)l * c->hd, res + rp, 4 * c->hd); rp += c->hd; memcpy(E->kn + (size_t)l * c->hd, res + rp, 4 * c->hd); rp += c->hd; }
    memcpy(E->fnrm, res + rp, 4 * hid); free(res);
    E->kc = calloc((size_t)c->nl * ctx * kvd, 4); E->vc = calloc((size_t)c->nl * ctx * kvd, 4);
    int half = c->hd / 2; E->cosT = malloc(4 * (size_t)ctx * half); E->sinT = malloc(4 * (size_t)ctx * half);
    for (int pos = 0; pos < ctx; pos++) for (int i = 0; i < half; i++) { double fr = pow(c->theta, -2.0 * i / c->hd), a = pos * fr;
        E->cosT[(size_t)pos * half + i] = (float)cos(a); E->sinT[(size_t)pos * half + i] = (float)sin(a); }
    E->H = malloc(4 * (size_t)B * hid); E->XN = malloc(4 * (size_t)B * hid); E->Q = malloc(4 * (size_t)B * qd); E->K = malloc(4 * (size_t)B * kvd); E->V = malloc(4 * (size_t)B * kvd);
    E->ATT = malloc(4 * (size_t)B * qd); E->O = malloc(4 * (size_t)B * hid); E->G = malloc(4 * (size_t)B * c->inter); float *U = malloc(4 * (size_t)B * c->inter); E->logits = malloc(4 * (size_t)c->vocab);
    E->t_attn = 0; E->pos = 0; (void)U; free(U);
}
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + eps); for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * inv);
}
static void rope_head(float *v, const float *cs, const float *sn, int half) {
    for (int i = 0; i < half; i++) { float x0 = v[i], x1 = v[i + half]; v[i] = x0 * cs[i] - x1 * sn[i]; v[i + half] = x1 * cs[i] + x0 * sn[i]; }
}
static void layer_compute(Eng *E, int l, const uint8_t *blob, int n, int pos0) {
    Cfg *c = &E->c; int hid = c->hid, hd = c->hd, nh = c->nh, nkv = c->nkv, qd = nh * hd, kvd = nkv * hd, inter = c->inter, half = hd / 2; float eps = (float)c->eps;
    const uint8_t *sub[7]; for (int i = 0; i < 7; i++) { uint64_t o; memcpy(&o, blob + 8 + 8 * i, 8); sub[i] = blob + o; }
    const float *lni = E->ln_in + (size_t)l * hid, *lnp = E->ln_post + (size_t)l * hid, *qn = E->qn + (size_t)l * hd, *kn = E->kn + (size_t)l * hd;
    for (int t = 0; t < n; t++) rmsnorm(E->XN + (size_t)t * hid, E->H + (size_t)t * hid, lni, hid, eps);
    mv_batch(sub[0], E->XN, E->Q, n); mv_batch(sub[1], E->XN, E->K, n); mv_batch(sub[2], E->XN, E->V, n);
    double ta = now(); float *kcl = E->kc + (size_t)l * E->ctx * kvd, *vcl = E->vc + (size_t)l * E->ctx * kvd; int grp = nh / nkv; float scale = 1.0f / sqrtf((float)hd);
    for (int t = 0; t < n; t++) {
        int p = pos0 + t; float *q = E->Q + (size_t)t * qd, *k = E->K + (size_t)t * kvd, *v = E->V + (size_t)t * kvd;
        for (int h = 0; h < nh; h++) rmsnorm(q + h * hd, q + h * hd, qn, hd, eps);
        for (int h = 0; h < nkv; h++) rmsnorm(k + h * hd, k + h * hd, kn, hd, eps);
        const float *cs = E->cosT + (size_t)p * half, *sn = E->sinT + (size_t)p * half;
        for (int h = 0; h < nh; h++) rope_head(q + h * hd, cs, sn, half);
        for (int h = 0; h < nkv; h++) rope_head(k + h * hd, cs, sn, half);
        memcpy(kcl + (size_t)p * kvd, k, 4 * (size_t)kvd); memcpy(vcl + (size_t)p * kvd, v, 4 * (size_t)kvd);
#pragma omp parallel for schedule(static)
        for (int h = 0; h < nh; h++) {
            float *sc = malloc(4 * (size_t)(p + 1)); const float *qh = q + h * hd; int kvh = h / grp; float mx = -1e30f;
            for (int tt = 0; tt <= p; tt++) { const float *kk = kcl + (size_t)tt * kvd + kvh * hd; float s = 0; for (int d = 0; d < hd; d++) s += qh[d] * kk[d]; s *= scale; sc[tt] = s; if (s > mx) mx = s; }
            float sum = 0; for (int tt = 0; tt <= p; tt++) { sc[tt] = expf(sc[tt] - mx); sum += sc[tt]; }
            float inv = 1.0f / sum; float *o = E->ATT + (size_t)t * qd + h * hd; for (int d = 0; d < hd; d++) o[d] = 0;
            for (int tt = 0; tt <= p; tt++) { const float *vv = vcl + (size_t)tt * kvd + kvh * hd; float w = sc[tt] * inv; for (int d = 0; d < hd; d++) o[d] += w * vv[d]; }
            free(sc);
        }
    }
    E->t_attn += (now() - ta) * 1e3;
    mv_batch(sub[3], E->ATT, E->O, n); for (size_t i = 0; i < (size_t)n * hid; i++) E->H[i] += E->O[i];
    for (int t = 0; t < n; t++) rmsnorm(E->XN + (size_t)t * hid, E->H + (size_t)t * hid, lnp, hid, eps);
    float *U = malloc(4 * (size_t)n * inter);
    mv_batch(sub[4], E->XN, E->G, n); mv_batch(sub[5], E->XN, U, n);
    for (size_t i = 0; i < (size_t)n * inter; i++) { float g = E->G[i]; E->G[i] = g / (1.0f + expf(-g)) * U[i]; }
    free(U); mv_batch(sub[6], E->G, E->O, n); for (size_t i = 0; i < (size_t)n * hid; i++) E->H[i] += E->O[i];
}
static void forward(Eng *E, const int *tok, int n, int want_logits) {
    if (E->gfwd) { E->gfwd(E, tok, n, want_logits); return; }
    Cfg *c = &E->c; int pos0 = E->pos; double t0 = now();
    for (int t = 0; t < n; t++) embed_row(E, tok[t], E->H + (size_t)t * c->hid);
    E->t_embed = (now() - t0) * 1e3; E->t_attn = 0;
    int ns = c->nl + (want_logits ? 1 : 0);
    for (int i = 0; i < ns; i++) {
        int s = (i < c->nl) ? i : E->lm_s; const uint8_t *blob; int ub = -1; double tw = now();
        E->rec[i].read = 0; E->rec[i].bytes = 0;
        if (E->resident[s]) blob = E->resident[s];
        else {
            if (E->pend_s == s) { thr_join(&E->pth); ub = E->pend_buf; E->pend_s = -1; }
            else { if (E->pend_s >= 0) { thr_join(&E->pth); E->pend_s = -1; } ub = 0; E->ld.E = E; E->ld.s = s; E->ld.buf = E->bufs[0]; loader_fn(&E->ld); }
            E->rec[i].read = E->ld.read_ms; E->rec[i].bytes = E->st[s].padded; blob = E->bufs[ub];
        }
        E->rec[i].wait = (now() - tw) * 1e3; E->rec[i].s = s;
        if (E->pend_s < 0) {
            int nx = -1;
            for (int j = i + 1; j < ns; j++) { int sj = (j < c->nl) ? j : E->lm_s; if (!E->resident[sj]) { nx = sj; break; } }
            if (nx < 0) nx = E->first_streamed;
            if (nx >= 0) { int other = (ub < 0) ? 0 : (ub ^ 1); E->ld.E = E; E->ld.s = nx; E->ld.buf = E->bufs[other]; E->pend_s = nx; E->pend_buf = other; thr_start(&E->pth, (void (*)(void *))loader_fn, &E->ld); }
        }
        double tc = now();
        if (i < c->nl) layer_compute(E, i, blob, n, pos0);
        else { uint64_t o; memcpy(&o, blob + 8, 8); rmsnorm(E->XN, E->H + (size_t)(n - 1) * c->hid, E->fnrm, c->hid, (float)c->eps); mv_batch(blob + o, E->XN, E->logits, 1); }
        E->rec[i].comp = (now() - tc) * 1e3;
    }
    E->pos += n;
}

/* ---------- sampling ---------- */
static uint64_t rng_next(uint64_t *s) { uint64_t x = *s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; *s = x; return x * 2685821657736338717ull; }
static float rng_f(uint64_t *s) { return (float)(rng_next(s) >> 40) / 16777216.0f; }
static int sample_tok(const float *lgt, int V, float temp, float topp, int topk, uint64_t *rng) {
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (lgt[i] > lgt[b]) b = i; return b; }
    if (topk < 1 || topk > 128) topk = 128; if (topk > V) topk = V;
    int idx[128]; float val[128]; int k = 0;
    for (int i = 0; i < V; i++) { float v = lgt[i]; if (k == topk && v <= val[k - 1]) continue; int j = (k < topk) ? k++ : k - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; } val[j] = v; idx[j] = i; }
    float p[128], sum = 0, mx = val[0]; for (int j = 0; j < k; j++) { p[j] = expf((val[j] - mx) / temp); sum += p[j]; }
    for (int j = 0; j < k; j++) p[j] /= sum;
    float cum = 0; int kk = k; for (int j = 0; j < k; j++) { cum += p[j]; if (cum >= topp) { kk = j + 1; break; } }
    float s2 = 0; for (int j = 0; j < kk; j++) s2 += p[j]; float r = rng_f(rng) * s2, c2 = 0;
    for (int j = 0; j < kk; j++) { c2 += p[j]; if (r <= c2) return idx[j]; }
    return idx[kk - 1];
}
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double pct(double *a, int n, double q) { if (!n) return 0; double *t = malloc(8 * (size_t)n); memcpy(t, a, 8 * (size_t)n); qsort(t, n, 8, cmpd); double v = t[(int)((n - 1) * q)]; free(t); return v; }
static void rec_sums(Eng *E, int ns, double *w, double *r, double *cmp, double *by) {
    *w = *r = *cmp = *by = 0; for (int i = 0; i < ns; i++) { *w += E->rec[i].wait; *r += E->rec[i].read; *cmp += E->rec[i].comp; *by += (double)E->rec[i].bytes; }
}
static void log_streams(Eng *E, int ns, const char *tag) {
    for (int i = 0; i < ns; i++) { SRec *s = &E->rec[i]; lg("[stream] %s s=%d kind=%s MB=%.1f read_ms=%.1f read_MBps=%.0f wait_ms=%.1f comp_ms=%.1f", tag, s->s, s->s < E->c.nl ? "layer" : "lm_head", s->bytes / 1e6, s->read, s->read > 0 ? s->bytes / 1e3 / s->read : 0, s->wait, s->comp); }
}

/* ---------- generation ---------- */
static int G_COUNT = 0, G_IGNORE_EOS = 0, B_TOKENS = 16, B_PROMPT = 8;
static void gen_cmd(Eng *E, int max_new, float temp, float topp, int topk, uint64_t seed, int *pt, int ntok) {
    Cfg *c = &E->c; int g = ++G_COUNT; double ram_t, ram_a0, ws0, pk0, ram_a1, ws1, pk1;
    if (E->pos + ntok + 1 > E->ctx) { printf("ERR ctx %d %d\n", E->pos, E->ctx); fflush(stdout); lg("[gen %d] REFUSED context full pos=%d new=%d ctx=%d", g, E->pos, ntok, E->ctx); return; }
    mem_info(&ram_t, &ram_a0, &ws0, &pk0); uint64_t rng = seed ? seed : 88172645463325252ull;
    lg("[gen %d] begin prompt_tokens=%d pos_before=%d max_new=%d temp=%.2f top_p=%.2f top_k=%d ram_avail_MB=%.0f", g, ntok, E->pos, max_new, temp, topp, topk, ram_a0);
    double tp0 = now(); int calls = 0; double pw = 0, pr = 0, pc = 0, pb = 0;
    for (int i = 0; i < ntok;) {
        int n = ntok - i; if (n > E->B) n = E->B; int last = (i + n == ntok); double tc0 = now();
        forward(E, pt + i, n, last); int ns = c->nl + (last ? 1 : 0); double w, r, cm, by; rec_sums(E, ns, &w, &r, &cm, &by); pw += w; pr += r; pc += cm; pb += by; calls++;
        lg("[prefill] g=%d call=%d tokens=%d logits=%d total_ms=%.1f wait_ms=%.1f read_ms=%.1f comp_ms=%.1f attn_ms=%.1f", g, calls, n, last, (now() - tc0) * 1e3, w, r, cm, E->t_attn);
        if (calls == 1) log_streams(E, ns, "prefill1"); i += n;
    }
    double prefill_ms = (now() - tp0) * 1e3;
    double *tms = malloc(8 * (size_t)(max_new + 2)); int nd = 0; double sw = 0, sr = 0, sc = 0, sb = 0, sa = 0, sl = 0, se = 0, ss = 0, rd_max = 0;
    double lw_sum[40] = {0}, lr_sum[40] = {0}, lc_sum[40] = {0}; double *waits = malloc(8 * (size_t)(max_new + 2) * 40); int nw = 0; uint64_t rawtok = 0;
    for (int i = 0; i < c->nl; i++) rawtok += E->st[i].raw; rawtok += E->st[E->lm_s].raw;
    double ts0 = now(); int tok = sample_tok(E->logits, c->vocab, temp, topp, topk, &rng); double first_samp = (now() - ts0) * 1e3;
    int new_tok = 0; const char *reason = "max";
    for (;;) {
        printf("T %d\n", tok); fflush(stdout); new_tok++;
        if (!G_IGNORE_EOS && (tok == c->eos || tok == c->eos2)) { reason = "eos"; break; }
        if (new_tok >= max_new) { forward(E, &tok, 1, 0); reason = "max"; break; }
        if (E->pos + 1 >= E->ctx) { reason = "ctx"; break; }
        double t1 = now(); forward(E, &tok, 1, 1); double t2 = now(); int ns = c->nl + 1; double w, r, cm, by; rec_sums(E, ns, &w, &r, &cm, &by);
        tok = sample_tok(E->logits, c->vocab, temp, topp, topk, &rng); double t3 = now(); double ms = (t3 - t1) * 1e3, samp = (t3 - t2) * 1e3;
        tms[nd++] = ms; sw += w; sr += r; sc += cm; sb += by; sa += E->t_attn; sl += E->rec[ns - 1].comp; se += E->t_embed; ss += samp;
        for (int i = 0; i < ns; i++) { lw_sum[i] += E->rec[i].wait; lr_sum[i] += E->rec[i].read; lc_sum[i] += E->rec[i].comp; waits[nw++] = E->rec[i].wait; if (E->rec[i].read > 0) { double mb = E->rec[i].bytes / 1e3 / E->rec[i].read; if (mb > rd_max) rd_max = mb; } }
        lg("[tok] g=%d i=%d pos=%d total_ms=%.1f wait_ms=%.1f read_ms=%.1f read_MBps=%.0f comp_ms=%.1f attn_ms=%.1f lm_ms=%.1f embed_ms=%.2f sample_ms=%.2f", g, nd, E->pos, ms, w, r, r > 0 ? by / 1e3 / r : 0, cm, E->t_attn, E->rec[ns - 1].comp, E->t_embed, samp);
        if (nd <= E->detail) log_streams(E, ns, "decode");
    }
    double dec_ms = 0; for (int i = 0; i < nd; i++) dec_ms += tms[i];
    double mean = nd ? dec_ms / nd : 0, p50 = pct(tms, nd, 0.5), p95 = pct(tms, nd, 0.95), mx = pct(tms, nd, 1.0);
    mem_info(&ram_t, &ram_a1, &ws1, &pk1);
    lg("[gen-summary] g=%d prompt_tokens=%d prefill_ms=%.1f prefill_calls=%d prefill_wait_ms=%.1f prefill_read_ms=%.1f prefill_comp_ms=%.1f first_sample_ms=%.2f", g, ntok, prefill_ms, calls, pw, pr, pc, first_samp);
    if (nd > 0) {
        double stall = dec_ms > 0 ? 100.0 * sw / dec_ms : 0, hidden = sr > 0 ? 100.0 * (1.0 - sw / sr) : 0;
        lg("[gen-summary] g=%d decode_tokens=%d decode_ms=%.1f tok_per_s=%.3f ms_per_tok_mean=%.1f p50=%.1f p95=%.1f max=%.1f", g, nd, dec_ms, 1000.0 * nd / dec_ms, mean, p50, p95, mx);
        lg("[gen-summary] g=%d wait_ms_total=%.1f stall_pct=%.1f read_ms_total=%.1f hidden_io_pct=%.1f read_MBps_mean=%.0f read_MBps_max=%.0f comp_ms_total=%.1f attn_pct=%.1f lm_head_pct=%.1f embed_ms_total=%.1f sample_ms_total=%.1f", g, sw, stall, sr, hidden, sr > 0 ? sb / 1e3 / sr : 0, rd_max, sc, dec_ms > 0 ? 100 * sa / dec_ms : 0, dec_ms > 0 ? 100 * sl / dec_ms : 0, se, ss);
        lg("[gen-summary] g=%d stored_GBps=%.3f raw_equiv_GBps=%.3f (raw bytes/token=%.2f GB) streamed_GBps=bytes read from disk/s", g, sb / 1e6 / dec_ms, (double)rawtok * nd / 1e6 / dec_ms, rawtok / 1e9);
        lg("[gen-summary] g=%d streamed_MB_per_token=%.0f resident_MB=%.0f", g, sb / 1e6 / nd, E->resident_used / 1e6);
        const char *v = stall > 25 ? "IO-BOUND" : stall < 5 ? "COMPUTE-BOUND" : "MIXED";
        lg("[gen-summary] g=%d verdict=%s (stall_pct=%.1f of decode time spent waiting for the next stream)", g, v, stall);
        char buf[2048]; int bl = 0; bl += snprintf(buf + bl, sizeof buf - bl, "[layer-mean-wait-ms]");
        for (int i = 0; i < c->nl + 1; i++) bl += snprintf(buf + bl, sizeof buf - bl, " %d:%.1f", i, lw_sum[i] / nd); lg("%s", buf);
        bl = 0; bl += snprintf(buf + bl, sizeof buf - bl, "[layer-mean-read-ms]");
        for (int i = 0; i < c->nl + 1; i++) bl += snprintf(buf + bl, sizeof buf - bl, " %d:%.1f", i, lr_sum[i] / nd); lg("%s", buf);
        bl = 0; bl += snprintf(buf + bl, sizeof buf - bl, "[layer-mean-comp-ms]");
        for (int i = 0; i < c->nl + 1; i++) bl += snprintf(buf + bl, sizeof buf - bl, " %d:%.1f", i, lc_sum[i] / nd); lg("%s", buf);
        lg("[gen-summary] g=%d wait_ms_p95_per_stream=%.1f max=%.1f", g, pct(waits, nw, 0.95), pct(waits, nw, 1.0));
        printf("S tok_s=%.3f ms_tok=%.0f p50_ms=%.0f stall_pct=%.0f read_MBps=%.0f prefill_ms=%.0f streamed_MB_tok=%.0f resident_MB=%.0f\n", 1000.0 * nd / dec_ms, mean, p50, stall, sr > 0 ? sb / 1e3 / sr : 0, prefill_ms, sb / 1e6 / nd, E->resident_used / 1e6);
    } else printf("S tok_s=0 ms_tok=0 stall_pct=0 read_MBps=0 prefill_ms=%.0f\n", prefill_ms);
    lg("[gen-summary] g=%d new_tokens=%d reason=%s pos_after=%d ram_avail_MB_before=%.0f after=%.0f working_set_MB=%.0f peak_working_set_MB=%.0f", g, new_tok, reason, E->pos, ram_a0, ram_a1, ws1, pk1);
    printf("END %s %d %d\n", reason, new_tok, E->pos); fflush(stdout); free(tms); free(waits);
}

static void log_header(Eng *E, const char *mode, int threads, int ctx) {
    char ts[64]; stamp(ts, sizeof ts); double t, a, ws, pk; mem_info(&t, &a, &ws, &pk); Cfg *c = &E->c;
    uint64_t tot = 0, mn = ~0ull, mxs = 0; for (int i = 0; i < c->nl; i++) { tot += E->st[i].padded; if (E->st[i].padded < mn) mn = E->st[i].padded; if (E->st[i].padded > mxs) mxs = E->st[i].padded; }
    lg("\n=== NVME STREAM LOG v1 | mode=%s | %s ===", mode, ts);
    lg("[system] os=%s logical_cpus=%d engine_threads=%d ram_total_MB=%.0f ram_avail_MB=%.0f", 
#ifdef _WIN32
       "windows",
#else
       "linux",
#endif
#ifdef _OPENMP
       omp_get_num_procs(),
#else
       1,
#endif
       threads, t, a);
    lg("[vault] dir=%s streams=%d layer_blob_MB(min/max/avg)=%.1f/%.1f/%.1f embed_or_lm_blob_MB=%.1f total_layer_MB=%.0f", E->vdir, E->nst, mn / 1e6, mxs / 1e6, tot / 1e6 / c->nl, E->st[E->c.nl].padded / 1e6, tot / 1e6);
    lg("[model] hidden=%d inter=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d tie_embeddings=%d rope_theta=%.0f eps=%g", c->hid, c->inter, c->nl, c->nh, c->nkv, c->hd, c->vocab, c->tie, c->theta, c->eps);
    lg("[stream-config] direct_io_requested=%d direct_io_active=%d io_threads=%d io_chunk_MB=%.0f buffers=2x%.0fMB sector=%d prefill_batch=%d ctx=%d kv_cache_MB=%.0f", E->direct, E->vf.direct, E->io_threads, E->io_chunk / 1048576.0, E->bufsz / 1e6, SECT, E->B, ctx, 2.0 * c->nl * ctx * c->nkv * c->hd * 4 / 1e6);
    int nres = 0; double all = 0; for (int i = 0; i < E->nst; i++) { all += (double)E->st[i].padded; if (E->resident[i]) nres++; }
    lg("[resident] requested_MB=%.0f pinned_streams=%d pinned_MB=%.0f lm_head_pinned=%d streamed_MB_per_token=%.0f", E->resident_mb, nres, E->resident_used / 1e6, E->resident[E->lm_s] != NULL, (all - E->resident_used) / 1e6);
}
static void eng_defaults(Eng *E) { memset(E, 0, sizeof *E); E->B = 32; E->io_threads = 2; E->io_chunk = 8u << 20; E->direct = 1; E->detail = 2; }
static int parse_opts(Eng *E, int argc, char **argv, int from, int *ctx, int *threads, const char **logp) {
    for (int i = from; i < argc; i++) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) *ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) *threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) E->B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--io-threads") && i + 1 < argc) E->io_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--io-chunk") && i + 1 < argc) E->io_chunk = (size_t)atoi(argv[++i]) << 20;
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) *logp = argv[++i];
        else if (!strcmp(argv[i], "--log-detail") && i + 1 < argc) E->detail = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--resident-mb") && i + 1 < argc) E->resident_mb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) B_TOKENS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) B_PROMPT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cached")) E->direct = 0;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return -1; }
    }
    if (E->B < 1) E->B = 1; if (E->io_chunk < SECT) E->io_chunk = SECT; E->io_chunk &= ~(size_t)(SECT - 1);
    return 0;
}
static int cmd_check(int argc, char **argv) {
    Eng E; eng_defaults(&E); int ctx = 0, th = 0; const char *logp = "nvme_stream_log.txt";
    if (argc < 3 || parse_opts(&E, argc, argv, 3, &ctx, &th, &logp)) return 2;
    G_LOG = fopen(logp, "a"); if (eng_open(&E, argv[2])) return 1; eng_buffers(&E, 0); log_header(&E, "check", 0, 0);
    double t0 = now(), tot = 0; int bad = 0;
    for (int i = 0; i < E.nst; i++) {
        double t = now(); if (read_stream(&E.vf, E.bufs[0], E.st[i].off, (size_t)E.st[i].padded, E.io_threads, E.io_chunk)) die("read failed");
        double rms = (now() - t) * 1e3; int ok = hash64(E.bufs[0], (size_t)E.st[i].padded) == E.st[i].hash; if (!ok) bad++; tot += E.st[i].padded;
        lg("[check] stream=%d MB=%.1f read_ms=%.1f read_MBps=%.0f hash=%s", i, E.st[i].padded / 1e6, rms, E.st[i].padded / 1e3 / rms, ok ? "ok" : "MISMATCH");
        printf("\rchecked %d/%d", i + 1, E.nst); fflush(stdout);
    }
    double el = now() - t0; lg("[check-summary] total_MB=%.0f seconds=%.2f MBps_incl_hash=%.0f bad_streams=%d", tot / 1e6, el, tot / 1e6 / el, bad);
    printf("\nvault %s: %d streams, %.0f MB in %.1fs (%.0f MB/s incl. hashing), bad=%d\n", bad ? "CORRUPT" : "OK", E.nst, tot / 1e6, el, tot / 1e6 / el, bad); return bad != 0;
}
static int cmd_serve(int argc, char **argv) {
    Eng E; eng_defaults(&E); int ctx = 2048, th = 0; const char *logp = "nvme_stream_log.txt";
    if (argc < 3 || parse_opts(&E, argc, argv, 3, &ctx, &th, &logp)) return 2;
#ifdef _OPENMP
    if (th <= 0) { th = omp_get_num_procs() - 2; if (th < 1) th = 1; } omp_set_num_threads(th);
#else
    th = 1;
#endif
    G_LOG = fopen(logp, "a"); double t0 = now(); if (eng_open(&E, argv[2])) return 1; eng_buffers(&E, E.resident_mb); embed_init(&E); eng_init_model(&E, ctx);
    log_header(&E, "serve", th, ctx); lg("[init] startup_s=%.2f", now() - t0);
    fprintf(stderr, "engine ready: %d layers, %d threads, direct_io=%d, log=%s\n", E.c.nl, th, E.vf.direct, logp); printf("READY\n"); fflush(stdout);
    char *line = malloc(1 << 22);
    while (fgets(line, 1 << 22, stdin)) {
        if (!strncmp(line, "QUIT", 4)) break;
        else if (!strncmp(line, "PING", 4)) { printf("PONG\n"); fflush(stdout); }
        else if (!strncmp(line, "RESET", 5)) { E.pos = 0; lg("[reset] kv cache cleared"); printf("OK\n"); fflush(stdout); }
        else if (!strncmp(line, "GEN", 3) || !strncmp(line, "LOGITS", 6)) {
            int dbg = line[0] == 'L', max_new = 0, topk = 20, ntok = 0; float temp = 0.7f, topp = 0.8f; unsigned long long seed = 0;
            if (dbg) sscanf(line, "LOGITS %d", &ntok); else if (sscanf(line, "GEN %d %f %f %d %llu %d", &max_new, &temp, &topp, &topk, &seed, &ntok) != 6) { printf("ERR bad GEN\n"); fflush(stdout); continue; }
            int *pt = malloc(sizeof(int) * (size_t)(ntok + 1)); if (!fgets(line, 1 << 22, stdin)) break; char *p = line;
            for (int i = 0; i < ntok; i++) { char *e; pt[i] = (int)strtol(p, &e, 10); p = e; }
            if (dbg) { for (int i = 0; i < ntok;) { int n = ntok - i; if (n > E.B) n = E.B; int last = (i + n == ntok); forward(&E, pt + i, n, last); i += n; }
                printf("L"); for (int i = 0; i < E.c.vocab && i < 4096; i++) printf(" %.7g", E.logits[i]); printf("\nEND dbg 0 %d\n", E.pos); fflush(stdout); }
            else gen_cmd(&E, max_new, temp, topp, topk, seed, pt, ntok);
            free(pt);
        } else if (line[0] != '\n' && line[0] != '\r') { printf("ERR unknown command\n"); fflush(stdout); }
    }
    double t, a, ws, pk; mem_info(&t, &a, &ws, &pk); lg("[exit] peak_working_set_MB=%.0f ram_avail_MB=%.0f total_generations=%d", pk, a, G_COUNT); return 0;
}

/* benchmark: fixed prompt, greedy, exactly --tokens decode steps (EOS ignored); prints BENCH/T/S/END lines */
static int cmd_bench(int argc, char **argv) {
    Eng E; eng_defaults(&E); int ctx = 256, th = 0; const char *logp = "nvme_stream_log.txt";
    if (argc < 3 || parse_opts(&E, argc, argv, 3, &ctx, &th, &logp)) return 2;
    if (ctx < B_PROMPT + B_TOKENS + 8) ctx = B_PROMPT + B_TOKENS + 8;
#ifdef _OPENMP
    if (th <= 0) { th = omp_get_num_procs() - 2; if (th < 1) th = 1; } omp_set_num_threads(th);
#else
    th = 1;
#endif
    G_LOG = fopen(logp, "a"); if (eng_open(&E, argv[2])) return 1; eng_buffers(&E, E.resident_mb); embed_init(&E); eng_init_model(&E, ctx);
    log_header(&E, "bench", th, ctx); G_IGNORE_EOS = 1;
    int *pt = malloc(sizeof(int) * (size_t)B_PROMPT); for (int i = 0; i < B_PROMPT; i++) pt[i] = (1000 + 7919 * i) % E.c.vocab;
    printf("BENCH threads=%d io_threads=%d io_chunk_MB=%.0f direct=%d resident_MB=%.0f\n", th, E.io_threads, E.io_chunk / 1048576.0, E.vf.direct, E.resident_used / 1e6);
    gen_cmd(&E, B_TOKENS, 0.0f, 1.0f, 1, 1, pt, B_PROMPT);
    return 0;
}
static uint64_t srs = 88172645463325252ull;
static uint32_t srnd(void) { return (uint32_t)(rng_next(&srs) >> 16); }
static int fsame(float a, float b) { uint32_t x, y; memcpy(&x, &a, 4); memcpy(&y, &b, 4); return x == y || (a != a && b != b); }
static int selftest(void) {
    int shapes[][2] = {{1, 1}, {7, 13}, {5, 4097}, {64, 64}, {9, 2563}, {16, 4096}}, bad = 0;
    for (int k = 0; k < 6; k++) for (int mode = 0; mode < 4; mode++) {
        int R = shapes[k][0], C = shapes[k][1]; size_t n = (size_t)R * C; uint16_t *w = malloc(2 * n);
        for (size_t i = 0; i < n; i++) { uint32_t r = srnd(); uint16_t e, sm = (uint16_t)(srnd() & 0xFF);
            if (mode == 0) e = 127; else if (mode == 1) e = r & 255;
            else if (mode == 2) { int z = 0; uint32_t q = r | 0x80000000u; while (!(q & 1)) { q >>= 1; z++; } e = (uint16_t)(z > 255 ? 255 : z); } else e = (uint16_t)(120 + (r & 7));
            w[i] = (uint16_t)((sm & 0x80) << 8 | e << 7 | (sm & 0x7F)); }
        size_t sz; uint8_t *b = compress_layer(w, R, C, &sz, NULL); int ok = verify_blob(b, w);
        float *x = malloc(4 * (size_t)C * 3), *y1 = malloc(4 * (size_t)R * 3), *y2 = malloc(4 * (size_t)R * 3), *row = malloc(4 * (size_t)C);
        for (int j = 0; j < C * 3; j++) x[j] = (float)(j % 17) * 0.125f - 1.0f;
        mv_batch(b, x, y1, 3); for (int t = 0; t < 3; t++) mv_batch(b, x + (size_t)t * C, y2 + (size_t)t * R, 1);
        for (int q = 0; q < R * 3; q++) if (!fsame(y1[q], y2[q])) ok = 0;
        for (int r = 0; r < R; r++) { raw_row(w + (size_t)r * C, C, row); { float d = dot_row(row, x, C); if (!fsame(d, y1[r])) ok = 0; } }
#if defined(__AVX2__)
        {   /* the AVX2 kernel must reproduce every stored weight bit for bit (NaN/Inf patterns included) */
            uint32_t rr, cc; memcpy(&rr, b, 4); memcpy(&cc, b + 4, 4); const uint32_t *offv = (const uint32_t *)(b + 264);
            const uint8_t *esv = b + 264 + 4 * (4 * (size_t)rr + 1); const uint8_t *smv = esv + offv[4 * (size_t)rr] + 8;
            uint16_t *lutv = malloc(sizeof(uint16_t) << LMAX); build_lut(b + 8, lutv); uint32_t *l3v = malloc(4 << LMAX);
            for (int v = 0; v < (1 << LMAX); v++) l3v[v] = lut3_entry(lutv, (uint32_t)v);
            uint8_t *ebv = malloc(4 * (((size_t)C >> 2) + 16) + 64); float *outv = malloc(4 * (size_t)C + 64);
            for (int r = 0; r < R; r++) { decode_row3(l3v, esv, offv + 4 * (size_t)r, smv + (size_t)r * C, C, outv, ebv); raw_row(w + (size_t)r * C, C, row); if (memcmp(outv, row, 4 * (size_t)C)) ok = 0; }
            free(lutv); free(l3v); free(ebv); free(outv);
        }
#endif
        printf("  %dx%d mode %d: %s\n", R, C, mode, ok ? "bit-exact" : "FAIL"); bad += !ok; free(w); free(b); free(x); free(y1); free(y2); free(row);
    }
    puts(bad ? "SELFTEST FAILED" : "SELFTEST PASSED"); return bad;
}
int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "selftest")) return selftest();
    if (argc >= 4 && !strcmp(argv[1], "pack")) { G_LOG = fopen("nvme_stream_log.txt", "a"); return cmd_pack(argv[2], argv[3]); }
    if (argc >= 3 && !strcmp(argv[1], "serve")) return cmd_serve(argc, argv);
    if (argc >= 3 && !strcmp(argv[1], "check")) return cmd_check(argc, argv);
    if (argc >= 3 && !strcmp(argv[1], "bench")) return cmd_bench(argc, argv);
    fprintf(stderr, "usage: nvme_engine selftest | pack <model_dir> <vault_dir> | serve <vault_dir> [opts] | check <vault_dir> [opts]\n"); return 2;
}
