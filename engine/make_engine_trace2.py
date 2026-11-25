"""make_engine_trace2.py - NVME_MOE_TRACE routing trace for gpt-oss too (same record format as the Qwen path: int16 layer, position, top-k ids)."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
old = '    E->pf_n = 0;\n    if (getenv("NVME_MOE_PREDICT") || E->pf_on) {'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''    { static FILE *tr2 = NULL; static int ti2 = 0; if (!ti2) { const char *p = getenv("NVME_MOE_TRACE"); if (p) tr2 = fopen(p, "ab"); ti2 = 1; }
      if (tr2) for (int t = 0; t < n; t++) { int16_t rec[2 + 16]; rec[0] = (int16_t)l; rec[1] = (int16_t)(E->pos + t);
          for (int j = 0; j < K && j < 16; j++) rec[2 + j] = (int16_t)sel[t * K + j]; fwrite(rec, 2, 2 + (K < 16 ? K : 16), tr2); fflush(tr2); } }
''' + old)
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("gpt-oss routing trace added")

# NVME_VRAM_LRU=1: VRAM as a pure victim cache (always keep the expert RAM gives up; evict the least recently used VRAM expert),
# i.e. RAM + VRAM behave like one exclusive LRU - for comparison with the frequency-gated admission
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
def rp(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)
rp("    if (w >= 0 && E->v2owner[w] >= 0 && E->xmap[E->v2owner[w]] < 0 && E->xcnt[E->v2owner[w]] > E->xcnt[old]) w = -1;   /* not worth displacing */\n    if (w < 0) return -1;",
   "    static int vlru = -1; if (vlru < 0) vlru = getenv(\"NVME_VRAM_LRU\") != NULL;\n"
   "    if (vlru) { w = -1; for (int q = 0; q < E->nv; q++) { if (E->v2tick[q] >= stamp) continue; int o = E->v2owner[q];\n"
   "            if (o < 0) { w = q; break; } if (w < 0 || E->v2tick[q] < E->v2tick[w]) w = q; } }\n"
   "    else if (w >= 0 && E->v2owner[w] >= 0 && E->xmap[E->v2owner[w]] < 0 && E->xcnt[E->v2owner[w]] > E->xcnt[old]) w = -1;   /* not worth displacing */\n    if (w < 0) return -1;")
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("NVME_VRAM_LRU switch")
