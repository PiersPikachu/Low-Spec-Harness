"""make_engine_mxf2.py - CPU matvec reads MXF2 sub-blobs (MXFP4 with packed block scales, pack_gptoss.py --mxf2) as well as MXF4.
Scales per row: base + 2-bit offsets, or the raw bytes from the exception table; expanded into a small row buffer before the
block loop, so the arithmetic is the same as MXF4's."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old)); s = s.replace(old, new)
rep("    const uint8_t *S = blob + 12, *Bk = S + (size_t)rows * nb;\n",
    "    const uint8_t *S = blob + 12, *Bk = S + (size_t)rows * nb; int mx2 = blob[3] == '2'; uint32_t pb2 = 0, boff2 = 0;\n"
    "    const uint32_t *dsc2 = NULL; const uint8_t *pk2 = NULL, *ex2 = NULL;\n"
    "    if (mx2) { memcpy(&pb2, blob + 12, 4); memcpy(&boff2, blob + 20, 4); dsc2 = (const uint32_t *)(blob + 24); pk2 = blob + 24 + 4 * (size_t)rows;\n"
    "        ex2 = pk2 + (size_t)rows * pb2; Bk = blob + boff2; }\n")
rep("        const uint8_t *sr = S + (size_t)r * nb, *br = Bk + (size_t)r * nb * 16;\n",
    "        uint8_t rs2[512]; const uint8_t *sr = S + (size_t)r * nb, *br = Bk + (size_t)r * nb * 16;\n"
    "        if (mx2) { uint32_t d = dsc2[r]; if (d >> 31) sr = ex2 + (size_t)(d & 0x7fffffffu) * nb;\n"
    "            else { const uint8_t *pk = pk2 + (size_t)r * pb2; for (int b = 0; b < nb; b++) rs2[b] = (uint8_t)((d & 255u) + ((pk[b >> 2] >> ((b & 3) * 2)) & 3u)); sr = rs2; } }\n", cnt=2)
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("MXF2 reading added to the CPU kernel")

# MXF2 sub-blobs vary in size per expert (exception tables), so sub-blob offsets differ per expert: record them per VRAM slot
# when an expert is saved there (from the RAM copy's stream header) and hand those to the GPU, instead of one layout for all
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
rep("int *xcnt; int pred_l", "int *xcnt; uint64_t *vsub; int pred_l")
rep("E->xcnt = calloc(nx, sizeof(int));", "E->xcnt = calloc(nx, sizeof(int)); E->vsub = calloc((size_t)nv * 4, sizeof(uint64_t));")
rep("    if (E->v2owner[w] >= 0) E->v2map[E->v2owner[w]] = -1; E->v2owner[w] = old; E->v2map[old] = w; E->v2tick[w] = stamp; E->vsave++; return w;",
    "    if (E->v2owner[w] >= 0) E->v2map[E->v2owner[w]] = -1; E->v2owner[w] = old; E->v2map[old] = w; E->v2tick[w] = stamp; E->vsave++;\n"
    "    for (int i = 0; i < 4; i++) memcpy(&E->vsub[(size_t)w * 4 + i], E->xslot[v] + 8 + 8 * i, 8);\n    return w;")
rep("E->gxoff[0], E->gxoff[1], E->gxoff[2], E->gxoff[3],",
    "E->vsub[(size_t)P->slot[ix] * 4], E->vsub[(size_t)P->slot[ix] * 4 + 1], E->vsub[(size_t)P->slot[ix] * 4 + 2], E->vsub[(size_t)P->slot[ix] * 4 + 3],")
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("per-slot sub-blob offsets for the GPU")
