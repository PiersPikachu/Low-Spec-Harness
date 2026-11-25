#!/usr/bin/env python3
"""
pack_universal_v3.py - metadata-driven vault3 packer for any HF safetensors model.

Output (vault_dir):
  vault.bin    streams: super-blob of huffman4 v3 sub-blobs (byte-identical to
               `nvme_engine pack` for Qwen3-style dense models)
  resident.bin float32 for every 1-D tensor (norms, biases), in table order
  vault.idx    WVAULT 3 (engine-compatible) or WVAULT 4 (non-Qwen3 layout)
  vault.json   FULL config.json + per-tensor table (name, shape, dtype, stream,
               sub-blob offset/size, resident offset)  <- the universal part

Commands:
  pack <model_dir> <vault_dir> [--convert-matrices]
  check <model_dir> <vault_dir>      decode vault back, compare to source
  compare <vault_a> <vault_b>        byte-compare two vaults (vs nvme_engine pack)
  selftest                           synthetic models, full roundtrip
Deps: numpy, numba. No torch.
"""
import argparse, glob, json, os, re, struct, sys, time
from pathlib import Path
import numpy as np
from numba import njit

LMAX = 12
SECT = 4096
MAGIC = 0x534C5657
KNOWN_MAT = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
KNOWN_NORM = ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"]
LAYER_RE = re.compile(r'^(?P<pre>.*?)(?P<cont>layers|h|blocks)\.(?P<idx>\d+)\.(?P<rest>.+)$')
EMBED_RE = re.compile(r'(^|\.)(embed_tokens|wte|tok_embeddings|word_embeddings)\.weight$')
HEAD_RE = re.compile(r'(^|\.)(lm_head|output)\.weight$')
EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
DT = {"BF16": "<u2", "F16": "<u2", "F32": "<f4"}


# ------------------------------------------------------------------ huffman (mirror of nvme_engine.c)
def huff_lengths(freq):
    f = [int(x) for x in freq]
    while True:
        ln = [0] * 256
        idx = [i for i in range(256) if f[i]]
        n = len(idx)
        if n == 0:
            return ln
        if n == 1:
            ln[idx[0]] = 1
            return ln
        w = [0] * (2 * n + 2); par = [-1] * (2 * n + 2); al = list(range(n)); na = n; m = n
        for k in range(n):
            w[k] = f[idx[k]]
        while na > 1:
            a = 0; b = -1
            for k in range(1, na):
                if w[al[k]] < w[al[a]]:
                    a = k
            for k in range(na):
                if k == a:
                    continue
                if b < 0 or w[al[k]] < w[al[b]]:
                    b = k
            w[m] = w[al[a]] + w[al[b]]; par[m] = -1; par[al[a]] = m; par[al[b]] = m
            al[a] = m; al[b] = al[na - 1]; na -= 1; m += 1
        mx = 0
        for k in range(n):
            d = 0; p = par[k]
            while p >= 0:
                d += 1; p = par[p]
            ln[idx[k]] = d
            mx = max(mx, d)
        if mx <= LMAX:
            return ln
        f = [((x + 1) >> 1) if x else 0 for x in f]


def make_rcodes(ln):
    cnt = [0] * (LMAX + 2); nxt = [0] * (LMAX + 2); c = 0
    for s in range(256):
        if ln[s]:
            cnt[ln[s]] += 1
    for l in range(1, LMAX + 1):
        c = (c + cnt[l - 1]) << 1
        nxt[l] = c
    rc = [0] * 256
    for s in range(256):
        l = ln[s]
        if not l:
            continue
        code = nxt[l]; nxt[l] += 1; r = 0
        for i in range(l):
            r |= ((code >> i) & 1) << (l - 1 - i)
        rc[s] = r
    return rc


def build_lut(ln):
    rc = make_rcodes(ln)
    lut = np.zeros(1 << LMAX, np.int64)
    for s in range(256):
        l = ln[s]
        if not l:
            continue
        ks = np.arange(1 << (LMAX - l), dtype=np.int64)
        lut[rc[s] | (ks << l)] = s | (l << 8)
    return lut


# ------------------------------------------------------------------ numba kernels
@njit(cache=True)
def _freq(w):
    f = np.zeros(256, np.int64)
    for r in range(w.shape[0]):
        for j in range(w.shape[1]):
            f[(np.int64(w[r, j]) >> 7) & 255] += 1
    return f


@njit(cache=True)
def _sizes(w, ln):
    rows, cols = w.shape
    off = np.empty(4 * rows + 1, np.int64); pos = 0
    for r in range(rows):
        for k in range(4):
            off[4 * r + k] = pos; bits = 0
            for j in range(k, cols, 4):
                bits += ln[(np.int64(w[r, j]) >> 7) & 255]
            pos += (bits + 7) >> 3
    off[4 * rows] = pos
    return off


@njit(cache=True)
def _write(w, ln, rc, off, es, sm):
    rows, cols = w.shape
    for r in range(rows):
        for k in range(4):
            pos = off[4 * r + k]; acc = np.int64(0); nb = 0
            for j in range(k, cols, 4):
                e = (np.int64(w[r, j]) >> 7) & 255
                acc |= rc[e] << nb
                nb += ln[e]
                while nb >= 8:
                    es[pos] = acc & 255; pos += 1; acc >>= 8; nb -= 8
            if nb > 0:
                es[pos] = acc & 255; pos += 1
        for j in range(cols):
            x = np.int64(w[r, j])
            sm[r * cols + j] = ((x >> 8) & 0x80) | (x & 0x7F)


@njit(cache=True)
def _decode(es, off, sm, lut, out):
    rows, cols = out.shape
    for r in range(rows):
        for k in range(4):
            pos = off[4 * r + k]; acc = np.int64(0); nb = 0
            for j in range(k, cols, 4):
                while nb < 12:
                    b = 0
                    if pos < es.size:
                        b = es[pos]
                    acc |= np.int64(b) << nb; pos += 1; nb += 8
                ent = lut[acc & 4095]
                l = ent >> 8
                acc >>= l; nb -= l
                s = np.int64(sm[r * cols + j])
                out[r, j] = ((s & 0x80) << 8) | ((ent & 255) << 7) | (s & 0x7F)


@njit(cache=True)
def hash64(buf):
    h = np.uint64(1469598103934665603); P = np.uint64(1099511628211); S = np.uint64(29)
    n = buf.size; m = (n // 8) * 8
    w8 = buf[:m].view(np.uint64)
    for i in range(w8.size):
        h = h ^ w8[i]; h = h * P; h = h ^ (h >> S)
    for i in range(m, n):
        h = h ^ np.uint64(buf[i]); h = h * P
    return h


def hash64_ref(b):
    M = (1 << 64) - 1; h = 1469598103934665603; P = 1099511628211; m = (len(b) // 8) * 8
    for i in range(0, m, 8):
        h ^= int.from_bytes(b[i:i + 8], 'little'); h = (h * P) & M; h ^= h >> 29
    for i in range(m, len(b)):
        h ^= b[i]; h = (h * P) & M
    return h


# ------------------------------------------------------------------ v3 blob
def compress_matrix(w):
    """w: C-contiguous uint16 (rows, cols) bf16 bits -> v3 blob (np.uint8), verified bit-exact."""
    rows, cols = w.shape; n = rows * cols
    ln = huff_lengths(_freq(w)); rc = make_rcodes(ln)
    ln_a = np.array(ln, np.int64); rc_a = np.array(rc, np.int64)
    off = _sizes(w, ln_a); pos = int(off[-1])
    if pos > 0xFFFFFFF0:
        raise SystemExit("matrix too large for 32-bit stream offsets")
    no4 = 4 * rows + 1
    p_off = 264; p_es = p_off + 4 * no4; p_sm = p_es + pos + 8; total = p_sm + n
    blob = np.zeros(total, np.uint8)
    blob[0:8] = np.frombuffer(struct.pack('<II', rows, cols), np.uint8)
    blob[8:264] = np.array(ln, np.uint8)
    blob[p_off:p_off + 4 * no4] = off.astype('<u4').view(np.uint8)
    _write(w, ln_a, rc_a, off, blob[p_es:p_es + pos], blob[p_sm:p_sm + n])
    return blob


def parse_blob(blob):
    rows, cols = struct.unpack('<II', bytes(blob[0:8]))
    ln = [int(x) for x in blob[8:264]]
    no4 = 4 * rows + 1
    off = blob[264:264 + 4 * no4].view('<u4').astype(np.int64)
    p_es = 264 + 4 * no4; pos = int(off[-1]); p_sm = p_es + pos + 8
    return rows, cols, ln, off, blob[p_es:], blob[p_sm:p_sm + rows * cols]


def decode_blob(blob):
    rows, cols, ln, off, es, sm = parse_blob(blob)
    out = np.empty((rows, cols), np.uint16)
    _decode(es, off, sm, build_lut(ln), out)
    return out


# ------------------------------------------------------------------ safetensors (no torch)
def scan_model(model_dir, allow_missing=False):
    model_dir = Path(model_dir)
    ip = model_dir / "model.safetensors.index.json"
    if ip.exists():
        shards = sorted(set(json.load(open(ip, encoding='utf-8'))["weight_map"].values()))
    else:
        shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit("no .safetensors files in " + str(model_dir))
    T = {}
    for sh in shards:
        p = model_dir / sh
        if allow_missing and not p.exists():
            continue
        with open(p, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            b, e = v["data_offsets"]
            T[k] = dict(path=str(p), data0=8 + n, dtype=v["dtype"], shape=tuple(v["shape"]), b=b, e=e)
    return T


def load_raw(t):
    if t["dtype"] not in DT:
        raise SystemExit(f"unsupported dtype {t['dtype']} (supported: {sorted(DT)})")
    cnt = int(np.prod(t["shape"], dtype=np.int64)) if t["shape"] else 1
    return np.fromfile(t["path"], dtype=DT[t["dtype"]], count=cnt, offset=t["data0"] + t["b"])


def to_bf16_bits(arr, dtype, allow, name):
    if dtype == "BF16":
        return arr
    if not allow:
        raise SystemExit(f"{name} is {dtype}: matrices must be BF16 to stay lossless. "
                         f"Re-run with --convert-matrices to round to bf16 (LOSSY, flagged in vault.json).")
    f = arr.view(np.float16).astype(np.float32) if dtype == "F16" else arr
    u = f.view(np.uint32).astype(np.uint64)
    out = ((u + ((u >> 16) & 1) + 0x7FFF) >> 16).astype(np.uint16)
    out[np.isnan(f)] = 0x7FC0
    return out


def to_f32(arr, dtype):
    if dtype == "BF16":
        return (arr.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return arr.view(np.float16).astype(np.float32)
    return arr.astype(np.float32, copy=False)


# ------------------------------------------------------------------ planning
def _rank(stem, order):
    return order.index(stem) if stem in order else len(order)


def _stem(rest):
    return re.sub(r'\.(weight|bias)$', '', rest)


def plan_streams(T, cfg):
    tc = cfg.get("text_config", cfg)
    nl = tc.get("num_hidden_layers")
    groups = {}
    for name in T:
        m = LAYER_RE.match(name)
        if m:
            groups.setdefault(m["pre"] + m["cont"], {}).setdefault(int(m["idx"]), []).append((name, m["rest"]))
    main = None
    if groups:
        cands = [k for k, g in groups.items() if nl is not None and len(g) == nl]
        if len(cands) >= 1:
            main = sorted(cands, key=lambda k: (k != "model.layers", k))[0]
        elif nl is None:
            main = max(groups, key=lambda k: len(groups[k]))
        else:
            raise SystemExit(f"no layer group with num_hidden_layers={nl}; found { {k: len(g) for k, g in groups.items()} }")
    layers = []; used = set(); resident = []
    if main:
        g = groups[main]
        if sorted(g) != list(range(len(g))):
            raise SystemExit("layer indices are not contiguous from 0")
        experts = []
        for li in range(len(g)):
            ents = g[li]
            ex = {}
            for n, r in ents:
                m = EXPERT_RE.match(r)
                if m: ex.setdefault(int(m[1]), {})[m[2]] = n
            if ex:
                for e in sorted(ex): experts.append(dict(kind="expert", layer=li, expert=e, mats=[ex[e]["gate_proj"], ex[e]["up_proj"], ex[e]["down_proj"]]))
                ents = [(n, r) for n, r in ents if not EXPERT_RE.match(r)] + [(n, r) for n, r in ents if EXPERT_RE.match(r)]
            mats = sorted([(n, r) for n, r in ents if len(T[n]["shape"]) >= 2 and not EXPERT_RE.match(r)], key=lambda x: (_rank(_stem(x[1]), KNOWN_MAT), x[0]))
            norms = sorted([(n, r) for n, r in ents if len(T[n]["shape"]) < 2], key=lambda x: (_rank(_stem(x[1]), KNOWN_NORM), x[0]))
            layers.append(dict(kind="layer", layer=li, mats=[n for n, _ in mats], stems=[_stem(r) for _, r in mats]))
            resident += [n for n, _ in norms]
            used.update(n for n, _ in ents)
    rest = sorted(n for n in T if n not in used)
    rest_mats = [n for n in rest if len(T[n]["shape"]) >= 2]
    resident += [n for n in rest if len(T[n]["shape"]) < 2]
    embed = next((n for n in rest_mats if EMBED_RE.search(n)), None)
    head = next((n for n in rest_mats if HEAD_RE.search(n) and n != embed), None)
    streams = list(layers)
    if embed:
        streams.append(dict(kind="embed", layer=None, mats=[embed]))
    if head:
        streams.append(dict(kind="lm_head", layer=None, mats=[head]))
    for n in rest_mats:
        if n not in (embed, head):
            streams.append(dict(kind="extra", layer=None, mats=[n]))
    if main:
        streams += experts
    return streams, resident, len(layers)


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
                   "generation_config.json", "chat_template.jinja", "chat_template.json", "tokenizer.model")


def tokenizer_eos_id(model_dir):
    """id of tokenizer_config's eos_token, looked up in tokenizer.json (no transformers import)"""
    try:
        tc = json.load(open(Path(model_dir) / "tokenizer_config.json", encoding="utf-8")); e = tc.get("eos_token")
        e = e.get("content") if isinstance(e, dict) else e
        if not e:
            return None
        tj = json.load(open(Path(model_dir) / "tokenizer.json", encoding="utf-8"))
        for a in tj.get("added_tokens", []):
            if a.get("content") == e:
                return int(a["id"])
        v = tj.get("model", {}).get("vocab", {})
        return int(v[e]) if isinstance(v, dict) and e in v else None
    except Exception:
        return None


def copy_tokenizer(model_dir, vault_dir):
    """the vault carries its own tokenizer + chat template, so chat.py needs only the vault folder"""
    import shutil
    n = 0
    for f in TOKENIZER_FILES:
        if (Path(model_dir) / f).exists():
            shutil.copy(Path(model_dir) / f, Path(vault_dir) / f); n += 1
    return n


def cfg_scalars(cfg, model_dir):
    c = cfg.get("text_config", cfg)
    heads = c.get("num_attention_heads", 0)
    hid = c.get("hidden_size", 0)
    eos = c.get("eos_token_id", 0)
    gp = Path(model_dir) / "generation_config.json"
    if gp.exists():
        eos = json.load(open(gp, encoding='utf-8')).get("eos_token_id", eos)
    eos = list(eos) if isinstance(eos, list) else [eos]
    eos = [e for e in eos if e is not None] or [-1]          # no end token in the config: never stop on one
    tid = tokenizer_eos_id(model_dir)
    if tid is not None and tid not in eos:
        eos.append(tid)
    rp = c.get("rope_parameters") or {}
    theta = c.get("rope_theta") or rp.get("rope_theta") or 0
    return dict(hidden=hid, inter=c.get("intermediate_size", 0), layers=c.get("num_hidden_layers", 0), heads=heads,
                kv_heads=c.get("num_key_value_heads", heads), head_dim=c.get("head_dim") or (hid // heads if heads else 0),
                vocab=c.get("vocab_size", 0), tie=1 if c.get("tie_word_embeddings") else 0, eos=eos[0],
                eos2=eos[1] if len(eos) > 1 else eos[0], eps=c.get("rms_norm_eps", 0), theta=theta)


# ------------------------------------------------------------------ pack
ROLE = {"input_layernorm.weight": "ln_in", "post_attention_layernorm.weight": "ln_post",
        "self_attn.q_norm.weight": "qn", "self_attn.k_norm.weight": "kn",
        "self_attn.q_proj.bias": "bq", "self_attn.k_proj.bias": "bk", "self_attn.v_proj.bias": "bv", "self_attn.o_proj.bias": "bo",
        "mlp.gate_proj.bias": "bg", "mlp.up_proj.bias": "bu", "mlp.down_proj.bias": "bd"}


def engine_u_layout(cfg, sc, streams, nl, res_names, T, lossy):
    """Can nvme_engine_u.exe run this vault? Dense decoder: q/k/v/o + gate/up/down, RMSNorm, SiLU, RoPE (default or llama3)."""
    c = cfg.get("text_config", cfg); why = []
    if lossy:
        why.append("matrices were rounded to bf16 (lossy)")
    if nl <= 0 or any(x["stems"] != KNOWN_MAT for x in streams[:nl]):
        why.append("layer matrices are not exactly q/k/v/o_proj + gate/up/down_proj")
    if len(streams) <= nl or streams[nl]["kind"] != "embed":
        why.append("no embed_tokens stream after the layers")
    if (sc["tie"] == 0) and (len(streams) <= nl + 1 or streams[nl + 1]["kind"] != "lm_head"):
        why.append("untied model without lm_head stream")
    if len(streams) > nl + 1 + (0 if sc["tie"] else 1):
        why.append("extra non-layer matrices")
    if c.get("hidden_act", "silu") not in ("silu", "swish"):
        why.append(f"activation {c.get('hidden_act')} (engine has silu only)")
    if c.get("use_sliding_window"):
        why.append("sliding-window attention")
    if c.get("partial_rotary_factor") not in (None, 1, 1.0):
        why.append("partial rotary")
    rs = c.get("rope_scaling") or c.get("rope_parameters") or {}
    rt = rs.get("rope_type") or rs.get("type")
    extra = []
    if rt in (None, "default"):
        pass
    elif rt == "llama3":
        try:
            extra += [f"rope_llama3 1", f"rope_factor {'%.10g' % rs['factor']}", f"rope_low {'%.10g' % rs['low_freq_factor']}",
                      f"rope_high {'%.10g' % rs['high_freq_factor']}", f"rope_orig {'%.10g' % rs['original_max_position_embeddings']}"]
        except KeyError as e:
            why.append(f"llama3 rope missing {e}")
    else:
        why.append(f"rope scaling type {rt}")
    hid, hd, nh, nkv, inter = sc["hidden"], sc["head_dim"], sc["heads"], sc["kv_heads"], sc["inter"]
    want = {"ln_in": hid, "ln_post": hid, "qn": hd, "kn": hd, "bq": nh * hd, "bk": nkv * hd, "bv": nkv * hd, "bo": hid, "bg": inter, "bu": inter, "bd": hid}
    layout = None; layer_names = set()
    for li in range(nl if not why else 0):
        pre = streams[li]["mats"][0].rsplit(".self_attn.", 1)[0] + "."
        mine = [n for n in res_names if n.startswith(pre)]; layer_names.update(mine)
        cur = []; o = 0
        for n in mine:
            role = ROLE.get(n[len(pre):])
            if role is None:
                why.append(f"unsupported per-layer tensor {n[len(pre):]}"); break
            cnt = int(np.prod(T[n]["shape"], dtype=np.int64)) if T[n]["shape"] else 1
            if cnt != want[role]:
                why.append(f"{n} has {cnt} values, engine expects {want[role]}"); break
            cur.append((role, o, cnt)); o += cnt
        else:
            if li == 0:
                layout = (cur, o)
                if "ln_in" not in [r for r, _, _ in cur] or "ln_post" not in [r for r, _, _ in cur]:
                    why.append("missing input/post-attention layernorm")
            elif (cur, o) != layout:
                why.append(f"layer {li} resident layout differs from layer 0")
            if [r for r, _, _ in (layout[0] if layout else [])].count("qn") != [r for r, _, _ in (layout[0] if layout else [])].count("kn"):
                why.append("q_norm without k_norm (or vice versa)")
        if why:
            break
    tail = [n for n in res_names if n not in layer_names]
    if not why and (len(tail) != 1 or not tail[0].endswith("norm.weight")):
        why.append(f"expected exactly one final norm outside the layers, found {tail}")
    lines = ""
    if not why:
        cur, tot = layout
        offs = {r: o for r, o, _ in cur}
        lines = f"lres {tot}\n" + "".join(f"r_{r} {offs[r]}\n" for r in offs) + "".join(x + "\n" for x in extra) + "supported 1\n"
    return dict(supported=not why, reasons=why, idx_lines=lines, layout=[(r, o, n) for r, o, n in layout[0]] if layout and not why else None)


def engine_moe_layout(cfg, sc, streams, nl, res_names, T):
    """MoE engine contract: per layer q/k/v/o + router (mlp.gate), norms ln_in/ln_post/q_norm/k_norm; experts gate/up/down, SiLU"""
    c = cfg.get("text_config", cfg); why = []
    ex = [s for s in streams if s["kind"] == "expert"]; E = c.get("num_experts") or c.get("num_local_experts"); K = c.get("num_experts_per_tok")
    if not ex: return None
    if len(ex) != nl * (E or 0): why.append(f"{len(ex)} expert streams, expected {nl} x {E}")
    for li in range(nl):
        if streams[li]["stems"] != ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate"]:
            why.append(f"layer {li} matrices {streams[li]['stems']}"); break
    if c.get("hidden_act", "silu") != "silu": why.append("activation")
    if c.get("mlp_only_layers") or c.get("decoder_sparse_step", 1) != 1: why.append("mixed dense/MoE layers")
    if c.get("shared_expert_intermediate_size"): why.append("shared expert")
    if (c.get("rope_scaling") or {}).get("rope_type", "default") not in ("default", None): why.append("rope scaling")
    first = next(i for i, s in enumerate(streams) if s["kind"] == "expert")
    want = {"input_layernorm.weight": "ln_in", "post_attention_layernorm.weight": "ln_post", "self_attn.q_norm.weight": "qn", "self_attn.k_norm.weight": "kn"}
    pre0 = streams[0]["mats"][0].rsplit(".self_attn.", 1)[0] + "."
    mine = [n for n in res_names if n.startswith(pre0)]; cur = []; o = 0
    for n in mine:
        r = want.get(n[len(pre0):])
        if r is None: why.append(f"unsupported per-layer tensor {n[len(pre0):]}"); break
        cnt = int(np.prod(T[n]["shape"])); cur.append((r, o)); o += cnt
    lines = (f"moe 1\nn_experts {E}\ntop_k {K}\nmoe_inter {c.get('moe_intermediate_size')}\nnorm_topk {1 if c.get('norm_topk_prob') else 0}\n"
             f"expert_stream0 {first}\nlres {o}\n" + "".join(f"r_{r} {off}\n" for r, off in cur) + "supported 1\n")
    return dict(supported=not why, reasons=why, idx_lines=lines if not why else "")


def pack(model_dir, vault_dir, allow_convert=False, delete_consumed=False, codec="huffman4"):
    if codec not in ("huffman4", "rans4"):
        raise SystemExit(f"unknown codec {codec}")
    import rans4 as _r4
    t0 = time.time(); model_dir = Path(model_dir); vd = Path(vault_dir); vd.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(model_dir / "config.json", encoding='utf-8'))
    T = scan_model(model_dir)
    streams, res_names, nl = plan_streams(T, cfg)
    sc = cfg_scalars(cfg, model_dir)
    aliases = []
    if sc["tie"]:
        for S in list(streams):
            if S["kind"] != "lm_head":
                continue
            E = next((x for x in streams if x["kind"] == "embed"), None)
            h, e = S["mats"][0], (E["mats"][0] if E else None)
            if e and T[h]["shape"] == T[e]["shape"] and T[h]["dtype"] == T[e]["dtype"] and np.array_equal(load_raw(T[h]), load_raw(T[e])):
                streams.remove(S); aliases.append((h, e))
                print(f"tie_word_embeddings=1: {h} is byte-identical to {e}; stored once (alias)")
            else:
                print(f"WARNING: tie_word_embeddings=1 but {h} differs from embed; keeping its own stream")
    print(f"model_type={cfg.get('model_type')} tensors={len(T)} layers={nl} streams={len(streams)} resident_tensors={len(res_names)}")
    table = []; lossy = False; sum_raw = 0; sum_c = 0
    res_cache = {}; last_use = {}
    if delete_consumed:
        # disk-tight mode: every matrix is verified bit-exact in memory before it is written, then a source shard is deleted as soon as no later
        # stream needs it. The small 1-D tensors (norms/biases) are read up front so no shard has to outlive its matrices.
        print("--delete-consumed: source .safetensors shards are removed as they are consumed (re-download to re-verify later)")
        for name in res_names:
            t = T[name]; res_cache[name] = to_f32(load_raw(t), t["dtype"])
        for si_, S_ in enumerate(streams):
            for n_ in S_["mats"]: last_use[T[n_]["path"]] = si_
        for pth in sorted({t["path"] for t in T.values()} - set(last_use)):
            if os.path.exists(pth): os.remove(pth); print(f"deleted {Path(pth).name} (only small tensors / tied duplicate)")
    with open(vd / "vault.bin", "wb") as vf:
        for si, S in enumerate(streams):
            subs = []; raw = 0
            for sub, name in enumerate(S["mats"]):
                t = T[name]; arr = to_bf16_bits(load_raw(t), t["dtype"], allow_convert, name)
                if t["dtype"] != "BF16":
                    lossy = True
                rows = t["shape"][0]; cols = int(np.prod(t["shape"][1:], dtype=np.int64))
                w = np.ascontiguousarray(arr.reshape(rows, cols))
                blob = compress_matrix(w) if codec == "huffman4" else _r4.encode(w, 12, 16)
                dec = decode_blob(blob) if codec == "huffman4" else _r4.decode(blob)
                if not np.array_equal(dec, w):
                    raise SystemExit(f"VERIFY FAILED: {name} does not decode bit-exact")
                subs.append(blob); raw += 2 * rows * cols
                table.append(dict(name=name, shape=list(t["shape"]), src_dtype=t["dtype"], store="huff4" if codec == "huffman4" else "rans4", stream=si, sub=sub,
                                  rows=rows, cols=cols, blob_size=int(blob.size)))
                del w, arr, dec
            n = len(subs); hdr = max(128, (8 + 8 * n + 63) & ~63); pos = hdr; offs = []
            for b in subs:
                pos = (pos + 63) & ~63; offs.append(pos); pos += b.size
            padded = (pos + SECT - 1) & ~(SECT - 1)
            buf = np.zeros(padded, np.uint8)
            buf[0:8] = np.frombuffer(struct.pack('<II', MAGIC, n), np.uint8)
            for i, (o, b) in enumerate(zip(offs, subs)):
                buf[8 + 8 * i:16 + 8 * i] = np.frombuffer(struct.pack('<Q', o), np.uint8)
                buf[o:o + b.size] = b
            for i in range(n):
                table[-n + i]["blob_off"] = offs[i]
            h = int(hash64(buf)); off = vf.tell(); vf.write(buf.tobytes())
            S.update(off=off, size=int(pos), padded=int(padded), raw=int(raw), hash=h)
            sum_raw += raw; sum_c += pos
            print(f"\rpacked stream {si + 1}/{len(streams)}", end="", flush=True)
            if delete_consumed:
                for pth, last in list(last_use.items()):
                    if last == si and os.path.exists(pth): os.remove(pth); print(f"\ndeleted consumed shard {Path(pth).name}")
    print()
    for h, e in aliases:
        table.append(dict(name=h, shape=list(T[h]["shape"]), src_dtype=T[h]["dtype"], store="alias", alias_of=e))
    res_chunks = []; rp = 0
    for name in res_names:
        t = T[name]; f32 = res_cache[name] if name in res_cache else to_f32(load_raw(t), t["dtype"])
        table.append(dict(name=name, shape=list(t["shape"]), src_dtype=t["dtype"], store="resident_f32", res_off=rp, count=int(f32.size)))
        res_chunks.append(f32); rp += int(f32.size)
    (np.concatenate(res_chunks) if res_chunks else np.zeros(0, np.float32)).astype('<f4').tofile(vd / "resident.bin")

    # engine-compat check (nvme_engine.exe reads exactly this layout)
    ls = streams[:nl]
    compat = (nl > 0 and all(s["stems"] == KNOWN_MAT for s in ls)
              and len(streams) == nl + 1 + (0 if sc["tie"] else 1)
              and streams[nl]["kind"] == "embed"
              and (sc["tie"] == 1 or streams[nl + 1]["kind"] == "lm_head")
              and all(not lossy for _ in [0]))
    compat = compat and codec == "huffman4"      # nvme_engine.exe (original) reads huffman4 only
    if compat:
        lay_res = [t["name"] for t in table if t["store"] == "resident_f32"]
        expect = []
        for li in range(nl):
            pre = streams[li]["mats"][0].rsplit(".self_attn.", 1)[0]
            expect += [f"{pre}.{x}.weight" for x in KNOWN_NORM]
        compat = len(lay_res) == len(expect) + 1 and lay_res[:len(expect)] == expect
    mo = engine_moe_layout(cfg, sc, streams, nl, res_names, T)
    eu = engine_u_layout(cfg, sc, streams, nl, res_names, T, lossy) if mo is None else mo
    if mo is not None:
        compat = False; sc = dict(sc, inter=cfg.get("text_config", cfg).get("moe_intermediate_size", sc["inter"]))
        if not mo["supported"]: print("MoE engine cannot run this vault:", "; ".join(mo["reasons"]))
    with open(vd / "vault.idx", "w") as f:
        f.write(f"WVAULT {6 if mo is not None else (3 if compat else 4)}\nhidden {sc['hidden']}\ninter {sc['inter']}\nlayers {sc['layers']}\nheads {sc['heads']}\n"
                f"kv_heads {sc['kv_heads']}\nhead_dim {sc['head_dim']}\nvocab {sc['vocab']}\ntie {sc['tie']}\neos {sc['eos']}\neos2 {sc['eos2']}\n"
                f"eps {'%.10g' % sc['eps']}\ntheta {'%.10g' % sc['theta']}\nnstreams {len(streams)}\n")
        for i, S in enumerate(streams):
            f.write(f"stream {i} {S['off']} {S['size']} {S['padded']} {S['raw']} {S['hash']:x}\n")
        if not compat and eu["supported"]:
            f.write(eu["idx_lines"])
        if codec == "rans4":
            f.write("codec 1\n")
    gp = model_dir / "generation_config.json"
    meta = dict(format="vault3-universal", version=4, codec="huffman4-v3" if codec == "huffman4" else "rans4-m12-r16", engine_compatible_v3=bool(compat), lossy=lossy,
                config=cfg, generation_config=json.load(open(gp, encoding='utf-8')) if gp.exists() else None,
                scalars=sc, engine_u={k: v for k, v in eu.items() if k != 'idx_lines'}, streams=[{k: v for k, v in S.items() if k != "stems"} for S in streams], tensors=table)
    with open(vd / "vault.json", "w", encoding='utf-8') as f:
        json.dump(meta, f)
    copy_tokenizer(model_dir, vd)
    print(f"streams={len(streams)} raw={sum_raw} stored={sum_c} ratio={sum_c / max(sum_raw, 1):.4f} "
          f"engine_compatible_v3={compat} engine_u={eu['supported'] or compat} lossy={lossy} time_s={time.time() - t0:.1f}")
    return compat


# ------------------------------------------------------------------ read back / check / compare
def read_tensor(vault_dir, name, V=None):
    vd = Path(vault_dir)
    V = V or json.load(open(vd / "vault.json", encoding='utf-8'))
    t = next(x for x in V["tensors"] if x["name"] == name)
    if t["store"] == "alias":
        return read_tensor(vault_dir, t["alias_of"], V)
    if t["store"] == "resident_f32":
        return np.fromfile(vd / "resident.bin", '<f4', count=t["count"], offset=4 * t["res_off"]).reshape(t["shape"] or ())
    S = V["streams"][t["stream"]]
    with open(vd / "vault.bin", "rb") as f:
        f.seek(S["off"] + t["blob_off"]); blob = np.frombuffer(f.read(t["blob_size"]), np.uint8)
    if t["store"] == "rans4":
        import rans4 as _r4
        return _r4.decode(blob).reshape(t["shape"])
    return decode_blob(blob).reshape(t["shape"])


def check(model_dir, vault_dir, partial=False, hashes=True):
    T = scan_model(model_dir, allow_missing=partial); vd = Path(vault_dir)
    V = json.load(open(vd / "vault.json", encoding='utf-8')); bad = 0
    compared = 0
    for t in V["tensors"]:
        if partial and t["name"] not in T:
            continue
        compared += 1; src = T[t["name"]]; raw = load_raw(src); got = read_tensor(vd, t["name"], V)
        if t["store"] in ("huff4", "rans4", "alias"):
            exp = to_bf16_bits(raw, src["dtype"], True, t["name"]).reshape(src["shape"])
        else:
            exp = to_f32(raw, src["dtype"]).reshape(src["shape"] or ())
        ok = got.shape == exp.shape and np.array_equal(got, exp) if t["store"] in ("huff4", "rans4", "alias") else np.array_equal(got.view(np.uint32), exp.view(np.uint32))
        if not ok:
            bad += 1; print("MISMATCH", t["name"])
    for S in (V["streams"] if hashes else []):
        with open(vd / "vault.bin", "rb") as f:
            f.seek(S["off"]); b = np.frombuffer(f.read(S["padded"]), np.uint8)
        if int(hash64(b)) != S["hash"]:
            bad += 1; print("HASH MISMATCH stream", S["off"])
    print(f"check: compared {compared}/{len(V['tensors'])} tensors{'' if hashes else ' (stream hashes skipped)'}, {len(V['streams']) if hashes else 0} stream hashes, mismatches={bad}")
    return bad == 0


def compare(a, b):
    ok = True
    for fn in ("vault.bin", "resident.bin", "vault.idx"):
        pa, pb = Path(a) / fn, Path(b) / fn
        sa, sb = pa.stat().st_size, pb.stat().st_size; diff = None; pos = 0
        with open(pa, 'rb') as fa, open(pb, 'rb') as fb:
            while True:
                x = fa.read(1 << 24); y = fb.read(1 << 24)
                if not x and not y:
                    break
                if x != y:
                    m = min(len(x), len(y)); nz = np.flatnonzero(np.frombuffer(x[:m], np.uint8) != np.frombuffer(y[:m], np.uint8))
                    diff = pos + (int(nz[0]) if nz.size else m); break
                pos += len(x)
        same = diff is None and sa == sb
        ok &= same
        print(f"{fn}: {'IDENTICAL' if same else 'DIFFERENT'}  sizes {sa} / {sb}" + ("" if same else f"  first diff at byte {diff}"))
    print("compare:", "BYTE-IDENTICAL" if ok else "FAILED")
    return ok


# ------------------------------------------------------------------ selftest
def _bf16(x):
    return to_bf16_bits(x.astype(np.float32), "F32", True, "synthetic")


def _write_st(path, tensors):
    hdr = {}; pos = 0; blobs = []
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        hdr[name] = dict(dtype=dt, shape=list(arr.shape), data_offsets=[pos, pos + len(b)]); pos += len(b); blobs.append(b)
    h = json.dumps(hdr).encode(); h += b' ' * ((-len(h)) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h))); f.write(h)
        for b in blobs:
            f.write(b)


def pu_load(t):
    return load_raw(t)


def pu_expect(t, raw):
    return to_bf16_bits(raw, t["dtype"], True, "x").reshape(t["shape"])


def selftest():
    import tempfile
    rng = np.random.default_rng(0); fails = 0

    def expect(c, msg):
        nonlocal fails
        print(("PASS " if c else "FAIL ") + msg); fails += (not c)

    # 1. hash64 vs reference
    for n in (0, 1, 7, 8, 9, 64, 1000):
        b = rng.integers(0, 256, n, dtype=np.uint8)
        expect(int(hash64(b)) == hash64_ref(b.tobytes()), f"hash64 matches reference n={n}")
    # 2. huffman: Kraft, max length, exact decode, odd shapes
    for shape in ((37, 101), (5, 3), (1, 1), (4, 4), (9, 2050)):
        w = _bf16(rng.standard_normal(shape) * 0.02)
        ln = huff_lengths(_freq(w)); kraft = sum(2.0 ** -l for l in ln if l)
        expect(max(ln) <= LMAX and kraft <= 1.0 + 1e-12, f"huffman lengths valid {shape} (max {max(ln)}, kraft {kraft:.4f})")
        expect(np.array_equal(decode_blob(compress_matrix(w)), w), f"blob roundtrip {shape}")
    w = np.full((6, 17), 0x3F80, np.uint16)
    expect(np.array_equal(decode_blob(compress_matrix(w)), w), "single-symbol matrix roundtrip")
    w = rng.integers(0, 65536, (23, 130), dtype=np.uint16)
    expect(np.array_equal(decode_blob(compress_matrix(w)), w), "all-256-exponents random bits roundtrip (length limiter)")
    # 3. end-to-end qwen3-like model
    def mk(d, tie, odd, head=None):
        hid, inter, nl, hd, nh, nkv, vocab = 64, 96, 2, 16, 4, 2, 100
        cfg = dict(model_type="qwen3", hidden_size=hid, intermediate_size=inter, num_hidden_layers=nl, num_attention_heads=nh,
                   num_key_value_heads=nkv, head_dim=hd, vocab_size=vocab, rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=tie)
        json.dump(cfg, open(d / "config.json", "w"))
        r = lambda *s: _bf16(rng.standard_normal(s) * 0.02)
        T = {"model.embed_tokens.weight": ("BF16", r(vocab, hid)), "model.norm.weight": ("BF16", r(hid))}
        if not tie:
            T["lm_head.weight"] = ("BF16", r(vocab, hid))
        if head == "dup":
            T["lm_head.weight"] = T["model.embed_tokens.weight"]
        elif head == "diff":
            T["lm_head.weight"] = ("BF16", r(vocab, hid))
        for l in range(nl):
            p = f"model.layers.{l}."
            for nm, sh in (("self_attn.q_proj", (nh * hd, hid)), ("self_attn.k_proj", (nkv * hd, hid)), ("self_attn.v_proj", (nkv * hd, hid)),
                           ("self_attn.o_proj", (hid, nh * hd)), ("mlp.gate_proj", (inter, hid)), ("mlp.up_proj", (inter, hid)), ("mlp.down_proj", (hid, inter))):
                T[p + nm + ".weight"] = ("BF16", r(*sh))
            for nm, n in (("input_layernorm", hid), ("post_attention_layernorm", hid), ("self_attn.q_norm", hd), ("self_attn.k_norm", hd)):
                T[p + nm + ".weight"] = ("F16", (rng.standard_normal(n) * 0.1).astype(np.float16)) if odd else ("BF16", r(n))
        if odd:
            T["model.layers.0.conv.weight"] = ("F32", (rng.standard_normal((8, 1, 3)) * 0.1).astype(np.float32))
        _write_st(d / "model.safetensors", T)
    for label, tie, odd, conv, want_compat in (("tied qwen3-like", 1, False, False, True), ("untied qwen3-like", 0, False, False, True),
                                               ("odd (f16 norms, 3-D conv, f32 matrix)", 0, True, True, False)):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); (td / "m").mkdir(); mk(td / "m", tie, odd)
            if odd:
                try:
                    pack(td / "m", td / "v0"); expect(False, "f32 matrix refused without --convert-matrices")
                except SystemExit as e:
                    expect("convert-matrices" in str(e), "f32 matrix refused without --convert-matrices")
            compat = pack(td / "m", td / "v", allow_convert=conv)
            expect(compat == want_compat, f"{label}: engine_compatible_v3={compat}")
            expect(check(td / "m", td / "v"), f"{label}: pack -> read back bit-exact, stream hashes ok")
            if label.startswith("tied"):
                lines = open(td / "v" / "vault.idx").read().splitlines()
                expect(lines[0] == "WVAULT 3" and lines[8] == "tie 1" and "eps 1e-06" in lines and "theta 1000000" in lines, "vault.idx header format")
    # delete-consumed mode on a 2-shard copy: shards vanish, vault stays bit-exact
    import shutil
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); (td / "m").mkdir(); mk(td / "m", 0, False)
        T0 = scan_model(td / "m"); names = sorted(T0); half = len(names) // 2
        src = {n: pu_load(T0[n]) for n in names}
        for i, grp in enumerate((names[:half], names[half:])):
            _write_st(td / "m" / f"model-{i + 1}.safetensors", {n: ({"BF16": "BF16", "F16": "F16", "F32": "F32"}[T0[n]["dtype"]], src[n].reshape(T0[n]["shape"])) for n in grp})
        os.remove(td / "m" / "model.safetensors")
        pack(td / "m", td / "v", delete_consumed=True)
        left = sorted(p.name for p in (td / "m").glob("*.safetensors"))
        expect(left == [], f"delete-consumed: all source shards removed ({left})")
        good = all(np.array_equal(read_tensor(td / "v", n), pu_expect(T0[n], src[n])) for n in names if len(T0[n]["shape"]) >= 2)
        expect(good, "delete-consumed: every matrix still reads back bit-exact from the vault")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); (td / "m").mkdir(); mk(td / "m", 1, False)
        c = pack(td / "m", td / "v", codec="rans4"); idx = open(td / "v" / "vault.idx").read()
        expect((not c) and idx.startswith("WVAULT 4") and "codec 1" in idx and "supported 1" in idx, "rans4 vault: WVAULT 4 + codec 1 + layout lines")
        expect(check(td / "m", td / "v"), "rans4 vault: every tensor reads back bit-exact")
    for head, want_c, want_s in (("dup", True, 3), ("diff", False, 4)):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); (td / "m").mkdir(); mk(td / "m", 1, False, head)
            compat = pack(td / "m", td / "v")
            V = json.load(open(td / "v" / "vault.json"))
            expect(compat == want_c and len(V["streams"]) == want_s, f"tied + {head} lm_head: compat={compat} streams={len(V['streams'])}")
            expect(check(td / "m", td / "v"), f"tied + {head} lm_head: read back bit-exact")
    print(f"\nSELFTEST {'PASSED' if not fails else 'FAILED (%d)' % fails}")
    return fails == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("pack"); a.add_argument("model_dir"); a.add_argument("vault_dir"); a.add_argument("--convert-matrices", action="store_true"); a.add_argument("--delete-consumed", action="store_true"); a.add_argument("--codec", default="huffman4", choices=["huffman4", "rans4"])
    a = sp.add_parser("check"); a.add_argument("model_dir"); a.add_argument("vault_dir"); a.add_argument("--partial", action="store_true"); a.add_argument("--no-hashes", action="store_true")
    a = sp.add_parser("compare"); a.add_argument("vault_a"); a.add_argument("vault_b")
    sp.add_parser("selftest")
    x = ap.parse_args()
    if x.cmd == "pack":
        pack(x.model_dir, x.vault_dir, x.convert_matrices, x.delete_consumed, x.codec)
    elif x.cmd == "check":
        sys.exit(0 if check(x.model_dir, x.vault_dir, x.partial, not x.no_hashes) else 1)
    elif x.cmd == "compare":
        sys.exit(0 if compare(x.vault_a, x.vault_b) else 1)
    else:
        sys.exit(0 if selftest() else 1)


if __name__ == "__main__":
    main()
