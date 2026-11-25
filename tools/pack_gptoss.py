#!/usr/bin/env python3
"""pack_gptoss.py <hf_model_dir> <vault_dir> - pack an OpenAI gpt-oss checkpoint (gpt-oss-20b / -120b) into a WVAULT 7 vault.

Exact to the release: bf16 tensors (attention, router, embedding, lm_head) are entropy-coded with huffman4 and verified bit-exact;
the MXFP4 expert weights are stored byte-for-byte as released (FP4 codes + E8M0 block scales; entropy coding them is ~8% smaller,
measured in X3a, and left for later). Each (layer, expert) is one stream so the engine reads only routed experts.
  streams: [layer l: q, k, v, o, router] x L, [embed], [lm_head], [expert (l, e): MXF4 gate_up, F32 gate_up bias, MXF4 down, F32 down bias]
  resident.bin per layer: ln_in, ln_post, bq, bk, bv, bo, sinks, router bias; then the final norm
  vault.idx: dense keys + gptoss 1, n_experts, top_k, moe_inter, expert_stream0, sliding_window, layer_types (1 = sliding), swiglu_limit,
             swiglu_alpha, rope_scale (YaRN attention factor), rope_inv (the YaRN inverse frequencies, computed by transformers)
"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import json, struct, sys, time
from pathlib import Path
import numpy as np
from safetensors import safe_open
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import pack_universal_v3 as pu

SECT = 4096


def mxf4_blob(blocks, scales):
    """blocks [rows, cols/32, 16] uint8, scales [rows, cols/32] uint8 -> MXF4 sub-blob (stored as released)"""
    rows, nb = scales.shape; hdr = struct.pack("<4sII", b"MXF4", rows, nb * 32)
    return np.frombuffer(hdr + scales.astype(np.uint8).tobytes() + blocks.astype(np.uint8).tobytes(), np.uint8)


def mxf2_blob(blocks, scales):
    """MXFP4 with packed block scales (lossless): per row, scales = base + 2-bit offset when the row's range fits (98.5% of rows in
    gpt-oss-120b), else the row's raw scale bytes in an exception table. FP4 codes unchanged.
    layout: 'MXF2' rows cols pb nexc blocks_off | desc u32[rows] (bit31 = exception, low bits = base or exception index)
            | packed u8[rows][pb] (4 offsets per byte, low bits first) | exceptions u8[nexc][nb] | pad to 16 | blocks u8[rows][nb*16]"""
    rows, nb = scales.shape; pb = ((nb + 3) // 4 + 3) & ~3; S = scales.astype(np.int16)
    lo = S.min(1); fit = (S.max(1) - lo) <= 3; exc = np.nonzero(~fit)[0]
    desc = np.where(fit, lo, 0).astype(np.uint32); desc[exc] = (1 << 31) | np.arange(len(exc), dtype=np.uint32)
    off = np.where(fit[:, None], S - lo[:, None], 0).astype(np.uint8); off = np.pad(off, ((0, 0), (0, pb * 4 - nb)))
    packed = (off[:, 0::4] | (off[:, 1::4] << 2) | (off[:, 2::4] << 4) | (off[:, 3::4] << 6)).astype(np.uint8)
    hdr_len = 24; body = desc.tobytes() + packed.tobytes() + scales[exc].astype(np.uint8).tobytes()
    boff = (hdr_len + len(body) + 15) & ~15
    head = struct.pack("<4sIIIII", b"MXF2", rows, nb * 32, pb, len(exc), boff)
    out = head + body + b"\0" * (boff - hdr_len - len(body)) + blocks.astype(np.uint8).tobytes()
    # verify: decode the scales back and compare
    dsc = np.frombuffer(out, np.uint32, rows, hdr_len); pk = np.frombuffer(out, np.uint8, rows * pb, hdr_len + 4 * rows).reshape(rows, pb)
    un = np.stack([(pk >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(rows, pb * 4)[:, :nb].astype(np.int16) + (dsc & 255).astype(np.int16)[:, None]
    if len(exc): un[exc] = scales[exc]
    assert np.array_equal(un.astype(np.uint8), scales.astype(np.uint8)), "MXF2 scale packing not lossless"
    return np.frombuffer(out, np.uint8)


def f32_blob(a):
    a = np.ascontiguousarray(a, dtype="<f4").ravel(); return np.frombuffer(struct.pack("<4sI", b"F32 ", a.size) + a.tobytes(), np.uint8)


def stream_bytes(subs):
    n = len(subs); hdr = max(128, (8 + 8 * n + 63) & ~63); pos = hdr; offs = []
    for b in subs: pos = (pos + 63) & ~63; offs.append(pos); pos += b.size
    padded = (pos + SECT - 1) & ~(SECT - 1); buf = np.zeros(padded, np.uint8)
    buf[0:8] = np.frombuffer(struct.pack("<II", pu.MAGIC, n), np.uint8)
    for i, (o, b) in enumerate(zip(offs, subs)): buf[8 + 8 * i:16 + 8 * i] = np.frombuffer(struct.pack("<Q", o), np.uint8); buf[o:o + b.size] = b
    return buf, pos


def main():
    src, vd = Path(sys.argv[1]), Path(sys.argv[2]); vd.mkdir(parents=True, exist_ok=True); t0 = time.time()
    cfg = json.load(open(src / "config.json")); L = cfg["num_hidden_layers"]; E = cfg["num_local_experts"]; K = cfg["num_experts_per_tok"]
    hid, nh, nkv, hd, inter = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["intermediate_size"]
    idx = src / "model.safetensors.index.json"
    wm = json.load(open(idx))["weight_map"] if idx.exists() else {k: "model.safetensors" for k in safe_open(str(src / "model.safetensors"), "np").keys()}
    files = {}
    def get(name):
        f = wm[name]
        if f not in files:                      # one shard mapped at a time: Windows counts each mapped shard against the commit limit
            files.clear(); files[f] = safe_open(str(src / f), "pt")
        return files[f].get_tensor(name)
    def bf16_bits(t):
        import torch
        assert t.dtype == torch.bfloat16, t.dtype; return t.contiguous().view(torch.int16).numpy().view(np.uint16)
    def f32(t): return t.float().numpy()
    streams = []; res = []; raw_total = 0
    with open(vd / "vault.bin", "wb") as vf:
        def put(subs, raw):
            buf, size = stream_bytes(subs); off = vf.tell(); vf.write(buf.tobytes())
            streams.append(dict(off=off, size=size, padded=buf.size, raw=raw, hash=int(pu.hash64(buf))))
        def huff(name):
            w = bf16_bits(get(name)); w = w.reshape(w.shape[0], -1); b = pu.compress_matrix(w)
            if not np.array_equal(pu.decode_blob(b), w): raise SystemExit(f"VERIFY FAILED {name}")
            return b, 2 * w.size
        for l in range(L):
            p = f"model.layers.{l}."; subs = []; raw = 0
            for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.router"):
                b, r = huff(p + m + ".weight"); subs.append(b); raw += r
            put(subs, raw); raw_total += raw
            for n in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.bias", "self_attn.k_proj.bias",
                      "self_attn.v_proj.bias", "self_attn.o_proj.bias", "self_attn.sinks", "mlp.router.bias"):
                res.append(f32(get(p + n)))
            print(f"\rlayer streams {l + 1}/{L}", end="", flush=True)
        for n in ("model.embed_tokens.weight", "lm_head.weight"):
            b, r = huff(n); put([b], r); raw_total += r
        print(); e0 = len(streams)
        for l in range(L):
            p = f"model.layers.{l}.mlp.experts."
            GB, GS, GBi = get(p + "gate_up_proj_blocks").numpy(), get(p + "gate_up_proj_scales").numpy(), f32(get(p + "gate_up_proj_bias"))
            DB, DS, DBi = get(p + "down_proj_blocks").numpy(), get(p + "down_proj_scales").numpy(), f32(get(p + "down_proj_bias"))
            for e in range(E):
                mx = mxf2_blob if "--mxf2" in sys.argv else mxf4_blob
                subs = [mx(GB[e], GS[e]), f32_blob(GBi[e]), mx(DB[e], DS[e]), f32_blob(DBi[e])]
                put(subs, GB[e].nbytes + GS[e].nbytes + DB[e].nbytes + DS[e].nbytes); raw_total += streams[-1]["raw"]
            print(f"\rexpert layers {l + 1}/{L}", end="", flush=True)
        print()
    res.append(f32(get("model.norm.weight"))); np.concatenate([r.ravel() for r in res]).astype("<f4").tofile(vd / "resident.bin")
    # YaRN inverse frequencies and attention factor, exactly as transformers computes them
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    c = AutoConfig.from_pretrained(str(src)); rt = c.rope_parameters["rope_type"]
    inv, ascale = ROPE_INIT_FUNCTIONS[rt](c, "cpu") if rt != "default" else (None, 1.0)
    inv = inv.float().numpy()
    lt = [1 if t == "sliding_attention" else 0 for t in cfg.get("layer_types", ["full_attention"] * L)]
    eos = [cfg.get("eos_token_id")]; gp = src / "generation_config.json"
    if gp.exists():
        g = json.load(open(gp)).get("eos_token_id"); eos = g if isinstance(g, list) else [g]
    lres = 2 * hid + nh * hd + 2 * nkv * hd + hid + nh + E
    o = {"ln_in": 0, "ln_post": hid, "bq": 2 * hid, "bk": 2 * hid + nh * hd, "bv": 2 * hid + nh * hd + nkv * hd, "bo": 2 * hid + nh * hd + 2 * nkv * hd}
    with open(vd / "vault.idx", "w") as f:
        f.write(f"WVAULT 7\nhidden {hid}\ninter {inter}\nlayers {L}\nheads {nh}\nkv_heads {nkv}\nhead_dim {hd}\nvocab {cfg['vocab_size']}\ntie 0\n"
                f"eos {eos[0]}\neos2 {eos[-1]}\neps {cfg['rms_norm_eps']:.10g}\ntheta {cfg.get('rope_theta', 0):.10g}\nnstreams {len(streams)}\n")
        for i, s in enumerate(streams): f.write(f"stream {i} {s['off']} {s['size']} {s['padded']} {s['raw']} {s['hash']:x}\n")
        f.write(f"gptoss 1\nmoe 1\nn_experts {E}\ntop_k {K}\nmoe_inter {inter}\nexpert_stream0 {e0}\nsliding_window {cfg.get('sliding_window', 0)}\n"
                f"swiglu_limit {cfg.get('swiglu_limit', 7.0):.10g}\nswiglu_alpha 1.702\nrope_scale {float(ascale):.10g}\nlres {lres}\n"
                + "".join(f"r_{k} {v}\n" for k, v in o.items()) + f"r_sinks {o['bo'] + hid}\nr_rbias {o['bo'] + hid + nh}\nsupported 1\n"
                + "layer_types " + " ".join(map(str, lt)) + "\nrope_inv " + " ".join(f"{x:.9g}" for x in inv) + "\n")
    meta = dict(format="vault-gptoss", version=7, config=cfg, streams=streams)
    json.dump(meta, open(vd / "vault.json", "w"))
    pu.copy_tokenizer(src, vd)
    stored = sum(s["size"] for s in streams)
    print(f"streams={len(streams)} stored={stored / 1e9:.2f} GB (release tensors {raw_total / 1e9:.2f} GB) time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
