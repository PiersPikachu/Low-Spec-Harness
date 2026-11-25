#!/usr/bin/env python3
"""
verify_engine.py - end-to-end proof for nvme_engine_u.exe.

  synth                       tiny random models (qwen3-like, llama3-rope, qwen2-bias, all-roles): pack -> engine -> compare
  real <model_dir> <vault>    real checkpoint + its vault: engine vs numpy reference (and vs old nvme_engine.exe if given)

The reference is plain numpy float32 read straight from the SOURCE safetensors (no vault, no C),
so engine(vault) == reference(source) proves packer + vault + engine together.
Compared: final-token logits and the last-token hidden state after every layer.
"""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pack_universal_v3 as pu

EXE_U = Path(os.environ.get("ENGINE_EXE", str(_ROOT / "engine" / "nvme_engine_u.exe")))


# ------------------------------------------------------------------ numpy reference
_PATCH = None


def _patches():
    global _PATCH
    if _PATCH is None:
        _PATCH = {}
        if os.environ.get("PATCH_JSON"):
            for q in json.load(open(os.environ["PATCH_JSON"]))["patches"]:
                m = q["module"]; m = m if m.startswith("model.") else "model." + m
                _PATCH.setdefault(m + ".weight", []).append((np.asarray(q["u"], np.float32), np.asarray(q["w"], np.float32)))
    return _PATCH


def _f32(T, name):
    t = T[name]
    W = pu.to_f32(pu.load_raw(t), t["dtype"]).reshape(t["shape"] or ())
    for u, w in _patches().get(name, []):
        W = W - np.outer(u, w)
    return W


def _rms(x, w, eps):
    inv = 1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
    return (w * (x * inv.astype(np.float32))).astype(np.float32)


def _rope_tables(hd, theta, n, rs):
    half = hd // 2; i = np.arange(half)
    fr = theta ** (-2.0 * i / hd)
    rt = (rs or {}).get("rope_type") or (rs or {}).get("type")
    if rt == "llama3":
        f, lo_f, hi_f, orig = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], rs["original_max_position_embeddings"]
        wl = 2 * np.pi / fr; lo = orig / lo_f; hi = orig / hi_f
        fl = np.where(wl > lo, fr / f, fr)
        sm = (orig / wl - lo_f) / (hi_f - lo_f)
        med = (~(wl < hi)) & (~(wl > lo))
        fr = np.where(med, (1 - sm) * fl / f + sm * fl, fl)
    a = np.arange(n)[:, None] * fr[None, :]
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def _rope(x, cs, sn):          # x (n, heads, hd), half-split rotation
    half = x.shape[-1] // 2; x0, x1 = x[..., :half], x[..., half:]
    c, s = cs[:, None, :], sn[:, None, :]
    return np.concatenate([x0 * c - x1 * s, x1 * c + x0 * s], axis=-1)


def reference(model_dir, tokens):
    cfg = json.load(open(Path(model_dir) / "config.json")); c = cfg.get("text_config", cfg)
    T = pu.scan_model(model_dir)
    nl, hid, nh = c["num_hidden_layers"], c["hidden_size"], c["num_attention_heads"]
    nkv = c.get("num_key_value_heads", nh); hd = c.get("head_dim") or hid // nh; eps = c["rms_norm_eps"]
    rp = c.get("rope_parameters") or {}; theta = c.get("rope_theta") or rp.get("rope_theta")
    rs = c.get("rope_scaling") or c.get("rope_parameters")
    n = len(tokens); cs, sn = _rope_tables(hd, theta, n, rs); grp = nh // nkv
    pre = next(k for k in T if k.endswith("layers.0.self_attn.q_proj.weight"))[:-len("0.self_attn.q_proj.weight")]
    embed = next(k for k in T if pu.EMBED_RE.search(k))
    H = _f32(T, embed)[np.array(tokens)].copy(); dumps = []
    mask = np.triu(np.full((n, n), -np.inf, np.float32), 1)
    for l in range(nl):
        g = lambda s: _f32(T, f"{pre}{l}.{s}")
        has = lambda s: f"{pre}{l}.{s}" in T
        x = _rms(H, g("input_layernorm.weight"), eps)
        q = x @ g("self_attn.q_proj.weight").T; k = x @ g("self_attn.k_proj.weight").T; v = x @ g("self_attn.v_proj.weight").T
        if has("self_attn.q_proj.bias"): q = q + g("self_attn.q_proj.bias")
        if has("self_attn.k_proj.bias"): k = k + g("self_attn.k_proj.bias")
        if has("self_attn.v_proj.bias"): v = v + g("self_attn.v_proj.bias")
        q = q.reshape(n, nh, hd); k = k.reshape(n, nkv, hd); v = v.reshape(n, nkv, hd)
        if has("self_attn.q_norm.weight"): q = _rms(q, g("self_attn.q_norm.weight"), eps)
        if has("self_attn.k_norm.weight"): k = _rms(k, g("self_attn.k_norm.weight"), eps)
        q = _rope(q, cs, sn); k = _rope(k, cs, sn)
        K = np.repeat(k, grp, axis=1); V = np.repeat(v, grp, axis=1)
        sc = np.einsum('qhd,khd->hqk', q, K) / np.float32(np.sqrt(hd)) + mask
        sc = sc - sc.max(-1, keepdims=True); pr = np.exp(sc); pr /= pr.sum(-1, keepdims=True)
        att = np.einsum('hqk,khd->qhd', pr, V).reshape(n, nh * hd)
        o = att @ g("self_attn.o_proj.weight").T
        if has("self_attn.o_proj.bias"): o = o + g("self_attn.o_proj.bias")
        H = H + o
        x = _rms(H, g("post_attention_layernorm.weight"), eps)
        ga = x @ g("mlp.gate_proj.weight").T; up = x @ g("mlp.up_proj.weight").T
        if has("mlp.gate_proj.bias"): ga = ga + g("mlp.gate_proj.bias")
        if has("mlp.up_proj.bias"): up = up + g("mlp.up_proj.bias")
        d = (ga / (1 + np.exp(-np.clip(ga, -80, 80))) * up) @ g("mlp.down_proj.weight").T
        if has("mlp.down_proj.bias"): d = d + g("mlp.down_proj.bias")
        H = H + d
        dumps.append(H[-1].copy())
    fn = next(k for k in T if k.endswith("norm.weight") and ".layers." not in k)
    head = embed if c.get("tie_word_embeddings") else next((k for k in T if pu.HEAD_RE.search(k)), embed)
    logits = _rms(H[-1:], _f32(T, fn), eps)[0] @ _f32(T, head).T
    return logits.astype(np.float32), np.stack(dumps)


# ------------------------------------------------------------------ engine driver
def run_engine(exe, vault, tokens, nl, hid, want_dump=True):
    td = tempfile.mkdtemp(); dump = os.path.join(td, "h.bin"); env = dict(os.environ)
    if want_dump: env["NVME_DUMP_H"] = dump
    p = subprocess.Popen([str(exe), "serve", str(vault), "--ctx", "64", "--threads", "2", "--cached", "--log", os.path.join(td, "log.txt")] + os.environ.get("ENGINE_EXTRA_ARGS", "").split(),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    first = p.stdout.readline()
    if first.strip() != "READY":
        err = p.stderr.read(); p.kill(); raise RuntimeError(f"engine did not start: {first!r} {err[-400:]}")
    p.stdin.write(f"LOGITS {len(tokens)}\n{' '.join(map(str, tokens))}\n"); p.stdin.flush()
    logits = None
    while True:
        ln = p.stdout.readline()
        if not ln or ln.startswith("END") or ln.startswith("ERR"):
            break
        if ln.startswith("L "):
            logits = np.array(ln[2:].split(), dtype=np.float32)
    p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=30)
    h = np.fromfile(dump, np.float32).reshape(-1, hid)[:nl] if want_dump and os.path.exists(dump) else None
    return logits, h


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def compare_one(label, model_dir, vault, tokens, tol=1e-3, exe=EXE_U):
    cfg = json.load(open(Path(model_dir) / "config.json")); c = cfg.get("text_config", cfg)
    t0 = time.time(); rl, rh = reference(model_dir, tokens); t_ref = time.time() - t0
    el, eh = run_engine(exe, vault, tokens, c["num_hidden_layers"], c["hidden_size"])
    if el is None or el.size != rl.size:
        print(f"FAIL {label}: engine returned {None if el is None else el.size} logits, expected {rl.size}"); return False
    lrel = rel(el, rl); top_e, top_r = np.argsort(-el)[:5], np.argsort(-rl)[:5]
    lay = [rel(eh[i], rh[i]) for i in range(rh.shape[0])] if eh is not None and eh.shape == rh.shape else [float('nan')]
    worst = int(np.nanargmax(lay)); ok = lrel < tol and all(x < tol for x in lay)
    print(f"{'PASS' if ok else 'FAIL'} {label}: logits relL2={lrel:.2e} maxabs={float(np.abs(el - rl).max()):.2e} argmax {'==' if top_e[0] == top_r[0] else '!='} "
          f"(engine {top_e[0]} / ref {top_r[0]}) top5 {'same' if set(top_e) == set(top_r) else 'DIFF'} | hidden relL2 worst layer {worst}: {lay[worst]:.2e} (ref {t_ref:.1f}s)")
    return ok


# ------------------------------------------------------------------ synthetic models
def make_model(d, kind, seed=1):
    rng = np.random.default_rng(seed); d = Path(d); d.mkdir(parents=True, exist_ok=True)
    hid, nh, nkv, hd, inter, nl, vocab = 64, 4, 2, 16, 112, 3, 120
    cfg = dict(hidden_size=hid, intermediate_size=inter, num_hidden_layers=nl, num_attention_heads=nh, num_key_value_heads=nkv, head_dim=hd,
               vocab_size=vocab, rms_norm_eps=1e-5, rope_theta=500000.0, hidden_act="silu", tie_word_embeddings=False)
    qk = bias = obias = mbias = False
    if kind == "llama":
        cfg.update(model_type="llama", rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=64))
    elif kind == "qwen2":
        cfg.update(model_type="qwen2", tie_word_embeddings=True, rope_theta=1000000.0, rms_norm_eps=1e-6); bias = True
    elif kind == "qwen3":
        cfg.update(model_type="qwen3", tie_word_embeddings=True, rope_theta=1000000.0, rms_norm_eps=1e-6); qk = True
    elif kind == "allroles":
        cfg.update(model_type="synthetic"); qk = bias = obias = mbias = True
    json.dump(cfg, open(d / "config.json", "w"))
    W = lambda *s: pu._bf16(rng.standard_normal(s) * 0.08)
    N = lambda n: pu._bf16(1.0 + 0.1 * rng.standard_normal(n))
    B = lambda n: pu._bf16(0.1 * rng.standard_normal(n))
    T = {"model.embed_tokens.weight": ("BF16", pu._bf16(rng.standard_normal((vocab, hid)) * 0.5)), "model.norm.weight": ("BF16", N(hid))}
    if not cfg["tie_word_embeddings"]:
        T["lm_head.weight"] = ("BF16", W(vocab, hid))
    for l in range(nl):
        p = f"model.layers.{l}."
        for nm, sh in (("self_attn.q_proj", (nh * hd, hid)), ("self_attn.k_proj", (nkv * hd, hid)), ("self_attn.v_proj", (nkv * hd, hid)),
                       ("self_attn.o_proj", (hid, nh * hd)), ("mlp.gate_proj", (inter, hid)), ("mlp.up_proj", (inter, hid)), ("mlp.down_proj", (hid, inter))):
            T[p + nm + ".weight"] = ("BF16", W(*sh))
        T[p + "input_layernorm.weight"] = ("BF16", N(hid)); T[p + "post_attention_layernorm.weight"] = ("BF16", N(hid))
        if qk:
            T[p + "self_attn.q_norm.weight"] = ("BF16", N(hd)); T[p + "self_attn.k_norm.weight"] = ("BF16", N(hd))
        if bias:
            T[p + "self_attn.q_proj.bias"] = ("BF16", B(nh * hd)); T[p + "self_attn.k_proj.bias"] = ("BF16", B(nkv * hd)); T[p + "self_attn.v_proj.bias"] = ("BF16", B(nkv * hd))
        if obias:
            T[p + "self_attn.o_proj.bias"] = ("BF16", B(hid))
        if mbias:
            T[p + "mlp.gate_proj.bias"] = ("BF16", B(inter)); T[p + "mlp.up_proj.bias"] = ("BF16", B(inter)); T[p + "mlp.down_proj.bias"] = ("BF16", B(hid))
    pu._write_st(d / "model.safetensors", T)
    return vocab


def synth():
    ok = True
    for kind in ("qwen3", "llama", "qwen2", "allroles"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); vocab = make_model(td / "m", kind)
            compat = pu.pack(td / "m", td / "v", codec=os.environ.get("PACK_CODEC", "huffman4"))
            V = json.load(open(td / "v" / "vault.json")); eu = V["engine_u"]
            print(f"  [{kind}] idx=WVAULT {3 if compat else 4}  engine_u.supported={eu['supported']}  layout={eu.get('layout')}")
            if not (eu["supported"] or compat):
                print(f"FAIL {kind}: packer says unsupported: {eu['reasons']}"); ok = False; continue
            toks = [int((17 + 31 * i) % vocab) for i in range(11)]
            ok &= compare_one(f"synthetic {kind}", td / "m", td / "v", toks)
    # packer must refuse what the engine cannot run
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); make_model(td / "m", "llama"); cfg = json.load(open(td / "m" / "config.json")); cfg["hidden_act"] = "gelu"
        json.dump(cfg, open(td / "m" / "config.json", "w")); pu.pack(td / "m", td / "v")
        eu = json.load(open(td / "v" / "vault.json"))["engine_u"]; idx = open(td / "v" / "vault.idx").read()
        good = (not eu["supported"]) and "supported 1" not in idx
        print(f"{'PASS' if good else 'FAIL'} gelu model flagged unsupported: {eu['reasons']}"); ok &= good
        r = subprocess.run([str(EXE_U), "serve", str(td / "v"), "--ctx", "8", "--cached"], capture_output=True, text=True, input="QUIT\n")
        good = r.returncode != 0 and "not supported" in r.stderr
        print(f"{'PASS' if good else 'FAIL'} engine refuses unsupported vault: {r.stderr.strip()[:90]}"); ok &= good
    print("\nVERIFY SYNTH", "PASSED" if ok else "FAILED")
    return ok


def real(model_dir, vault, old_exe=None):
    cfg = json.load(open(Path(model_dir) / "config.json")); c = cfg.get("text_config", cfg); vocab = c["vocab_size"]
    toks = [int((1000 + 7919 * i) % vocab) for i in range(12)]
    ok = compare_one(f"real {Path(model_dir).name}", model_dir, vault, toks)
    if old_exe and Path(old_exe).exists():
        a, _ = run_engine(EXE_U, vault, toks, c["num_hidden_layers"], c["hidden_size"], False)
        b, _ = run_engine(old_exe, vault, toks, c["num_hidden_layers"], c["hidden_size"], False)
        same = a is not None and b is not None and np.array_equal(a[:b.size], b)
        print(f"{'PASS' if same else 'FAIL'} old engine vs engine_u: first {0 if b is None else b.size} logits bit-identical"); ok &= same
    print("\nVERIFY REAL", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("synth"); a = sp.add_parser("real"); a.add_argument("model_dir"); a.add_argument("vault"); a.add_argument("--old-exe")
    x = ap.parse_args()
    sys.exit(0 if (synth() if x.cmd == "synth" else real(x.model_dir, x.vault, x.old_exe)) else 1)
