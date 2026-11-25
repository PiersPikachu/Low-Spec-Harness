#!/usr/bin/env python3
"""model_hub.py - find Hugging Face models this harness can run on THIS machine, install them (download -> compress -> check),
and tune each compressed model's engine settings before it is used.

  search(query)       Hugging Face text-generation models, most downloaded first
  assess(repo)        the decision tree: architecture -> weights format -> compressed size -> disk -> memory tier -> speed -> verdict
                      (sizes from the repo's safetensors metadata, layer/expert shapes from its config.json, speeds from the
                       measured calibration of this machine - an estimate, roughly +-30%)
  install(repo, ...)  snapshot download (weights, config, tokenizer only) -> pack (huffman4 lossless, or MXF2 for gpt-oss MXFP4) ->
                      stream check -> tune
  tune(vault)         short benchmark of the candidate engine settings for that model; the fastest one that produces a sane
                      reply is saved to <vault>/tuned.json and used by 'Model defaults'
CLI:  model_hub.py search <words> | assess <repo> | install <repo> | tune <vault>"""
import json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import harness_paths as hp

CACHE = hp.HUB_CACHE
DENSE = {"llama", "qwen2", "qwen3", "mistral"}
MOE = {"qwen3_moe": "qwen3_moe", "gpt_oss": "gpt_oss"}
BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "I32": 4, "I64": 8}
# calibration measured on this laptop (Extreme-scale log X5-X14 and the planner probe)
K_MOE_MXF = 0.065      # s per billion active parameters, MXFP4 experts (CPU+GPU), gpt-oss-20b 3.66 tok/s
K_MOE_BF16 = 0.136     # s per billion active parameters, huffman bf16 experts (CPU), Qwen3-30B-A3B
GPU_GBPS = 45e9        # effective decode bandwidth of a GPU-resident dense vault (Qwen3-0.6B: ~55 tok/s)
SPEC_TPP = 3.5         # tokens per pass with the Qwen3-0.6B draft (measured k=8 mean, SSD-bound target)
SPEC_TPP_RAM = 2.5     # speculative gain when the target is decode-bound in RAM (verifying k tokens costs more there; conservative)


def api():
    from huggingface_hub import HfApi
    return HfApi()


def search(query, limit=40):
    out = []
    for m in api().list_models(search=query or None, filter="text-generation", sort="downloads", limit=limit, expand=["downloads", "likes", "gated", "lastModified", "safetensors"]):
        st = getattr(m, "safetensors", None)
        out.append(dict(repo=m.id, downloads=m.downloads or 0, likes=m.likes or 0, gated=bool(m.gated), params=(st.total if st else None)))
    return out


def hardware():
    import planner as pl
    H = pl.probe(); cfg = hp.load_config()
    vd = hp.default_model_dir(cfg); dd = Path(cfg.get("hub_download_dir") or (r"D:\models" if Path("D:/").exists() else hp.DATA / "downloads"))
    for d in (vd, dd):
        try: d.mkdir(parents=True, exist_ok=True)
        except Exception: pass
    free = lambda d: shutil.disk_usage(d).free if d.exists() else 0
    ram_w = max(2e9, (H.get("ram_total_mb", 16000) - 6500) * 1048576)          # RAM the engine can pin for weights / experts
    return dict(H=H, vault_dir=vd, dl_dir=dd, vault_free=free(vd), dl_free=free(dd), ram_w=ram_w, vram=max(0, H.get("vram_free_mb", 0) - 900) * 1048576,
                disk_Bps=2.5e9, cpu_Bps=H.get("cpu_decode_GBps", 14.5) * 1e9)      # 2.5 GB/s: measured on the contiguous 14B vault


def _config(repo):
    from huggingface_hub import hf_hub_download
    CACHE.mkdir(exist_ok=True)
    return json.load(open(hf_hub_download(repo, "config.json", cache_dir=str(CACHE))))


def assess(repo, hw=None):
    """-> dict(repo, verdict, ok, tree=[(step, answer, status)], kind, packer, est_tok_s, vault_gb, download_gb, notes)"""
    hw = hw or hardware(); tree = []; R = dict(repo=repo, ok=False, kind=None, packer=None, est_tok_s=None, vault_gb=None, download_gb=None)
    def step(q, a, st): tree.append((q, a, st))
    try: info = api().model_info(repo, expand=["safetensors", "gated", "config"])
    except Exception as ex:
        step("Repository", f"cannot read it: {ex}", "bad"); R.update(tree=tree, verdict="unavailable"); return R
    if info.gated: step("Access", "gated: accept the licence on huggingface.co and log in (huggingface-cli login) before installing", "warn")
    st = info.safetensors
    if not st:
        step("Weights", "no safetensors weights (GGUF-only or PyTorch .bin) - this harness packs safetensors", "bad"); R.update(tree=tree, verdict="not compatible"); return R
    try: c = _config(repo)
    except Exception as ex:
        if info.gated:
            step("config.json", "hidden until you accept the licence and log in (huggingface-cli login), so compatibility cannot be checked yet", "warn")
            R.update(tree=tree, verdict="needs Hugging Face login"); return R
        step("config.json", f"missing or unreadable: {str(ex).splitlines()[0][:160]}", "bad"); R.update(tree=tree, verdict="not compatible"); return R
    c = c.get("text_config", c); mt = str(c.get("model_type", "?")); q = c.get("quantization_config") or (info.config or {}).get("quantization_config") or {}
    # 1 architecture
    if mt in MOE: kind = "gptoss" if mt == "gpt_oss" else "moe"
    elif mt in DENSE and not (mt == "mistral" and c.get("sliding_window")): kind = "dense"
    else:
        step("Architecture", f"{mt}: not supported by this engine (supported: {', '.join(sorted(DENSE | set(MOE)))})", "bad"); R.update(tree=tree, verdict="not compatible"); return R
    step("Architecture", f"{mt} ({'mixture of experts' if kind != 'dense' else 'dense decoder'}) - supported", "ok")
    # 2 weight format
    params = st.parameters or {}; total = st.total or sum(params.values()); qm = str(q.get("quant_method", "")).lower()
    if qm and not (qm == "mxfp4" and kind == "gptoss"):
        step("Weights format", f"pre-quantized ({qm}): the packer needs bf16/fp16 (or gpt-oss MXFP4) weights - look for the original repo", "bad"); R.update(tree=tree, verdict="not compatible"); return R
    # Hugging Face counts gpt-oss MXFP4 tensors (U8) in weights, not bytes: 4.25 bits each on disk
    raw = sum(n * (4.25 / 8 if (k == "U8" and kind == "gptoss") else BYTES.get(k, 2)) for k, n in params.items()); dl = raw
    step("Weights format", ", ".join(f"{k} {n / 1e9:.2f}B" for k, n in params.items()) + f" = {raw / 1e9:.1f} GB to download", "ok")
    # 3 compressed size
    if kind == "gptoss":
        vault = (params.get("BF16", 0) * 2 * 0.67) + params.get("U8", 0) * 4.25 / 8 * 0.965; packer = "pack_gptoss.py --mxf2"
        step("Compression", f"bf16 parts huffman4 (lossless, ~67%), MXFP4 experts MXF2 (exact to the release, ~96.5%) -> vault ~{vault / 1e9:.1f} GB", "ok")
    else:
        vault = raw * (0.67 if any(k in params for k in ("BF16", "F16")) else 0.335); packer = "pack_universal_v3.py"
        step("Compression", f"huffman4, lossless (every matrix verified bit-exact) -> vault ~{vault / 1e9:.1f} GB (~67% of bf16)", "ok")
    R.update(kind=kind, packer=packer, vault_gb=vault / 1e9, download_gb=dl / 1e9)
    # 4 disk
    ok_disk = True
    if hw["dl_free"] < dl * 1.05 + 2e9: step("Disk: download", f"{hw['dl_dir']} has {hw['dl_free'] / 1e9:.0f} GB free, needs ~{dl / 1e9 * 1.05 + 2:.0f} GB", "bad"); ok_disk = False
    else: step("Disk: download", f"{hw['dl_dir']}: {hw['dl_free'] / 1e9:.0f} GB free", "ok")
    if hw["vault_free"] < vault + 2e9: step("Disk: vault", f"{hw['vault_dir']} has {hw['vault_free'] / 1e9:.0f} GB free, needs ~{vault / 1e9 + 2:.0f} GB (fast NVMe)", "bad"); ok_disk = False
    else: step("Disk: vault", f"{hw['vault_dir']}: {hw['vault_free'] / 1e9:.0f} GB free", "ok")
    # 5 memory tier and speed
    L = int(c.get("num_hidden_layers", 0)); hid = int(c.get("hidden_size", 0)); fast = hw["ram_w"] + hw["vram"]
    if kind == "dense":
        spec = mt in ("qwen3", "qwen2") and 151000 <= int(c.get("vocab_size", 0)) <= 152100 and vault > 1.5e9
        if vault <= hw["vram"] - 0.5e9: tier = "fits in GPU memory"; tps = GPU_GBPS / vault
        elif vault <= hw["ram_w"]:
            tier = "fits in RAM (CPU decode)"; tps = 0.95 * hw["cpu_Bps"] / vault * (SPEC_TPP_RAM if spec else 1.0)
        else:
            # pass = SSD stream of what is not pinned + ~60% of the decode overlapped badly (Hermes-4-14B: 4.8 s measured, 4.8 predicted)
            tier = f"streams {(vault - hw['ram_w']) / 1e9:.1f} GB per pass from the SSD"
            t_pass = (vault - hw["ram_w"]) / hw["disk_Bps"] + 0.6 * vault / hw["cpu_Bps"]
            tps = (SPEC_TPP if spec else 1.0) / t_pass
        if spec and vault > hw["vram"] - 0.5e9: tier += "; speculative decoding with the Qwen3-0.6B draft (same tokenizer)"
    else:
        E = int(c.get("num_experts") or c.get("num_local_experts") or 1); K = int(c.get("num_experts_per_tok", 1))
        I = int(c.get("moe_intermediate_size") or c.get("intermediate_size", 0)); per_exp = 3 * I * hid
        ex_total = L * E * per_exp; active = total - ex_total + L * K * per_exp if total > ex_total else total
        bpp = (4.25 / 8 * 0.965) if kind == "gptoss" else 2 * 0.67
        ex_bytes = ex_total * bpp; c_frac = min(1.0, max(0.0, (fast - (vault - ex_bytes)) / max(ex_bytes, 1)))
        hit = 1.0 if c_frac >= 1 else (0.95 * (c_frac / 0.525) ** 0.5 if c_frac < 0.525 else 0.95 + 0.05 * (c_frac - 0.525) / 0.475)
        io = (1 - hit) * L * K * per_exp * bpp / hw["disk_Bps"]; comp = active / 1e9 * (K_MOE_MXF if kind == "gptoss" else K_MOE_BF16)
        tps = 1 / (max(io, comp) + 0.3 * min(io, comp) + 0.05)
        tier = f"{E} experts/layer, top {K}, ~{active / 1e9:.1f}B active; RAM + VRAM hold ~{100 * c_frac:.0f}% of experts (~{100 * hit:.0f}% hits)"
        R["active_b"] = active / 1e9
    st_speed = "ok" if tps >= 2 else ("warn" if tps >= 0.5 else "bad")
    step("Memory", tier, "ok" if tps >= 0.5 else "warn")
    step("Speed (estimate +-30%)", f"~{tps:.2f} tokens/s on this machine", st_speed)
    ctx_max = int(c.get("max_position_embeddings") or 0)
    if ctx_max: step("Context", f"model maximum {ctx_max:,} tokens; the harness sizes it to the KV cache it can afford", "ok")
    R.update(est_tok_s=tps, ok=ok_disk and tps >= 0.5)
    R["verdict"] = ("runs well" if tps >= 2 else "runs slowly" if tps >= 0.5 else "too slow") + ("" if ok_disk else " - not enough disk space")
    R["tree"] = tree; return R


# ------------------------------------------------------------------ install
def _run(cmd, log, cwd=None):
    log(" ".join(str(c) for c in cmd))
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd or str(hp.ROOT), bufsize=1)
    last = 0.0
    for line in p.stdout:
        line = line.rstrip()
        if line and (time.time() - last > 2 or "error" in line.lower() or "streams=" in line or "vault OK" in line): log("  " + line[-200:]); last = time.time()
    if p.wait(): raise RuntimeError(f"step failed ({p.returncode}): {cmd[1] if len(cmd) > 1 else cmd[0]}")


def install(repo, log=print, tune_after=True, stop=lambda: False):
    """download -> pack -> check -> tune; returns the vault path"""
    hw = hardware(); a = assess(repo, hw)
    if a["kind"] is None: raise RuntimeError(f"{repo}: {a['verdict']}")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", repo.split("/")[-1]); src = hw["dl_dir"] / name
    vd = hw["vault_dir"] / ("vault_" + name.lower().replace(".", "_") + ("_mxf2" if a["kind"] == "gptoss" else "_huff"))
    log(f"[1/4] downloading {repo} (~{a['download_gb']:.1f} GB) to {src}")
    from huggingface_hub import snapshot_download
    snapshot_download(repo, local_dir=str(src), allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.model", "*.txt"],
                      ignore_patterns=["original/*", "metal/*", "*.pth"], max_workers=4)
    if stop(): raise RuntimeError("stopped")
    log(f"[2/4] compressing into {vd} ({a['packer']})")
    if vd.exists(): shutil.rmtree(vd)
    if a["kind"] == "gptoss": _run([sys.executable, hp.TOOLS / "pack_gptoss.py", src, vd, "--mxf2"], log)
    else: _run([sys.executable, hp.TOOLS / "pack_universal_v3.py", "pack", src, vd], log)
    log("[3/4] checking every stream of the vault"); _run([hp.ENGINE, "check", vd], log)
    if tune_after and not stop(): log("[4/4] tuning the engine settings for this machine"); tune(vd, log)
    log(f"installed: {vd}  (source kept in {src}; delete it to free space)"); return vd


# ------------------------------------------------------------------ tune
PROMPTS = ["In two sentences, explain what a compiler does.", "Write a Python function that reverses a string."]


def _loops(text):
    t = re.sub(r"\s+", " ", text)
    return any(t.count(t[i:i + 24]) >= 5 for i in range(0, max(0, len(t) - 24), 12)) if len(t) > 200 else False


def _engine_key(vault, cfg):
    """the exact engine command line this configuration launches (nothing is loaded), or None if it is not possible here"""
    import chat, io, contextlib, types
    a = types.SimpleNamespace(mode=cfg["mode"], gpu_layers=int(cfg.get("gpu_layers") or 0), ctx=int(cfg["ctx"]), draft=cfg.get("draft") or "auto",
                              kv16=bool(cfg.get("kv16")), moe_prefetch=cfg.get("moe_prefetch"), moe_vram=cfg.get("moe_vram"), temp=0.0, top_p=1.0, top_k=1, k=int(cfg.get("k") or 8))
    try:
        with contextlib.redirect_stdout(io.StringIO()): args, how, _ = chat.pick_engine(vault, a)
    except (SystemExit, Exception): return None, None, None
    spec = isinstance(args, tuple)
    return json.dumps(args), (f"k={a.k}" if spec else ""), how


def tune(vault, log=print, tokens=64):
    import harness_core as core
    vault = Path(vault); base = dict(hp.load_config()); rec, _ = core.recommended_settings(vault); base.update(rec); base.update(vault=str(vault), greedy=True, max_new=tokens, think=False)
    idx = dict(l.split()[:2] for l in open(vault / "vault.idx") if len(l.split()) == 2)
    moe, gptoss = idx.get("moe") == "1", idx.get("gptoss") == "1"
    try:
        import storage; sp = storage.read_speed(vault.parent)
        if sp and sp < 500: log(f"note: {vault.name} is on a slow drive ({sp:.0f} MB/s) - tuning will measure the drive; move it to an SSD folder first for real numbers")
    except Exception: pass
    if gptoss: cands = [dict(moe_prefetch=2, moe_vram=True), dict(moe_prefetch=0, moe_vram=True), dict(moe_prefetch=2, moe_vram=False)]
    elif moe: cands = [dict(moe_vram=True), dict(moe_vram=False)]
    else:
        cands = [dict(mode=m, draft=d, k=k) for m in ("gpu", "split", "cpu") for d, ks in (("none", (8,)), ("auto", (8, 4))) for k in ks]
    plans, seen = [], set()
    for cd in cands:
        key, kk, how = _engine_key(vault, dict(base, **cd))
        if key is None or (key, kk) in seen: continue
        seen.add((key, kk)); plans.append((cd, key, kk, how))
    log(f"{len(plans)} distinct configuration(s) to test (of {len(cands)} candidates; duplicates and impossible layouts skipped): " + "; ".join(p[3] for p in plans))
    results, best = [], 0.0; held = None                              # held = (engine key, session) kept for a k-only change
    for cd, key, kk, how in plans:
        cfg = dict(base, **cd); t0 = time.time()
        try:
            if held and held[0] == key: s = held[1]; s.cfg.update(cd); setattr(s.eng, "k", int(cd.get("k", 8)))
            else:
                if held: held[1].close(); held = None
                s = core.Session(cfg)
            speeds, texts, early = [], [], False
            for i, p in enumerate(PROMPTS):
                new, st = s.reply([{"role": "user", "content": p}], "", None, {}, lambda k, t: None, lambda: False, 0)
                m = re.search(r"([\d.]+) tok/s decode", st); speeds.append(float(m.group(1)) if m else 0.0); texts.append(new[-1]["content"] + new[-1].get("thinking", ""))
                if i == 0 and best and speeds[0] < 0.6 * best: early = True; break
                if speeds[-1] < 1.5 and base["max_new"] > 32:      # slow model: 32-token replies measure the speed just as well
                    base["max_new"] = 32; s.cfg["max_new"] = 32; log("  (slow model: test replies shortened to 32 tokens)")
            held = (key, s)
            sane = all(t.strip() for t in texts) and not any(_loops(t) for t in texts)
            r = dict(settings=cd, tok_s=round(sum(speeds) / len(speeds), 3), sane=sane, how=s.how, secs=round(time.time() - t0), early_stop=early)
            if sane: best = max(best, r["tok_s"])
        except Exception as ex:
            r = dict(settings=cd, tok_s=0.0, sane=False, error=str(ex)[-300:], secs=round(time.time() - t0))
            if held: held[1].close(); held = None
        results.append(r)
        log(f"  {r.get('how', how)} {cd}: {r['tok_s']} tok/s in {r['secs']} s" + ("  (stopped early: slower)" if r.get("early_stop") else "")
            + ("" if r["sane"] else "  (rejected: " + (r.get("error") or "empty or looping reply") + ")"))
    if held: held[1].close()
    good = [r for r in results if r["sane"]]
    if not good: log("tuning found no working setting - keeping the recommended defaults"); return None
    best_r = max(good, key=lambda r: r["tok_s"])
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), settings=best_r["settings"], tok_s=best_r["tok_s"], results=results)
    json.dump(out, open(vault / "tuned.json", "w"), indent=1)
    log(f"best: {best_r.get('how', '')} {best_r['settings']} -> {best_r['tok_s']} tok/s in {sum(r['secs'] for r in results)} s total (saved to {vault.name}/tuned.json)")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "search"
    if cmd == "search":
        for m in search(" ".join(sys.argv[2:]), 15): print(f"{m['repo']:55s} {m['downloads']:>10,}  params {m['params'] / 1e9 if m['params'] else 0:.1f}B")
    elif cmd == "assess":
        a = assess(sys.argv[2]); print(a["repo"], "->", a["verdict"])
        for q, ans, st in a["tree"]: print(f"  [{st:4s}] {q}: {ans}")
    elif cmd == "install": install(sys.argv[2])
    elif cmd == "tune": tune(hp.resolve_vault(sys.argv[2]))
