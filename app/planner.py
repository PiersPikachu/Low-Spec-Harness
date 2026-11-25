#!/usr/bin/env python3
"""
planner.py - automatic execution-mode planner for compressed vaults.

Looks at (a) the model's per-token compressed bytes (vault.json) and (b) THIS machine
(free VRAM, free RAM, NVMe read speed, PCIe speed, GPU/CPU decode rate - all measured),
then picks where every layer lives and who executes it, and says so in a one-line banner.

  planner.py probe [--force]                    measure the machine (cached in hw_profile.json)
  planner.py plan  <vault_dir> [--ctx N]        mode table + banner for a vault on disk
  planner.py whatif --params-b 27 --layers 62   same for a model you have not packed (bytes = params*2*ratio)
  planner.py bench <vault_dir>                  run the CPU engine with the planned residency, compare predicted vs measured
  planner.py serve <vault_dir> [engine opts]    plan, print banner, launch the best AVAILABLE engine

Cost model (seconds per token, ideal overlap of transfer and compute):
  GPU-RESIDENT      all path bytes in VRAM                    T = P/Rg
  GPU STREAMING     VRAM part + rest streamed over PCIe       T = max(P/Rg, (R+N)/Bpcie, N/Bnvme)
  GPU+CPU SPLIT     VRAM part on GPU, rest executed by CPU    T = V/Rg + max((R+N)/Rc, N/Bnvme)
  CPU               RAM part + NVMe remainder                 T = max(P/Rc, N/Bnvme)
  (+ per-layer GPU launch overhead; V/R/N = bytes in VRAM / RAM / NVMe, P = V+R+N)
"""
import argparse, ctypes, json, math, os, re, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import harness_paths as hp
PROFILE = hp.PROFILE
ENGINE = hp.ENGINE
GPU_ENGINE = hp.GPU_ENGINE        # G2 - not built yet
MB = 1e6; GB = 1e9
WORKSPACE = 512 * MB            # GPU activations, LUTs, CUDA context slack
RAM_RESERVE = 1500 * MB         # same rule the C engine uses
GPU_LAYER_OVERHEAD = 60e-6      # ASSUMPTION until the GPU engine exists: ~10 kernel launches per layer
RATIO = 0.67                    # measured: Qwen3-0.6B .674, Llama-3.2-1B .669, Qwen3-4B .670


# ------------------------------------------------------------------ machine probes
def ram_mb():
    class M(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong),
                    ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = M(); m.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.tp / 1048576, m.ap / 1048576


def run_gpu_probe():
    exe = hp.ENGINE_DIR / "gpu" / "bw_probe.exe"
    if not exe.exists():
        return None
    try:
        return json.loads(subprocess.run([str(exe)], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[-1])
    except Exception:
        return None


def interp(points, x, default):
    """piecewise-linear in log(x) through [[x, y], ...]; clamps outside the measured range"""
    if not points:
        return default
    pts = sorted(points)
    if x <= pts[0][0]: return pts[0][1]
    if x >= pts[-1][0]: return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0)); return y0 + t * (y1 - y0)
    return pts[-1][1]


def avg_layer_bytes(vault):
    V = json.load(open(Path(vault) / "vault.json")); nl = V["scalars"]["layers"] if "scalars" in V else int(idx_kv(vault)["layers"])
    return sum(V["streams"][i]["padded"] for i in range(nl)) / nl


def find_vaults():
    """(path, size) of every packed vault the harness knows (vaults/, legacy vault_* folders, extra folders), smallest first"""
    import harness_paths as hp
    v = [(p, (p / "vault.bin").stat().st_size) for p in hp.find_vaults().values()
         if (p / "vault.json").exists() and (p / "vault.bin").stat().st_size > 0]
    return sorted(set(v), key=lambda t: t[1])


def probe_nvme(vault):
    with tempfile.TemporaryDirectory() as td:
        log = os.path.join(td, "l.txt")
        subprocess.run([str(ENGINE), "check", str(vault), "--log", log], capture_output=True, text=True, timeout=600)
        mb = ms = 0.0; rates = []
        for ln in open(log):
            m = re.search(r"\[check\] stream=\d+ MB=([\d.]+) read_ms=([\d.]+) read_MBps=(\d+)", ln)
            if m:
                mb += float(m[1]); ms += float(m[2]); rates.append(int(m[3]))
    return dict(nvme_MBps=mb / (ms / 1e3) if ms else 0, nvme_min_MBps=min(rates) if rates else 0, nvme_max_MBps=max(rates) if rates else 0, nvme_vault=str(vault))


def probe_cpu(vault):
    r = subprocess.run([str(ENGINE), "bench", str(vault), "--tokens", "24", "--prompt", "8", "--resident-mb", "6000", "--log", os.devnull], capture_output=True, text=True, timeout=600)
    t = re.search(r"S tok_s=([\d.]+).*?resident_MB=(\d+)", r.stdout)
    if not t:
        return {}
    return dict(cpu_tok_s=float(t[1]), cpu_decode_GBps=float(t[1]) * float(t[2]) / 1e3, cpu_vault=str(vault))


def probe_gpu_decode(vault):
    exe = hp.ENGINE_DIR / "gpu" / "mv_bench.exe"
    if not exe.exists():
        return {}
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(hp.ENGINE_DIR / "gpu")); sys.path.insert(0, str(hp.TOOLS))
    import numpy as np, gpu_mv_test as g
    V = json.load(open(Path(vault) / "vault.json")); big = sorted([t for t in V["tensors"] if t["store"] == "huff4" and ".layers.0." in t["name"]], key=lambda t: -t["blob_size"])[:3]
    rates = []
    for t in big:
        blob, _ = g.extract(Path(vault), t["name"]); x = np.random.default_rng(0).standard_normal(t["cols"]).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            blob.tofile(td + "/b"); x.tofile(td + "/x")
            out = subprocess.run([str(exe), td + "/b", td + "/x", td + "/y", "300", "2", "64"], capture_output=True, text=True).stdout
        m = re.search(r"effective=(\d+) GB/s", out)
        if m: rates.append(float(m[1]))
    return dict(gpu_decode_GBps=sum(rates) / len(rates), gpu_vault=str(vault)) if rates else {}


def probe_gpu_engine(prof):
    exe = GPU_ENGINE
    if not exe.exists() or "gpu_decode_GBps" not in prof or "vram_free_mb" not in prof:
        return {}
    vs = [v for v in find_vaults() if v[1] <= 0.5 * prof["vram_free_mb"] * 1048576]
    if not vs:
        return {}
    v = vs[-1][0]; M = model_from_vault(v)
    out = subprocess.run([str(exe), "bench", str(v), "--tokens", "96", "--ctx", "256"], capture_output=True, text=True, timeout=300).stdout
    m = re.search(r"= ([\d.]+) tok/s", out)
    if not m:
        return {}
    P = sum(M["layer_bytes"]) + M["lm_bytes"]; T = 1 / float(m[1])
    return dict(gpu_bench_tok_s=float(m[1]), gpu_bench_vault=str(v), gpu_layer_overhead_s=max((T - P / (prof["gpu_decode_GBps"] * GB)) / M["nl"], 0.0))


def probe(force=False):
    prof = json.load(open(PROFILE)) if PROFILE.exists() and not force else {}
    vs = find_vaults()
    if not vs:
        raise SystemExit("no packed vault found next to planner.py")
    if "nvme_MBps" not in prof:
        print(f"probing NVMe read speed on {vs[-1][0].name} ...", flush=True); prof.update(probe_nvme(vs[-1][0]))
    if "cpu_curve" not in prof:
        # CPU decode rate rises with matrix size (0.6B ~9 GB/s, 1B ~13), so measure every vault that stays fully RAM-resident
        budget = 0.8 * (ram_mb()[1] * 1048576 - RAM_RESERVE)
        fit = [v for v in vs if v[1] <= budget] or [vs[0]]; curve = []
        for v, _ in fit:
            print(f"probing CPU decode rate on {v.name} ...", flush=True); r = probe_cpu(v)
            if r: curve.append([avg_layer_bytes(v), r["cpu_decode_GBps"]]); prof.update(r)
        prof["cpu_curve"] = curve
    g = run_gpu_probe()
    if g and "h2d_pinned_GBps" not in prof:
        prof.update({k: g[k] for k in ("gpu", "h2d_pinned_GBps", "d2d_GBps")})
    if g and "gpu_decode_GBps" not in prof:
        print("probing GPU decode rate ...", flush=True); prof.update(probe_gpu_decode(vs[-1][0] if len(vs) < 2 else vs[1][0]))
    if g: prof.setdefault("vram_free_mb", g["vram_free_mb"])
    if GPU_ENGINE.exists() and g and "gpu_ovh_curve" not in prof and "gpu_decode_GBps" in prof:
        curve = []
        for v, sz in vs:
            if sz > 0.5 * g["vram_free_mb"] * 1048576: continue
            print(f"probing GPU engine on {v.name} ...", flush=True); M = model_from_vault(v)
            out = subprocess.run([str(GPU_ENGINE), "bench", str(v), "--tokens", "96", "--ctx", "256"], capture_output=True, text=True, timeout=300).stdout
            m = re.search(r"= ([\d.]+) tok/s", out)
            if m:
                P = sum(M["layer_bytes"]) + M["lm_bytes"]; ovh = max((1 / float(m[1]) - P / (prof["gpu_decode_GBps"] * GB)) / M["nl"], 0.0); curve.append([avg_layer_bytes(v), ovh])
        prof["gpu_ovh_curve"] = curve
    if GPU_ENGINE.exists() and g and "split_ipc_s" not in prof and prof.get("gpu_ovh_curve") and prof.get("cpu_curve"):
        budget = 0.8 * (ram_mb()[1] * 1048576 - RAM_RESERVE); fit = [v for v in vs if v[1] <= budget and v[1] <= 0.5 * g["vram_free_mb"] * 1048576]
        if fit:
            v = fit[-1][0]; M = model_from_vault(v); n = max(1, M["nl"] // 2); H0 = dict(prof, ram_avail_mb=ram_mb()[1], vram_free_mb=g["vram_free_mb"]); print(f"probing split pipe overhead on {v.name} ...", flush=True)
            r = subprocess.run([str(ENGINE), "bench", str(v), "--tokens", "48", "--prompt", "8", "--gpu-layers", str(n), "--resident-mb", "6000", "--log", os.devnull], capture_output=True, text=True, timeout=600)
            m = re.search(r"S tok_s=([\d.]+)", r.stdout)
            if m:
                H0["split_ipc_s"] = 0.0; pred = split_T(M, H0, n, 2048)["T"]; prof["split_ipc_s"] = max(1 / float(m[1]) - pred, 0.0)
    prof.pop("vram_free_mb", None)
    json.dump(prof, open(PROFILE, "w"), indent=1)
    live = dict(prof); tp, ap = ram_mb(); live.update(ram_total_mb=tp, ram_avail_mb=ap)
    if g: live.update(vram_total_mb=g["vram_total_mb"], vram_free_mb=g["vram_free_mb"])
    return live


def drive_MBps(vault, force=False):
    """read speed of the drive that holds this vault (an external USB HDD is ~30x slower than the internal NVMe); cached per drive letter"""
    prof = json.load(open(PROFILE)) if PROFILE.exists() else {}; disks = prof.setdefault("disk_MBps", {})
    drv = str(Path(vault).resolve().drive).upper() or "?"
    if drv in disks and not force:
        return disks[drv]
    f = Path(vault) / "vault.bin"; size = f.stat().st_size; chunk = 64 << 20; n = max(1, min(8, size // chunk)); got = 0
    print(f"probing read speed of drive {drv} on {f.name} ...", flush=True); t0 = time.time()
    with open(f, "rb", buffering=0) as fh:
        for i in range(n):
            fh.seek((size - chunk) * i // max(n - 1, 1) // 4096 * 4096); got += len(fh.read(chunk))
    disks[drv] = got / 1e6 / (time.time() - t0); json.dump(prof, open(PROFILE, "w"), indent=1)
    return disks[drv]


# ------------------------------------------------------------------ models
def idx_kv(vault):
    kv = {}
    for l in open(Path(vault) / "vault.idx"):
        t = l.split()
        if len(t) == 2: kv[t[0]] = t[1]
    return kv


def model_from_vault(vault):
    V = json.load(open(Path(vault) / "vault.json")); c = V["config"].get("text_config", V["config"]); kv = idx_kv(vault)
    if kv.get("moe") == "1":
        # mixture of experts (pack_universal_v3 Qwen3-MoE, WVAULT 6, or pack_gptoss, WVAULT 7): per-token cost is attention + routed experts
        S = V["streams"]; nl = int(kv["layers"]); e0 = int(kv["expert_stream0"]); ne = int(kv["n_experts"]); ex = S[e0:]
        return dict(name=Path(c.get("_name_or_path", "") or Path(vault).name).name or Path(vault).name, nl=nl, layer_bytes=[S[i]["padded"] for i in range(nl)],
                    lm_bytes=S[nl + 1]["padded"] if kv.get("tie", "0") == "0" else S[nl]["padded"], nkv=int(kv["kv_heads"]), hd=int(kv["head_dim"]),
                    raw=sum(s["raw"] for s in S), total=sum(s["padded"] for s in S), codec=V.get("codec", kv.get("codec", "huffman4")), supported=True, vault=str(vault),
                    moe=True, n_experts=ne, top_k=int(kv["top_k"]), expert_count=len(ex), expert_bytes=max(s["padded"] for s in ex), gptoss=kv.get("gptoss") == "1")
    sc = V["scalars"]; nl = sc["layers"]; S = V["streams"]
    lay = [S[i]["padded"] for i in range(nl)]; lm = S[nl]["padded"] if sc["tie"] else S[nl + 1]["padded"]
    return dict(name=Path(c.get("_name_or_path", "") or Path(vault).name).name or Path(vault).name, nl=nl, layer_bytes=lay, lm_bytes=lm, nkv=sc["kv_heads"], hd=sc["head_dim"],
                raw=sum(s["raw"] for s in S), total=sum(s["padded"] for s in S), codec=V.get("codec", "huffman4-v3"), supported=bool(V.get("engine_u", {}).get("supported") or V.get("engine_compatible_v3")), vault=str(vault))


def model_whatif(params_b, layers, hidden=None, vocab=152000, kv_dim=None, ratio=RATIO, tie=True):
    layer_params = params_b * 1e9 - vocab * (hidden or 0) * (1 if tie else 2)
    hidden = hidden or int(math.sqrt(params_b * 1e9 / (11 * layers))); layer_params = params_b * 1e9 - vocab * hidden * (1 if tie else 2)
    lb = layer_params / layers * 2 * ratio; lm = vocab * hidden * 2 * ratio; kv = kv_dim or hidden // 4
    return dict(name=f"{params_b:g}B (what-if, ratio {ratio})", nl=layers, layer_bytes=[lb] * layers, lm_bytes=lm, nkv=1, hd=kv, raw=params_b * 2e9, total=params_b * 2e9 * ratio, supported=True, vault=None)


# ------------------------------------------------------------------ planner
RANS_CPU = 0.42    # rans4 / huffman4 CPU tok/s, all weights in RAM: Qwen3-0.6B 5.23/11.11, Llama-1B 3.00/8.32 (measured 2026-11-21)
RANS_GPU = 0.041   # rans4 / huffman4 GPU tok/s, Qwen3-0.6B GPU-resident: 2.53/61.4 (measured 2026-11-21)


def is_rans(M):
    return str(M.get("codec", "")).startswith("rans4")


SPLIT_IPC = 0.5e-3      # seconds per token for the pipe round trip; re-measured by probe


def split_T(M, H, ngpu, ctx, vfree=None):
    """GPU runs layers [0, ngpu) from VRAM; CPU runs the rest + lm_head from RAM (then NVMe). Returns the cost terms."""
    nl = M["nl"]; avgL = sum(M["layer_bytes"]) / nl; Rg = H["gpu_decode_GBps"] * GB * (RANS_GPU if is_rans(M) else 1.0); Rc = interp(H.get("cpu_curve"), avgL, H["cpu_decode_GBps"]) * GB * (RANS_CPU if is_rans(M) else 1.0)
    Bn = H["nvme_MBps"] * MB; ov = interp(H.get("gpu_ovh_curve"), avgL, GPU_LAYER_OVERHEAD); kvc = 2 * ctx * M["nkv"] * M["hd"] * 4
    V = sum(M["layer_bytes"][:ngpu]); rleft = H["ram_avail_mb"] * 1048576 - RAM_RESERVE
    R = Nn = 0.0; nR = nN = 0
    for u, b in [("lm", M["lm_bytes"])] + [(i, M["layer_bytes"][i]) for i in range(ngpu, nl)]:
        kv = 0 if u == "lm" else kvc
        if rleft >= b + kv: rleft -= b + kv; R += b; nR += 1
        else: Nn += b; nN += 1
    terms = dict(gpu_resident_part=V / Rg + ngpu * ov, cpu_part=(R + Nn) / Rc, nvme=Nn / Bn if Nn else 0.0)
    return dict(T=terms["gpu_resident_part"] + max(terms["cpu_part"], terms["nvme"]) + H.get("split_ipc_s", SPLIT_IPC), V=V, R=R, N=Nn, nV=ngpu, nR=nR, nN=nN, terms=terms)


def max_gpu_layers(M, H, ctx):
    kvg = 2 * ctx * M["nkv"] * M["hd"] * 4; left = H["vram_free_mb"] * 1048576 - WORKSPACE; n = 0
    for b in M["layer_bytes"]:
        if left >= b + kvg: left -= b + kvg; n += 1
        else: break
    return n


def plan(M, H, ctx=2048, gpu_ready=None):
    gpu_ready = GPU_ENGINE.exists() if gpu_ready is None else gpu_ready
    nl = M["nl"]; avgL = sum(M["layer_bytes"]) / nl; ov = interp(H.get("gpu_ovh_curve"), avgL, GPU_LAYER_OVERHEAD); units = [("lm", M["lm_bytes"])] + [(i, b) for i, b in enumerate(M["layer_bytes"])]   # lm_head first (as the C engine pins it), then layers
    P = sum(b for _, b in units); kvg = 2 * ctx * M["nkv"] * M["hd"] * 4; kvc = 2 * ctx * M["nkv"] * M["hd"] * 4
    Rg = H.get("gpu_decode_GBps", 0) * GB * (RANS_GPU if is_rans(M) else 1.0); Rc = interp(H.get("cpu_curve"), avgL, H["cpu_decode_GBps"]) * GB * (RANS_CPU if is_rans(M) else 1.0); Bn = H["nvme_MBps"] * MB; Bp = H.get("h2d_pinned_GBps", 0) * GB
    vfree = H.get("vram_free_mb", 0) * 1048576 - WORKSPACE if Rg and Bp else 0
    rbudget = H["ram_avail_mb"] * 1048576 - RAM_RESERVE

    def fill(vram, ram, cpu_kv):
        V = R = N = 0.0; nV = nR = nN = 0; vleft, rleft = vram, ram
        for u, b in units:
            kvl = 0 if u == "lm" else kvg
            if vleft >= b + kvl: vleft -= b + kvl; V += b; nV += 1
            else:
                kvr = 0 if (u == "lm" or not cpu_kv) else kvc
                if rleft >= b + kvr: rleft -= b + kvr; R += b; nR += 1
                else: N += b; nN += 1
        return V, R, N, nV, nR, nN
    opts = []
    def add(code, name, V, R, N, nV, nR, nN, terms, extra=0.0):
        T = max(terms.values()) if code in ("H", "C") else None
        if code == "S": T = terms["gpu_resident_part"] + max(terms["cpu_part"], terms["nvme"])
        if code == "G": T = terms["gpu_decode"]
        T += extra; lim = max(terms, key=terms.get)
        opts.append(dict(code=code, name=name, V=V, R=R, N=N, nV=nV, nR=nR, nN=nN, T=T, tok_s=1 / T, limiter=lim, terms=terms, gpu=code in "GHS"))
    if vfree > 0:
        V, R, N, nV, nR, nN = fill(vfree, 0, False)
        if N == 0 and R == 0:
            add("G", "GPU-RESIDENT", V, R, N, nV, nR, nN, dict(gpu_decode=P / Rg), nl * ov)
        else:       # does not fit VRAM: compare streaming vs split
            V, R, N, nV, nR, nN = fill(vfree, rbudget, False)
            add("H", "GPU + PCIe STREAMING", V, R, N, nV, nR, nN, dict(gpu_decode=P / Rg, pcie=(R + N) / Bp, nvme=N / Bn if N else 0), nl * ov)
            ng = max_gpu_layers(M, dict(H, vram_free_mb=vfree / 1048576 + WORKSPACE / 1048576), ctx)
            if ng > 0:
                sp = split_T(M, H, ng, ctx)
                add("S", "GPU + CPU SPLIT", sp["V"], sp["R"], sp["N"], sp["nV"], sp["nR"], sp["nN"], sp["terms"], H.get("split_ipc_s", SPLIT_IPC)); opts[-1]["gpu_layers"] = ng
    V, R, N, nV, nR, nN = fill(0, rbudget, True)
    add("C", "CPU (RAM-resident)" if N == 0 else "CPU (NVMe-streaming)", V, R, N, nV, nR, nN, dict(cpu_decode=P / Rc, nvme=N / Bn if N else 0))
    gpu_codec_ok = True   # both engines read huffman4 and rans4
    for o in opts: o["available"] = (not o["gpu"]) or (gpu_ready and gpu_codec_ok and o["code"] in ("G", "S"))   # built: GPU-RESIDENT and GPU+CPU SPLIT; PCIe streaming is not
    best_any = min(opts, key=lambda o: o["T"]); best_now = min([o for o in opts if o["available"]], key=lambda o: o["T"])
    # what would unlock the next level?
    hint = None
    need = P + (nl * kvg) + WORKSPACE - (H.get("vram_free_mb", 0) * 1048576)
    if Rg and need > 0 and best_any["code"] != "G":
        hint = f"+{need / GB:.1f} GB free VRAM would make it GPU-RESIDENT (~{1 / (P / Rg + nl * ov):.1f} tok/s predicted)"
    return dict(options=opts, best=best_any, best_available=best_now, P=P, hint=hint, gpu_ready=gpu_ready)


def fmt(T):
    return f"{1 / T:.1f} tok/s" if T <= 1 else f"{1 / T:.2f} tok/s ({T:.1f} s/token)"


def banner(M, H, R, ctx):
    b, a = R["best"], R["best_available"]; M = dict(M, _avgL=sum(M["layer_bytes"]) / M["nl"])
    lines = []
    lines.append(f"[hw]    GPU {H.get('gpu', 'none')}: {H.get('vram_free_mb', 0) / 1024:.1f} GB free of {H.get('vram_total_mb', 0) / 1024:.1f} | RAM {H['ram_avail_mb'] / 1024:.1f} GB free of {H['ram_total_mb'] / 1024:.1f}"
                 f" | vault drive {H['nvme_MBps'] / 1000:.2f} GB/s | PCIe {H.get('h2d_pinned_GBps', 0):.1f} GB/s | decode: GPU {H.get('gpu_decode_GBps', 0):.0f} GB/s, CPU {interp(H.get('cpu_curve'), M['_avgL'], H['cpu_decode_GBps']):.1f} GB/s")
    lines.append(f"[model] {M['name']} [{M.get('codec', 'huffman4-v3')}]: {M['nl']} layers, {R['P'] / GB:.2f} GB compressed per token (bf16 would be {M['raw'] / GB:.2f} GB), ctx {ctx}")
    for o in sorted(R["options"], key=lambda o: o["T"]):
        tag = "" if o["available"] else ("  [not built yet]" if o["gpu"] else "")
        lines.append(f"        {o['name']:<24s} VRAM {o['V'] / GB:5.2f} GB ({o['nV']:>2d}) | RAM {o['R'] / GB:5.2f} GB ({o['nR']:>2d}) | NVMe {o['N'] / GB:5.2f} GB ({o['nN']:>2d}) -> {fmt(o['T'])}, limit: {o['limiter']}{tag}")
    lines.append(f"[plan]  MODE: {b['name']}  (predicted {fmt(b['T'])}, limited by {b['limiter']})")
    if b is not a:
        lines.append(f"[plan]  RUNNING NOW: {a['name']} (predicted {fmt(a['T'])}) - the {b['name']} engine is not built yet")
    if R["hint"]: lines.append(f"[hint]  {R['hint']}")
    rg_fit = (H.get("vram_free_mb", 0) * 1048576 - WORKSPACE - 0.4 * GB) / (2 * RATIO) / 1e9; rc_fit = (H["ram_avail_mb"] * 1048576 - RAM_RESERVE) / (2 * RATIO) / 1e9
    lines.append(f"[fit]   on this machine right now: GPU-resident up to ~{max(rg_fit, 0):.1f}B params, RAM-resident up to ~{max(rc_fit, 0):.1f}B params (bf16 packed at {RATIO})")
    if not M["supported"]: lines.append("[warn]  this vault's architecture is not supported by the engine")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("probe"); a.add_argument("--force", action="store_true")
    a = sp.add_parser("plan"); a.add_argument("vault"); a.add_argument("--ctx", type=int, default=2048)
    a = sp.add_parser("whatif"); a.add_argument("--params-b", type=float, required=True); a.add_argument("--layers", type=int, required=True); a.add_argument("--hidden", type=int)
    a.add_argument("--vocab", type=int, default=152000); a.add_argument("--kv-dim", type=int); a.add_argument("--ratio", type=float, default=RATIO); a.add_argument("--ctx", type=int, default=2048)
    a = sp.add_parser("bench"); a.add_argument("vault"); a.add_argument("--ctx", type=int, default=2048)
    a = sp.add_parser("serve"); a.add_argument("vault"); a.add_argument("rest", nargs=argparse.REMAINDER)
    x = ap.parse_args()
    if x.cmd == "probe":
        H = probe(x.force); print(json.dumps(H, indent=1)); return
    H = probe()
    if x.cmd == "whatif":
        M = model_whatif(x.params_b, x.layers, x.hidden, x.vocab, x.kv_dim, x.ratio); R = plan(M, H, x.ctx, gpu_ready=True); print(banner(M, H, R, x.ctx)); return
    M = model_from_vault(x.vault); ctx = getattr(x, "ctx", 2048); H = dict(H, nvme_MBps=drive_MBps(x.vault)); R = plan(M, H, ctx); print(banner(M, H, R, ctx))
    cpu = next(o for o in R["options"] if o["code"] == "C")
    if x.cmd == "bench":
        mb = int(cpu["R"] / MB) + 1
        print(f"\nrunning the CPU engine with --resident-mb {mb} (planned RAM-resident bytes) ...")
        r = subprocess.run([str(ENGINE), "bench", M["vault"], "--tokens", "24", "--prompt", "8", "--resident-mb", str(mb), "--log", os.devnull], capture_output=True, text=True)
        m = re.search(r"S tok_s=([\d.]+).*?resident_MB=(\d+)", r.stdout)
        print(f"CPU mode: predicted {cpu['tok_s']:.2f} tok/s, measured {float(m[1]):.2f} tok/s (resident {m[2]} MB) -> error {100 * (cpu['tok_s'] / float(m[1]) - 1):+.0f}%" if m else r.stdout[-400:] + r.stderr[-400:])
        sp = next((o for o in R["options"] if o["code"] == "S"), None)
        if sp and GPU_ENGINE.exists():
            r = subprocess.run([str(ENGINE), "bench", M["vault"], "--tokens", "24", "--prompt", "8", "--gpu-layers", str(sp["gpu_layers"]), "--resident-mb", str(int(sp["R"] / MB) + 1), "--log", os.devnull], capture_output=True, text=True)
            m = re.search(r"S tok_s=([\d.]+)", r.stdout)
            print(f"GPU+CPU SPLIT mode ({sp['gpu_layers']} layers on GPU): predicted {sp['tok_s']:.2f} tok/s, measured {float(m[1]):.2f} tok/s -> error {100 * (sp['tok_s'] / float(m[1]) - 1):+.0f}%" if m else r.stdout[-300:] + r.stderr[-300:])
        g = next((o for o in R["options"] if o["code"] == "G"), None)
        if g and GPU_ENGINE.exists():
            r = subprocess.run([str(GPU_ENGINE), "bench", M["vault"], "--tokens", "96", "--ctx", "256"], capture_output=True, text=True)
            m = re.search(r"= ([\d.]+) tok/s", r.stdout)
            print(f"GPU-RESIDENT mode: predicted {g['tok_s']:.2f} tok/s, measured {float(m[1]):.2f} tok/s -> error {100 * (g['tok_s'] / float(m[1]) - 1):+.0f}%" if m else r.stdout[-300:] + r.stderr[-300:])
    elif x.cmd == "serve":
        a = R["best_available"]; mb = int(a["R"] / MB) + 1 if a["code"] == "C" else int(cpu["R"] / MB) + 1
        if a["code"] == "S":
            print(f"\n[serve] launching {ENGINE.name} --gpu-layers {a['gpu_layers']} --resident-mb {int(a['R'] / MB) + 1}  (first {a['gpu_layers']} layers on the GPU, rest on the CPU)\n", flush=True)
            sys.exit(subprocess.call([str(ENGINE), "serve", M["vault"], "--gpu-layers", str(a["gpu_layers"]), "--resident-mb", str(int(a["R"] / MB) + 1)] + [r for r in x.rest if r != "--"]))
        if a["code"] == "G":
            print(f"\n[serve] launching {GPU_ENGINE.name} (all weights compressed in VRAM)\n", flush=True)
            sys.exit(subprocess.call([str(GPU_ENGINE), "serve", M["vault"]] + [r for r in x.rest if r != "--"]))
        print(f"\n[serve] launching {ENGINE.name} --resident-mb {mb}\n", flush=True)
        sys.exit(subprocess.call([str(ENGINE), "serve", M["vault"], "--resident-mb", str(mb)] + [r for r in x.rest if r != "--"]))   # execv detaches on Windows


if __name__ == "__main__":
    main()
