"""harness_paths.py - every path the harness uses, resolved from the folder it runs in (so the whole folder can be moved or copied).

  cube-harness/                 ROOT (holds cube-harness.bat and README.md)
    app/                        this file, the window, the core, the engine launcher, planner, model hub, setup, launch check
    engine/                     nvme_engine_u.exe, gpu_experts.dll, gpu/gpu_engine.exe and their sources
    tools/                      packers and vault tools (research/, patches/, bat/ below it)
    tests/                      test scripts and their small test data
    data/                       everything the harness writes:
      vaults/                   the built-in model folder (more folders, on or off the SSD, are added in the Model folders tab)
      conversations/ skills/ plugins/ artifacts/ workspace/ hub_cache/ packages/ logs/
      harness_config.json       settings; paths inside the harness folder are stored relative to ROOT
      hw_profile.json           measured hardware (re-measured when the harness finds itself on another machine)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
ENGINE_DIR = ROOT / "engine"
TOOLS = ROOT / "tools"
TESTS = ROOT / "tests"
DATA = ROOT / "data"
CONFIG = DATA / "harness_config.json"
PROFILE = DATA / "hw_profile.json"
VAULTS = DATA / "vaults"
CONVERSATIONS = DATA / "conversations"
SKILLS = DATA / "skills"
PLUGINS = DATA / "plugins"
ARTIFACTS = DATA / "artifacts"
WORKSPACE = DATA / "workspace"
HUB_CACHE = DATA / "hub_cache"
LOGS = DATA / "logs"
PACKAGES = DATA / "packages"
ENGINE = ENGINE_DIR / "nvme_engine_u.exe"
GPU_ENGINE = ENGINE_DIR / "gpu" / "gpu_engine.exe"

DEFAULTS = {
    "vault": "", "mode": "auto", "gpu_layers": 0, "ctx": 2048, "max_new": 512, "kv16": False, "max_file_growth": 0,
    "temp": 0.7, "top_p": 0.8, "top_k": 20, "greedy": False, "seed": None,
    "draft": "auto", "k": 8, "think": False, "system": "",
    "model_dirs": [], "enabled_plugins": ["calculator", "clock", "artifact"], "max_tool_rounds": 4,
}


def rel(p):
    """store a path relative to ROOT when it lives inside the harness folder"""
    p = Path(p).resolve()
    try: return str(p.relative_to(ROOT))
    except ValueError: return str(p)


def absolute(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG.exists():
        try: cfg.update(json.load(open(CONFIG, encoding="utf-8")))
        except Exception: pass
    # older configs: extra_vault_dirs / hub_vault_dir become entries of model_dirs
    md = list(cfg.get("model_dirs") or [])
    for x in cfg.pop("extra_vault_dirs", []) or []:
        if not any(e["path"] == x for e in md): md.append({"path": x, "default": False})
    hv = cfg.pop("hub_vault_dir", None)
    if hv and not any(e["path"] == hv for e in md): md.append({"path": hv, "default": True})
    cfg["model_dirs"] = md
    return cfg


def save_config(cfg):
    DATA.mkdir(exist_ok=True); json.dump(cfg, open(CONFIG, "w", encoding="utf-8"), indent=1)


def ensure_dirs():
    for d in (DATA, VAULTS, CONVERSATIONS, SKILLS, PLUGINS, ARTIFACTS, WORKSPACE, HUB_CACHE, LOGS, PACKAGES): d.mkdir(parents=True, exist_ok=True)


def is_vault(d):
    d = Path(d); return d.is_dir() and (d / "vault.idx").exists() and (d / "vault.bin").exists()


def model_dirs(cfg=None):
    """[{path, default, builtin}] - the built-in data/vaults first, then the folders added in the Model folders tab"""
    cfg = cfg or load_config(); out = [{"path": VAULTS, "default": False, "builtin": True}]
    for e in cfg.get("model_dirs", []):
        p = absolute(e["path"])
        if p != VAULTS and not any(o["path"] == p for o in out): out.append({"path": p, "default": bool(e.get("default")), "builtin": False})
        elif p == VAULTS: out[0]["default"] = bool(e.get("default"))
    if not any(o["default"] for o in out): out[0]["default"] = True
    return out


def default_model_dir(cfg=None):
    """where newly installed (compressed) models go"""
    return next(o["path"] for o in model_dirs(cfg) if o["default"])


def add_model_dir(path, default=False, cfg=None):
    cfg = cfg or load_config(); p = rel(path)
    md = [e for e in cfg.get("model_dirs", []) if absolute(e["path"]) != absolute(p)]
    if default:
        for e in md: e["default"] = False
    md.append({"path": p, "default": bool(default)}); cfg["model_dirs"] = md; save_config(cfg); return cfg


def vault_dirs(cfg=None):
    """folders searched for vaults: every model folder (existing ones), plus ROOT for very old vault_* layouts"""
    return [o["path"] for o in model_dirs(cfg) if o["path"].exists()] + [ROOT]


def find_vaults(cfg=None):
    """{name: path} of every vault found; earlier model folders win on a name clash"""
    found = {}
    for base in vault_dirs(cfg):
        for d in sorted(base.iterdir()) if base.is_dir() else []:
            if is_vault(d) and d.name not in found: found[d.name] = d
        if is_vault(base) and base.name not in found: found[base.name] = base
    df = DATA / "drafts.txt"
    if df.exists():
        for l in open(df, encoding="utf-8"):
            l = l.strip()
            if l and not l.startswith("#") and is_vault(absolute(l)): found.setdefault(absolute(l).name, absolute(l))
    return found


def resolve_vault(name_or_path, cfg=None):
    """a vault by name (searched in every model folder) or by path (absolute or relative to ROOT)"""
    p = absolute(name_or_path)
    if is_vault(p): return p
    v = find_vaults(cfg).get(Path(str(name_or_path)).name)
    if v: return v
    raise FileNotFoundError(f"vault '{name_or_path}' not found in {', '.join(str(d) for d in vault_dirs(cfg))}")
