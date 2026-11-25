"""fix_paths_after_reorg.py - point every script at the new layout (see app/harness_paths.py). Idempotent where it matters."""
import ast, re
from pathlib import Path
R = Path(__file__).resolve().parents[2]
done = []
def patch(rel, pairs, must=True):
    p = R / rel; s = p.read_text(encoding="utf-8"); o = s
    for a, b in pairs:
        if a not in s:
            if must and b not in s: raise SystemExit(f"{rel}: pattern not found: {a[:70]!r}")
            continue
        s = s.replace(a, b)
    if p.suffix == ".py": ast.parse(s)
    if s != o: p.write_text(s, encoding="utf-8"); done.append(rel)

# ---------------------------------------------------------------- app
patch("app/planner.py", [
    ('HERE = Path(__file__).resolve().parent\nPROFILE = HERE / "hw_profile.json"\nENGINE = HERE / "nvme_engine_u.exe"\nGPU_ENGINE = HERE / "gpu" / "gpu_engine.exe"',
     'HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))\nimport harness_paths as hp\nPROFILE = hp.PROFILE\nENGINE = hp.ENGINE\nGPU_ENGINE = hp.GPU_ENGINE'),
    ('HERE / "gpu" / "bw_probe.exe"', 'hp.ENGINE_DIR / "gpu" / "bw_probe.exe"'),
    ('HERE / "gpu" / "mv_bench.exe"', 'hp.ENGINE_DIR / "gpu" / "mv_bench.exe"'),
    ('sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "gpu"))', 'sys.path.insert(0, str(HERE)); sys.path.insert(0, str(hp.ENGINE_DIR / "gpu")); sys.path.insert(0, str(hp.TOOLS))'),
])
patch("app/spec.py", [
    ('HERE = Path(__file__).resolve().parent\n', 'HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))\nimport harness_paths as hp\n'),
    ('str(HERE / "nvme_engine_u.exe")', 'str(hp.ENGINE)'),
    ('str(HERE / "gpu" / "gpu_engine.exe")', 'str(hp.GPU_ENGINE)'),
    ('str(HERE / "spec_target.err")', 'str(hp.LOGS / "spec_target.err")'),
    ('str(HERE / "spec_draft.err")', 'str(hp.LOGS / "spec_draft.err")'),
])
patch("app/harness_app.py", [
    ('str(hp.ROOT / "planner.py")', 'str(hp.APP / "planner.py")'),
    ('run setup.bat', 'run cube-harness.bat'),
    ('Start it with "Cube Harness.bat"', 'Start it with cube-harness.bat'),
])
patch("app/model_hub.py", [
    ('CACHE = hp.ROOT / "hub_cache"', 'CACHE = hp.HUB_CACHE'),
    ('HERE / "pack_gptoss.py"', 'hp.TOOLS / "pack_gptoss.py"'),
    ('HERE / "pack_universal_v3.py"', 'hp.TOOLS / "pack_universal_v3.py"'),
    ('vd = Path(cfg.get("hub_vault_dir") or hp.VAULTS); dd = Path(cfg.get("hub_download_dir") or r"D:\\models")',
     'vd = hp.default_model_dir(cfg); dd = Path(cfg.get("hub_download_dir") or (r"D:\\models" if Path("D:/").exists() else hp.DATA / "downloads"))'),
])
patch("app/setup_harness.py", [
    ('ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))\nimport harness_paths as hp',
     'sys.path.insert(0, str(Path(__file__).resolve().parent))\nimport harness_paths as hp\nROOT = hp.ROOT'),
    ('subprocess.run(["cmd", "/c", str(ROOT / "build.bat")], cwd=str(ROOT))', 'subprocess.run(["cmd", "/c", str(hp.ENGINE_DIR / "build.bat")], cwd=str(hp.ENGINE_DIR))'),
    ('cwd=str(ROOT / "gpu"))', 'cwd=str(hp.ENGINE_DIR / "gpu"))'),
    ('step(2, "Folders"); hp.ensure_dirs(); (ROOT / "workspace").mkdir(exist_ok=True)', 'step(2, "Folders"); hp.ensure_dirs()'),
    ('for d in (hp.VAULTS, hp.CONVERSATIONS, hp.SKILLS, hp.PLUGINS, hp.LOGS, hp.PACKAGES, ROOT / "workspace"):', 'for d in (hp.VAULTS, hp.CONVERSATIONS, hp.SKILLS, hp.PLUGINS, hp.ARTIFACTS, hp.LOGS, hp.PACKAGES, hp.WORKSPACE):'),
    ('df = ROOT / "drafts.txt"; keep = []', 'df = hp.DATA / "drafts.txt"; keep = []'),
    ('str(ROOT / "vault_defrag.py")', 'str(hp.TOOLS / "vault_defrag.py")'),
    ('    cfg["extra_vault_dirs"] = [hp.rel(d) for d in cfg.get("extra_vault_dirs", []) if hp.absolute(d).exists()]',
     '    cfg["model_dirs"] = [e for e in cfg.get("model_dirs", []) if hp.absolute(e["path"]).exists()]'),
    ('''    bat = ROOT / "Cube Harness.bat"
    bat.write_text('@echo off\\r\\ncd /d "%~dp0"\\r\\nwhere pythonw >nul 2>nul && (start "" pythonw "%~dp0harness_app.py") || (start "" python "%~dp0harness_app.py")\\r\\n', encoding="ascii")
    say(f"    {bat.name}")''',
     '''    bat = ROOT / "cube-harness.bat"
    say(f"    {bat.name} {'found' if bat.exists() else 'MISSING - copy it back from the harness download'}")'''),
    ('str(ROOT / "harness_core.py")', 'str(hp.APP / "harness_core.py")'),
    ('Start the harness with "Cube Harness.bat".', 'Start the harness with cube-harness.bat.'),
    ('first-run setup for the Cube Harness (run through setup.bat)', 'first-run setup for the Cube Harness (run by cube-harness.bat when needed)'),
])

# ---------------------------------------------------------------- tools, research, tests: find ROOT from the folder layout
BOOT = ('import sys as _sys; from pathlib import Path as _P; '
        '_ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); '
        '_sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]\n')
def boot(rel):
    p = R / rel; s = p.read_text(encoding="utf-8")
    if "_ROOT = next(p for p in _P(__file__)" in s: return
    t = ast.parse(s); line = 0
    if t.body and isinstance(t.body[0], ast.Expr) and isinstance(getattr(t.body[0], "value", None), ast.Constant) and isinstance(t.body[0].value.value, str): line = t.body[0].end_lineno
    lines = s.splitlines(keepends=True); s2 = "".join(lines[:line]) + BOOT + "".join(lines[line:])
    s2 = (s2.replace('HERE / "nvme_engine_u.exe"', '_ROOT / "engine" / "nvme_engine_u.exe"').replace('HERE / "gpu" / "gpu_engine.exe"', '_ROOT / "engine" / "gpu" / "gpu_engine.exe"')
            .replace('HERE / "spec.py"', '_ROOT / "app" / "spec.py"').replace('HERE / "vaults"', '_ROOT / "data" / "vaults"'))
    ast.parse(s2); p.write_text(s2, encoding="utf-8"); done.append(rel)
for p in sorted((R / "tools").glob("*.py")) + sorted((R / "tools" / "research").glob("*.py")):
    if "Path(__file__)" in p.read_text(encoding="utf-8"): boot(str(p.relative_to(R)))
patch("tools/park_vault.py", [
    ('cfg = hp.load_config(); ex = list(cfg.get("extra_vault_dirs", []))\nif str(root) not in ex and str(root) != str(hp.VAULTS): ex.append(str(root)); cfg["extra_vault_dirs"] = ex; hp.save_config(cfg); print(f"added {root} to extra_vault_dirs")',
     'if root.resolve() not in [d["path"] for d in hp.model_dirs()]: hp.add_model_dir(root); print(f"added {root} to the model folders")'),
], must=False)
for p in sorted((R / "tests").glob("*.py")):
    s = p.read_text(encoding="utf-8"); o = s
    for a, b in [('H / "logs" / "gptoss_ids.txt"', 'H / "tests" / "data" / "gptoss_ids.txt"'), ('H / "logs" / "gptoss_hf_logits.npy"', 'H / "tests" / "data" / "gptoss_hf_logits.npy"'),
                 ('H / "ref_logits.npy"', 'H / "tests" / "data" / "ref_logits.npy"'), ('H / "plugins"', 'H / "data" / "plugins"'), ('H / "logs"', 'H / "data" / "logs"'),
                 ('H / "nvme_engine_u.exe"', 'H / "engine" / "nvme_engine_u.exe"'), ('H / "gpu_experts.dll"', 'H / "engine" / "gpu_experts.dll"'),
                 ('H / "vaults"', 'H / "data" / "vaults"'), ('H / "harness_core.py"', 'H / "app" / "harness_core.py"'),
                 ('sys.path.insert(0, str(H))', 'sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]'),
                 ('sys.path.insert(0, str(Path(__file__).resolve().parent.parent))', 'sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "app")]'),
                 ('os.path.dirname(os.path.dirname(os.path.abspath(__file__))); V = os.path.join(H, "vaults"', 'os.path.dirname(os.path.dirname(os.path.abspath(__file__))); V = os.path.join(H, "data", "vaults"'),
                 ('os.path.join(H, "gpu_experts.dll")', 'os.path.join(H, "engine", "gpu_experts.dll")')]:
        s = s.replace(a, b)
    if 'H = Path(__file__).resolve().parent.parent' in s and 'str(H / "app")' not in s and "import harness" in s: s = s.replace("import harness", 'import sys as _s; _s.path.insert(0, str(H / "app")); import harness', 1)
    ast.parse(s)
    if s != o: p.write_text(s, encoding="utf-8"); done.append(str(p.relative_to(R)))
print(len(done), "files updated:"); print("\n".join("  " + d for d in done))
