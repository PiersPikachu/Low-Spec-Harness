#!/usr/bin/env python3
"""setup_harness.py - first-run setup for the Cube Harness (run by cube-harness.bat when needed).

Makes the harness self-contained in the folder it runs from, so the whole folder can be moved or copied:
  1  Python packages        installs what is missing (numpy, transformers, tokenizers, safetensors, jinja2, pillow)
  2  folders                vaults\\ conversations\\ skills\\ plugins\\ logs\\ packages\\ workspace\\
  3  engines                checks nvme_engine_u.exe and gpu\\gpu_engine.exe; builds them if missing and compilers exist
  4  unpack                 packages\\*.zip that contain a vault are extracted into vaults\\
  5  route vaults           vault_* folders in the harness root move into vaults\\ (same drive: instant rename);
                            draft vaults listed in drafts.txt that live outside the folder are copied in (if under 4 GB)
  6  check vaults           tokenizer present; a fragmented vault.bin is rewritten contiguously (vault_defrag.py)
  7  settings               harness_config.json with relative paths only; default model = largest vault with a chat template
  8  launcher               "Cube Harness.bat" (opens the window, no console) and, if wanted, a desktop shortcut
  9  self-tests             core (plugins, skills, conversations) and engine selftest
  setup_harness.py [--yes]   --yes answers every question with its default
"""
import importlib, json, shutil, subprocess, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness_paths as hp
ROOT = hp.ROOT

YES = "--yes" in sys.argv
PKGS = [("numpy", "numpy"), ("transformers", "transformers"), ("tokenizers", "tokenizers"), ("safetensors", "safetensors"), ("jinja2", "jinja2"), ("PIL", "pillow")]
TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "generation_config.json"]
notes = []


def say(s=""): print(s, flush=True)
def step(n, s): say(f"\n[{n}] {s}")


def ask(q, default=True):
    if YES: return default
    a = input(f"    {q} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not a else a.startswith("y")


def packages():
    step(1, "Python packages")
    if sys.version_info < (3, 10): sys.exit(f"Python 3.10 or newer is needed (this is {sys.version.split()[0]})")
    missing = []
    for mod, pip in PKGS:
        try: importlib.import_module(mod); say(f"    ok       {pip}")
        except Exception: missing.append(pip); say(f"    missing  {pip}")
    try: import tkinter; say("    ok       tkinter")
    except Exception: notes.append("tkinter is missing: reinstall Python with the 'tcl/tk' option to get the window")
    if missing and ask(f"install {', '.join(missing)} with pip?"):
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", *missing], check=False)


def folders():
    step(2, "Folders"); hp.ensure_dirs()
    for d in (hp.VAULTS, hp.CONVERSATIONS, hp.SKILLS, hp.PLUGINS, hp.ARTIFACTS, hp.LOGS, hp.PACKAGES, hp.WORKSPACE): say(f"    {hp.rel(d)}\\")


def engines():
    step(3, "Engines")
    if not hp.ENGINE.exists():
        if shutil.which("gcc") and ask("nvme_engine_u.exe is missing - build it with gcc (build.bat)?"):
            subprocess.run(["cmd", "/c", str(hp.ENGINE_DIR / "build.bat")], cwd=str(hp.ENGINE_DIR))
        else: notes.append("nvme_engine_u.exe is missing and gcc is not on PATH (install MinGW-w64, then rerun setup)")
    if not hp.GPU_ENGINE.exists():
        if shutil.which("nvcc") and ask("gpu\\gpu_engine.exe is missing - build it with nvcc?"):
            subprocess.run(["cmd", "/c", "nvbuild.bat", "gpu_engine.cu"], cwd=str(hp.ENGINE_DIR / "gpu"))
        else: notes.append("gpu\\gpu_engine.exe is missing (needs CUDA nvcc + Visual Studio Build Tools); the CPU engine still works")
    for e in (hp.ENGINE, hp.GPU_ENGINE): say(f"    {'ok     ' if e.exists() else 'MISSING'}  {hp.rel(e)}")


def unpack():
    step(4, "Unpack packages\\*.zip")
    zips = sorted(hp.PACKAGES.glob("*.zip"))
    if not zips: say("    nothing to unpack"); return
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist(); idx = [n for n in names if n.endswith("vault.idx")]
            if not idx: say(f"    skip {z.name}: no vault.idx inside"); continue
            prefix = idx[0][:-len("vault.idx")]; dest = hp.VAULTS / (Path(prefix.rstrip("/")).name or z.stem)
            if dest.exists(): say(f"    skip {z.name}: {hp.rel(dest)} already exists"); continue
            say(f"    {z.name} -> {hp.rel(dest)}"); dest.mkdir(parents=True)
            for n in names:
                if n.startswith(prefix) and not n.endswith("/"):
                    out = dest / n[len(prefix):]; out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(n) as src, open(out, "wb") as dst: shutil.copyfileobj(src, dst, 64 << 20)


def engines_running():
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    return any(n in out for n in ("nvme_engine_u.exe", "gpu_engine.exe"))


def route():
    step(5, "Route vaults into vaults\\")
    legacy = [d for d in sorted(ROOT.glob("vault_*")) if hp.is_vault(d)]
    if legacy and engines_running():
        notes.append("engines were running, so vault_* folders were not moved; close the harness and rerun setup"); legacy = []
    for d in legacy:
        dest = hp.VAULTS / d.name
        if dest.exists(): say(f"    keep {d.name} in place: {hp.rel(dest)} exists"); continue
        if ask(f"move {d.name} into vaults\\?"):
            d.rename(dest); say(f"    moved {d.name} -> {hp.rel(dest)}")
    df = hp.DATA / "drafts.txt"; keep = []
    if df.exists():
        for line in open(df, encoding="utf-8").read().splitlines():
            p = line.strip()
            if not p or p.startswith("#"): keep.append(line); continue
            src = hp.absolute(p)
            inside = str(src).lower().startswith(str(ROOT).lower())
            if not hp.is_vault(src) or inside: keep.append(line); continue
            size = sum(f.stat().st_size for f in src.iterdir() if f.is_file())
            name = "vault_06b_huff" if src.name == "uni" else src.name; dest = hp.VAULTS / name
            if dest.exists(): say(f"    draft {src} already copied as {hp.rel(dest)}"); continue
            if size < 4e9 and ask(f"copy draft vault {src} ({size / 1e9:.2f} GB) into {hp.rel(dest)}?"):
                shutil.copytree(src, dest); say(f"    copied -> {hp.rel(dest)}")
            else: keep.append(line)
        open(df, "w", encoding="utf-8").write("\n".join(keep) + "\n")
    for name, p in hp.find_vaults().items(): say(f"    found {name:28s} {hp.rel(p)}")


def tokenizer_donor(vault, vaults):
    """a vault with the same vocabulary size that ships tokenizer files"""
    import chat
    v = chat.idx_info(vault).get("vocab")
    for p in vaults.values():
        if p != vault and (p / "tokenizer.json").exists() and chat.idx_info(p).get("vocab") == v: return p


def check():
    step(6, "Check vaults")
    import chat
    vaults = hp.find_vaults()
    for name, p in vaults.items():
        inside = str(p).lower().startswith(str(ROOT).lower()); line = f"    {name:28s}"
        if not (p / "tokenizer.json").exists() and not (p / "vocab.json").exists():
            donor = tokenizer_donor(p, vaults)
            if donor and inside:
                for f in TOK_FILES:
                    if (donor / f).exists() and f != "chat_template.jinja": shutil.copy2(donor / f, p / f)
                line += f" tokenizer copied from {donor.name};"
            else: line += " no tokenizer (fine for a draft-only vault);"
        size = (p / "vault.bin").stat().st_size
        if inside and size > 1e9:
            out = subprocess.run(["fsutil", "file", "queryextents", str(p / "vault.bin")], capture_output=True, text=True).stdout
            n = sum(1 for l in out.splitlines() if l.strip().startswith("VCN")); avg = size / 2**20 / max(n, 1)
            line += f" {n} fragments (avg {avg:.0f} MB)"
            if avg < 16 and ask(f"{name} is fragmented ({n} pieces) and will stream slowly - rewrite it contiguously?"):
                subprocess.run([sys.executable, str(hp.TOOLS / "vault_defrag.py"), str(p)])
        say(line)


def settings():
    step(7, "Settings (harness_config.json)")
    import chat
    cfg = hp.load_config(); vaults = hp.find_vaults(cfg)
    cfg["model_dirs"] = [e for e in cfg.get("model_dirs", []) if hp.absolute(e["path"]).exists()]
    if not cfg.get("vault") or cfg["vault"] not in vaults:
        chatty = [(p / "vault.bin").stat().st_size for p in vaults.values()]
        best = None
        for name, p in sorted(vaults.items(), key=lambda kv: -(kv[1] / "vault.bin").stat().st_size):
            tc = p / "tokenizer_config.json"
            if (p / "chat_template.jinja").exists() or (tc.exists() and "chat_template" in tc.read_text(encoding="utf-8", errors="ignore")): best = name; break
        cfg["vault"] = best or (next(iter(vaults)) if vaults else "")
    hp.save_config(cfg); say(f"    default model: {cfg['vault'] or '(none found)'}"); say(f"    written {hp.rel(hp.CONFIG)} (relative paths only)")


def launcher():
    step(8, "Launcher")
    bat = ROOT / "cube-harness.bat"
    say(f"    {bat.name} {'found' if bat.exists() else 'MISSING - copy it back from the harness download'}")
    if ask("create a desktop shortcut?", default=False):
        ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\\Cube Harness.lnk');"
              f"$s.TargetPath='{bat}';$s.WorkingDirectory='{ROOT}';$s.WindowStyle=7;$s.Save()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps]); say("    desktop shortcut created")


def selftests():
    step(9, "Self-tests")
    r = subprocess.run([sys.executable, str(hp.APP / "harness_core.py"), "selftest"], capture_output=True, text=True, cwd=str(ROOT))
    say("    core:   " + (r.stdout.strip().splitlines() or ["(no output)"])[-1])
    if hp.ENGINE.exists():
        r = subprocess.run([str(hp.ENGINE), "selftest"], capture_output=True, text=True, cwd=str(ROOT))
        say("    engine: " + ((r.stdout + r.stderr).strip().splitlines() or ["(no output)"])[-1])


def main():
    say(f"Cube Harness setup in {ROOT}")
    for f in (packages, folders, engines, unpack, route, check, settings, launcher, selftests): f()
    say("\nDone." + ("" if not notes else "\nNotes:\n" + "\n".join(f"  - {n}" for n in notes)))
    say('Start the harness with cube-harness.bat.')


if __name__ == "__main__":
    main()
