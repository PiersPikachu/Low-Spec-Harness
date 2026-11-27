#!/usr/bin/env python3
"""launch_check.py - run by cube-harness.bat before the window opens. Makes the harness work wherever its folder is and on
whatever machine it is on.

  exit 0  ready                          exit 2  first-time setup needed (Python packages missing)
  exit 1  cannot start (reason in data\\logs\\launch.txt)

  folder moved     (the recorded install folder differs from where it runs now): model folders that pointed inside the old
                   location are re-based to the new one; everything else is stored relative to the harness folder already
  other machine    (computer name / CPU / RAM differ): the hardware profile is set aside so the planner re-measures this machine,
                   and the engine is test-run; if it does not run here (e.g. built for another CPU) it is rebuilt when gcc exists
  model folders    folders that are not reachable now (a USB drive unplugged) are reported, not removed
"""
import importlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness_paths as hp

LOG = []
def say(s): LOG.append(s); print(s, flush=True)


def machine_id():
    try: ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except Exception:
        import ctypes
        class M(ctypes.Structure): _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "a", "b", "c", "d", "e")]
        m = M(); m.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); ram = round(m.tp / 2**30)
    return f"{platform.node()}|{platform.processor()}|{ram}GB"


def main():
    hp.ensure_dirs(); code = 0
    # 1 packages
    missing = []
    for mod in ("numpy", "transformers", "tokenizers", "safetensors", "jinja2"):
        try: importlib.import_module(mod)
        except Exception: missing.append(mod)
    try: import tkinter  # noqa: F401
    except Exception: say("tkinter is missing: reinstall Python with the 'tcl/tk and IDLE' option"); return 1
    if missing: say(f"missing Python packages: {', '.join(missing)} - running first-time setup"); return 2
    cfg = hp.load_config(); here = str(hp.ROOT); mid = machine_id()
    # 2 folder moved
    old = cfg.get("install_root")
    if old and Path(old).resolve() != hp.ROOT:
        say(f"the harness folder moved: {old} -> {here}")
        md = []
        for e in cfg.get("model_dirs", []):
            p = str(e["path"])
            if Path(p).is_absolute() and p.lower().startswith(old.lower()): e = dict(e, path=hp.rel(here + p[len(old):])); say(f"  model folder re-based: {p} -> {e['path']}")
            md.append(e)
        cfg["model_dirs"] = md
    # 3 other machine
    if cfg.get("machine") and cfg["machine"] != mid:
        say(f"running on a different machine ({cfg['machine']} -> {mid})")
        if hp.PROFILE.exists(): hp.PROFILE.replace(hp.PROFILE.with_name(f"hw_profile.{time.strftime('%Y%m%d-%H%M%S')}.json")); say("  hardware profile set aside: the planner re-measures this machine on the first load")
        di = hp.DATA / "drive_info.json"
        if di.exists(): di.unlink()
        for v in hp.find_vaults(cfg).values():
            if (v / "tuned.json").exists(): say("  note: tuned model settings came from the other machine - re-run Tune in the Models tab"); break
    # 4 engine runs here?
    if not hp.ENGINE.exists(): say(f"engine missing: {hp.rel(hp.ENGINE)}")
    else:
        try:
            r = subprocess.run([str(hp.ENGINE), "selftest"], capture_output=True, text=True, timeout=120, cwd=str(hp.ENGINE_DIR))
            ok = "SELFTEST PASSED" in r.stdout
        except Exception as ex: ok = False; say(f"engine did not run: {ex}")
        if not ok:
            say("the engine does not run on this machine (it may have been built for another CPU)")
            if shutil.which("gcc"):
                say("  rebuilding it with gcc ..."); subprocess.run(["cmd", "/c", str(hp.ENGINE_DIR / "build.bat")], cwd=str(hp.ENGINE_DIR))
            else: say("  gcc (the C compiler) is needed to rebuild it for this CPU: winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT,"
                    " then run engine\\build.bat - or copy an engine built for this CPU. See README.md, 'gcc: when you need it'"); code = 1
    # 5 model folders reachable?
    for o in hp.model_dirs(cfg):
        if not o["path"].exists(): say(f"model folder not reachable now (drive unplugged?): {o['path']}")
    n = len(hp.find_vaults(cfg)); say(f"{n} model(s) found in {len([o for o in hp.model_dirs(cfg) if o['path'].exists()])} folder(s)")
    if n == 0: say("no models yet: open the Models tab to install one")
    cfg["install_root"] = here; cfg["machine"] = mid; hp.save_config(cfg)
    return code


if __name__ == "__main__":
    try: rc = main()
    except Exception as ex: say(f"launch check failed: {ex}"); rc = 1
    try:
        hp.LOGS.mkdir(parents=True, exist_ok=True)
        (hp.LOGS / "launch.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f"  root {hp.ROOT}\n" + "\n".join(LOG) + "\n", encoding="utf-8")
    except Exception: pass
    sys.exit(rc)
