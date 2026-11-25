"""storage.py - model folders: what kind of drive each is on, how fast it reads, what it holds, and moving a vault between them.

A vault can live in any model folder, on or off the SSD. The engine streams weights from the vault while it runs, so a folder on
an external HDD/USB drive is fine for keeping models but slow for running them (~0.1 GB/s vs ~2.5 GB/s on the internal NVMe);
move a model to an SSD folder before chatting with it."""
import json, shutil, subprocess, time
from pathlib import Path
import harness_paths as hp

INFO = hp.DATA / "drive_info.json"


def _cache():
    try: return json.load(open(INFO))
    except Exception: return {}


def drive_kind(path):
    """(media, bus, model) of the physical disk holding path, e.g. ('SSD', 'NVMe', 'Samsung ...') or ('HDD', 'USB', ...); cached per drive letter"""
    d = (Path(path).resolve().drive or "?").rstrip(":").upper(); c = _cache()
    if d in c: return tuple(c[d])
    ps = (f"$d = Get-Partition -DriveLetter {d} -ErrorAction Stop | Get-Disk; $p = Get-PhysicalDisk | Where-Object DeviceId -eq $d.Number; "
          f"\"$($p.MediaType)|$($p.BusType)|$($d.FriendlyName)\"")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[-1]
        media, bus, model = (out.split("|") + ["", "", ""])[:3]
        media = {"Unspecified": "?", "": "?"}.get(media, media)
    except Exception:
        media, bus, model = "?", "?", "?"
    c[d] = [media, bus, model]; hp.DATA.mkdir(exist_ok=True); json.dump(c, open(INFO, "w"), indent=1)
    return media, bus, model


def read_speed(path, measure=False):
    """MB/s of the drive (from the hardware profile; measure=True reads a sample of the largest vault in the folder)"""
    import planner as pl
    folder = Path(path); d = (folder.resolve().drive or "?").upper()
    try: prof = json.load(open(hp.PROFILE))
    except Exception: prof = {}
    known = prof.get("disk_MBps", {}).get(d)
    if not measure: return known
    vaults = sorted((v for v in folder.iterdir() if hp.is_vault(v)), key=lambda v: -(v / "vault.bin").stat().st_size) if folder.is_dir() else []
    if not vaults: return known
    return pl.drive_MBps(vaults[0], force=True)


def folder_summary(path):
    p = Path(path)
    if not p.exists(): return dict(exists=False, free=0, vaults=0, size=0)
    vs = [v for v in p.iterdir() if hp.is_vault(v)] if p.is_dir() else []
    return dict(exists=True, free=shutil.disk_usage(p).free, vaults=len(vs), size=sum((v / "vault.bin").stat().st_size for v in vs), names=[v.name for v in vs])


def move_vault(src, dst_dir, log=print):
    """copy a vault to another model folder, verify every stream of the copy with the engine, then delete the original"""
    src, dst_dir = Path(src), Path(dst_dir); dst = dst_dir / src.name; t0 = time.time()
    if dst.exists(): raise RuntimeError(f"{dst} already exists")
    need = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()); dst_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dst_dir).free < need + 2e9: raise RuntimeError(f"not enough space in {dst_dir}: needs {need / 1e9:.1f} GB + 2 GB margin")
    tl = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    if "nvme_engine_u.exe" in tl: raise RuntimeError("an engine is running - unload the model first")
    log(f"copying {src.name} ({need / 1e9:.1f} GB) to {dst_dir} ...")
    r = subprocess.run(["robocopy", str(src), str(dst), "/E", "/J", "/NP", "/NFL", "/NDL", "/R:2", "/W:5"], capture_output=True, text=True)
    if r.returncode >= 8: raise RuntimeError(f"robocopy failed ({r.returncode})")
    log(f"copied in {(time.time() - t0) / 60:.1f} min - verifying every stream of the copy ...")
    c = subprocess.run([str(hp.ENGINE), "check", str(dst)], capture_output=True, text=True); last = (c.stdout.strip().splitlines() or [""])[-1]
    log(last)
    if c.returncode != 0 or "bad=0" not in last: raise RuntimeError("verification failed - the original was kept, the copy left for inspection")
    shutil.rmtree(src); log(f"moved: {dst}  ({(time.time() - t0) / 60:.1f} min)"); return dst
