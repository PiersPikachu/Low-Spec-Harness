#!/usr/bin/env python3
"""fetch_models.py [20b] [120b] [--base URL] - download the compressed gpt-oss vaults from this repository's GitHub release and
rebuild them in data\\vaults\\ (or the install folder chosen in the Model folders tab).

Each vault.bin is published as parts under 2 GiB (GitHub's limit per release file). Parts are downloaded one at a time, checked
against the SHA-256 in manifest.json, appended to vault.bin and deleted, so the disk only ever holds the vault plus one part.
Interrupted downloads resume from the last verified part. At the end the whole file's SHA-256 and every stream of the vault
(engine check) are verified. The small files (index, tokenizer, chat template) come from this folder.
  20b  ~12.3 GB      120b  ~61.6 GB"""
import hashlib, json, shutil, sys, time, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent; sys.path.insert(0, str(ROOT / "app"))
import harness_paths as hp

REPO = "PiersPikachu/Low-Spec-Harness"
NAMES = {"20b": "vault_gptoss_20b_mxf2", "120b": "vault_gptoss_120b_mxf2"}


def sha_file(p, off=0, size=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        f.seek(off); left = size if size is not None else Path(p).stat().st_size - off
        while left:
            b = f.read(min(64 << 20, left)); h.update(b); left -= len(b)
    return h.hexdigest()


def fetch(name, man, base, dest_root):
    v = man["vaults"][name]; dest = dest_root / name; dest.mkdir(parents=True, exist_ok=True)
    need = v["vault_bin_size"] + 3e9
    if shutil.disk_usage(dest_root).free < need - ((dest / "vault.bin").stat().st_size if (dest / "vault.bin").exists() else 0):
        sys.exit(f"not enough space in {dest_root}: {name} needs {v['vault_bin_size'] / 1e9:.1f} GB plus room for one part")
    for f in v["files"]: shutil.copy2(HERE / name / f, dest / f)
    out = dest / "vault.bin"; have = out.stat().st_size if out.exists() else 0
    done = 0                                              # resume: keep only whole, verified parts already in vault.bin
    for p in v["parts"]:
        if p["offset"] + p["size"] <= have and sha_file(out, p["offset"], p["size"]) == p["sha256"]: done += 1
        else: break
    keep = sum(p["size"] for p in v["parts"][:done])
    if have != keep:
        with open(out, "r+b" if out.exists() else "wb") as f: f.truncate(keep)
    print(f"{name}: {done}/{len(v['parts'])} parts already here", flush=True)
    for p in v["parts"][done:]:
        tmp = dest / (p["asset"] + ".download"); url = f"{base}/{p['asset']}"
        for attempt in range(3):
            t0 = time.time(); urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size == p["size"] and sha_file(tmp) == p["sha256"]: break
            print(f"  {p['asset']}: checksum mismatch, retrying", flush=True)
        else: sys.exit(f"{p['asset']} failed verification 3 times")
        with open(out, "ab") as f, open(tmp, "rb") as src: shutil.copyfileobj(src, f, 64 << 20)
        tmp.unlink(); mb = p["size"] / 1e6 / max(time.time() - t0, 1e-3)
        print(f"  {p['asset']} ok ({p['size'] / 1e9:.2f} GB, {mb:.0f} MB/s)", flush=True)
    print(f"{name}: checking the whole file ...", flush=True)
    if sha_file(out) != v["vault_bin_sha256"]: sys.exit(f"{name}: vault.bin checksum mismatch - delete it and run again")
    if hp.ENGINE.exists():
        import subprocess
        r = subprocess.run([str(hp.ENGINE), "check", str(dest)], capture_output=True, text=True); last = (r.stdout.strip().splitlines() or [""])[-1]
        print("  engine check:", last)
        if "bad=0" not in last: sys.exit(f"{name}: engine check failed")
    print(f"{name}: ready in {dest}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; base = None
    if "--base" in sys.argv: base = sys.argv[sys.argv.index("--base") + 1]; args = [a for a in args if a != base]
    man = json.load(open(HERE / "manifest.json"))
    base = base or f"https://github.com/{REPO}/releases/download/{man['release_tag']}"
    want = [NAMES[a] for a in (args or ["20b"]) if a in NAMES] or sys.exit("usage: fetch_models.py [20b] [120b]")
    dest_root = hp.default_model_dir()
    for n in want: fetch(n, man, base, dest_root)


if __name__ == "__main__":
    main()
