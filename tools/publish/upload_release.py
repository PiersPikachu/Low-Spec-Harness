#!/usr/bin/env python3
"""upload_release.py - upload the gpt-oss vault parts listed in models/gpt-oss/manifest.json to the GitHub release, using the
GitHub CLI (gh) with the login you made with 'gh auth login' (this script never sees a password or token).

One part at a time: it is cut from vault.bin into a temporary file (1.9 GB), checked against its SHA-256, uploaded, deleted.
Parts already on the release with the right size are skipped, so an interrupted upload can simply be run again."""
import hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "app"))
import harness_paths as hp

REPO = "PiersPikachu/Low-Spec-Harness"


def gh(*a, check=True):
    r = subprocess.run(["gh", *a], capture_output=True, text=True)
    if check and r.returncode: sys.exit(f"gh {' '.join(a)} failed:\n{r.stderr.strip()}")
    return r


def dry_run():
    """cut and hash the first part of each vault without uploading anything (checks the manifest against the local vaults)"""
    man = json.load(open(ROOT / "models" / "gpt-oss" / "manifest.json"))
    for name, v in man["vaults"].items():
        p = v["parts"][0]; h = hashlib.sha256()
        with open(hp.resolve_vault(name) / "vault.bin", "rb") as f:
            left = p["size"]
            while left:
                b = f.read(min(64 << 20, left)); h.update(b); left -= len(b)
        print(f"{p['asset']}: {'matches' if h.hexdigest() == p['sha256'] else 'DOES NOT MATCH'} the manifest; {len(v['parts'])} parts, {v['vault_bin_size']:,} bytes")


def main():
    if "--dry-run" in sys.argv: return dry_run()
    man = json.load(open(ROOT / "models" / "gpt-oss" / "manifest.json")); tag = man["release_tag"]
    if gh("release", "view", tag, "--repo", REPO, check=False).returncode:
        gh("release", "create", tag, "--repo", REPO, "--title", "gpt-oss vaults (lossless MXF2)",
           "--notes", "Compressed gpt-oss-20b and gpt-oss-120b vaults for the Cube Harness, split into parts under 2 GiB. "
                      "Do not download these by hand: run models\\gpt-oss\\fetch.bat, which downloads, verifies and reassembles them. "
                      "Derived from openai/gpt-oss-20b and openai/gpt-oss-120b (Apache 2.0); see models/gpt-oss/NOTICE.md.")
    have = {a["name"]: a["size"] for a in json.loads(gh("release", "view", tag, "--repo", REPO, "--json", "assets").stdout)["assets"]
            if a.get("state", "uploaded") == "uploaded"}           # a half-finished upload is not skipped
    tmpdir = Path(tempfile.mkdtemp(prefix="vault_parts_"))
    total = sum(len(v["parts"]) for v in man["vaults"].values()); n = 0
    try:
        for name, v in man["vaults"].items():
            src = hp.resolve_vault(name) / "vault.bin"
            for p in v["parts"]:
                n += 1
                if have.get(p["asset"]) == p["size"]: print(f"[{n}/{total}] {p['asset']} already uploaded", flush=True); continue
                tmp = tmpdir / p["asset"]; h = hashlib.sha256()
                with open(src, "rb") as f, open(tmp, "wb") as o:
                    f.seek(p["offset"]); left = p["size"]
                    while left:
                        b = f.read(min(64 << 20, left)); h.update(b); o.write(b); left -= len(b)
                if h.hexdigest() != p["sha256"]: sys.exit(f"{p['asset']}: the local vault does not match the manifest - rerun make_manifest.py")
                print(f"[{n}/{total}] uploading {p['asset']} ({p['size'] / 1e9:.2f} GB) ...", flush=True)
                for attempt in range(1, 6):                       # network hiccups (DNS, resets) are retried, not fatal
                    r = gh("release", "upload", tag, str(tmp), "--repo", REPO, "--clobber", check=False)
                    if r.returncode == 0: break
                    wait = 30 * attempt
                    print(f"    upload failed (attempt {attempt}/5): {r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else '?'} - retrying in {wait} s", flush=True)
                    time.sleep(wait)
                else: sys.exit(f"{p['asset']}: upload failed 5 times - check the connection and run the script again (finished parts are kept)")
                tmp.unlink()
    finally:
        for f in tmpdir.glob("*"): f.unlink()
        tmpdir.rmdir()
    print(f"all {total} parts are on https://github.com/{REPO}/releases/tag/{tag}")


if __name__ == "__main__":
    main()
