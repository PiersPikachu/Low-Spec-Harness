#!/usr/bin/env python3
"""vault_defrag.py <vault_dir> [--min-extent-mb 16] [--force]
A vault streamed from disk reads at the drive's sequential speed only if vault.bin is contiguous. Packed while C: was nearly full,
vault_hermes14b_rans/vault.bin had 234,023 fragments (84 KB average) and the 14B read at 1.1 GB/s; rewritten contiguously
(146 fragments) one weight pass fell from 12.0 s to 5.2 s.
Method: count the file's extents (fsutil); if they are small, preallocate a new file of the full size in one call
(fsutil file createnew: NTFS then picks large free runs; a plain or robocopy /J copy on a busy volume came out in 26k-195k pieces),
copy the data into it in place, check it is byte-identical (MD5), then swap it in. Needs free space equal to vault.bin. Windows only."""
import argparse, hashlib, shutil, subprocess, sys
from pathlib import Path


def extents(p):
    out = subprocess.run(["fsutil", "file", "queryextents", str(p)], capture_output=True, text=True).stdout
    return sum(1 for l in out.splitlines() if l.strip().startswith("VCN"))


def md5(p, bs=64 << 20):
    h = hashlib.md5()
    with open(p, "rb", buffering=0) as f:
        while True:
            b = f.read(bs)
            if not b: return h.hexdigest()
            h.update(b)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("vault"); ap.add_argument("--min-extent-mb", type=float, default=16); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); v = Path(a.vault); f = v / "vault.bin"; size = f.stat().st_size; n = extents(f); avg = size / 2**20 / max(n, 1)
    print(f"{f}: {size / 2**30:.2f} GiB in {n} extents (average {avg:.1f} MB)", flush=True)
    if avg >= a.min_extent_mb and not a.force: print("contiguous enough; nothing to do"); return
    new = v / "_prealloc.bin"
    if not new.exists() or new.stat().st_size != size:
        if new.exists(): new.unlink()
        free = shutil.disk_usage(v).free
        if free < size * 1.02: sys.exit(f"need {size / 1e9:.1f} GB free on that drive, have {free / 1e9:.1f} GB")
        subprocess.run(["fsutil", "file", "createnew", str(new), str(size)], check=True, capture_output=True)
    m = extents(new); print(f"preallocated: {m} extents (average {size / 2**20 / max(m, 1):.1f} MB)", flush=True)
    if m >= n and not a.force: new.unlink(); sys.exit("preallocation is not better than the current file; free space is too fragmented")
    with open(f, "rb", buffering=0) as src, open(new, "r+b", buffering=0) as dst:
        while True:
            b = src.read(64 << 20)
            if not b: break
            dst.write(b)
    print("copied; verifying ...", flush=True)
    if md5(f) != md5(new): new.unlink(); sys.exit("copy differs from original; removed the copy, original untouched")
    old = v / "vault.bin.old"; f.rename(old); new.rename(f); old.unlink()
    print(f"done: {f} now {extents(f)} extents, byte-identical (MD5)")


if __name__ == "__main__":
    main()
