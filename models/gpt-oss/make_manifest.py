#!/usr/bin/env python3
"""make_manifest.py - describe the gpt-oss vaults for publishing: small files are copied into this folder (they go into the git
repository); each vault.bin is described as parts under GitHub's 2 GiB release-asset limit, with a SHA-256 per part and for the
whole file. Run from the harness: python models\\gpt-oss\\make_manifest.py"""
import hashlib, json, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent; sys.path.insert(0, str(ROOT / "app"))
import harness_paths as hp

VAULTS = ["vault_gptoss_20b_mxf2", "vault_gptoss_120b_mxf2"]
PART = 1900 * 2**20                      # 1,992,294,400 bytes: below both 2 GB and 2 GiB
SMALL = ["vault.idx", "vault.json", "resident.bin", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"]
TAG = "gpt-oss-vaults-v1"


def main():
    man = {"release_tag": TAG, "part_bytes": PART, "vaults": {}}
    for v in VAULTS:
        src = hp.resolve_vault(v); dst = HERE / v; dst.mkdir(exist_ok=True)
        for f in SMALL:
            if (src / f).exists(): shutil.copy2(src / f, dst / f)
        big = src / "vault.bin"; size = big.stat().st_size; whole = hashlib.sha256(); parts = []
        with open(big, "rb") as fh:
            for i, off in enumerate(range(0, size, PART)):
                h = hashlib.sha256(); left = min(PART, size - off)
                while left:
                    b = fh.read(min(64 << 20, left)); h.update(b); whole.update(b); left -= len(b)
                parts.append({"asset": f"{v}.vault.bin.part{i:03d}", "offset": off, "size": min(PART, size - off), "sha256": h.hexdigest()})
                print(f"{v} part {i}: {parts[-1]['size']:,} bytes", flush=True)
        man["vaults"][v] = {"vault_bin_size": size, "vault_bin_sha256": whole.hexdigest(), "parts": parts, "files": [f for f in SMALL if (dst / f).exists()]}
    json.dump(man, open(HERE / "manifest.json", "w"), indent=1); print("manifest.json written")


if __name__ == "__main__":
    main()
