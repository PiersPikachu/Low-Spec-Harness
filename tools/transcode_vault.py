#!/usr/bin/env python3
"""transcode_vault.py <vault_in> <vault_out> [--codec rans4|huffman4]
Re-encodes every weight matrix of an existing vault into another codec WITHOUT the source safetensors:
decode (bit-exact) -> encode -> decode the new blob and compare to the old decode (must be bit-identical) -> write.
resident.bin is copied unchanged; vault.idx/vault.json are rewritten (stream table, codec, layout lines). Stream hashes are recomputed."""
import sys as _sys; from pathlib import Path as _P; _ROOT = next(p for p in _P(__file__).resolve().parents if (p / "app" / "harness_paths.py").exists()); _sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "tools"), str(_ROOT / "tools" / "research")]
import json, os, shutil, struct, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pack_universal_v3 as pu, rans4

SECT, MAGIC = pu.SECT, pu.MAGIC


def dec(blob, store):
    return rans4.decode(blob) if store == "rans4" else pu.decode_blob(blob)


def enc(w, codec):
    return rans4.encode(w, 12, 16) if codec == "rans4" else pu.compress_matrix(w)


def main(vin, vout, codec):
    vin, vout = Path(vin), Path(vout); vout.mkdir(parents=True, exist_ok=True); t0 = time.time()
    V = json.load(open(vin / "vault.json")); store_new = "rans4" if codec == "rans4" else "huff4"
    by_stream = {}
    for t in V["tensors"]:
        if t["store"] in ("huff4", "rans4"): by_stream.setdefault(t["stream"], []).append(t)
    raw_tot = new_tot = 0
    with open(vin / "vault.bin", "rb") as fi, open(vout / "vault.bin", "wb") as fo:
        for si, S in enumerate(V["streams"]):
            fi.seek(S["off"]); old = np.frombuffer(fi.read(S["padded"]), np.uint8)
            if int(pu.hash64(old)) != S["hash"]: raise SystemExit(f"input stream {si} fails its hash")
            subs = []; ts = sorted(by_stream[si], key=lambda t: t["sub"])
            for t in ts:
                ob = old[t["blob_off"]:t["blob_off"] + t["blob_size"]]; w = dec(ob, t["store"]); nb = enc(np.ascontiguousarray(w), codec)
                if not np.array_equal(dec(nb, store_new), w): raise SystemExit(f"VERIFY FAILED: {t['name']}")
                subs.append(nb); t["store"] = store_new; t["blob_size"] = int(nb.size)
            n = len(subs); hdr = max(128, (8 + 8 * n + 63) & ~63); pos = hdr; offs = []
            for b in subs: pos = (pos + 63) & ~63; offs.append(pos); pos += b.size
            padded = (pos + SECT - 1) & ~(SECT - 1); buf = np.zeros(padded, np.uint8)
            buf[0:8] = np.frombuffer(struct.pack('<II', MAGIC, n), np.uint8)
            for i, (o, b) in enumerate(zip(offs, subs)):
                buf[8 + 8 * i:16 + 8 * i] = np.frombuffer(struct.pack('<Q', o), np.uint8); buf[o:o + b.size] = b; ts[i]["blob_off"] = o
            S.update(off=fo.tell(), size=int(pos), padded=int(padded), hash=int(pu.hash64(buf))); fo.write(buf.tobytes())
            raw_tot += S["raw"]; new_tot += pos
            print(f"\rstream {si + 1}/{len(V['streams'])}  {time.time() - t0:.0f}s", end="", flush=True)
    print()
    shutil.copy(vin / "resident.bin", vout / "resident.bin")
    lines = open(vin / "vault.idx").read().splitlines(); ver = int(lines[0].split()[1])
    head = [l for l in lines[1:] if not l.startswith("stream ") and l.split()[0] not in ("codec",)]
    if ver == 3:   # v3 = implicit Qwen3 layout; v4 needs it written out
        sc = V["scalars"]; hid, hd = sc["hidden"], sc["head_dim"]
        lay = (V.get("engine_u") or {}).get("layout") or [["ln_in", 0, hid], ["ln_post", hid, hid], ["qn", 2 * hid, hd], ["kn", 2 * hid + hd, hd]]
        tot = sum(n for _, _, n in lay)
        head += [f"lres {tot}"] + [f"r_{r} {o}" for r, o, _ in lay] + ["supported 1"]
    keep_v3 = codec == "huffman4" and ver == 3
    out = [f"WVAULT {3 if keep_v3 else 4}"] + [l for l in head if not (keep_v3 is False and False)]
    hdr_part = [l for l in out[1:] if l.split()[0] in ("hidden", "inter", "layers", "heads", "kv_heads", "head_dim", "vocab", "tie", "eos", "eos2", "eps", "theta", "nstreams")]
    lay_part = [l for l in out[1:] if l not in hdr_part]
    txt = [out[0]] + hdr_part + [f"stream {i} {S['off']} {S['size']} {S['padded']} {S['raw']} {S['hash']:x}" for i, S in enumerate(V["streams"])]
    if not keep_v3: txt += lay_part
    if codec == "rans4": txt.append("codec 1")
    open(vout / "vault.idx", "w").write("\n".join(txt) + "\n")
    V["codec"] = "rans4-m12-r16" if codec == "rans4" else "huffman4-v3"; V["engine_compatible_v3"] = keep_v3; V["transcoded_from"] = str(vin)
    json.dump(V, open(vout / "vault.json", "w"))
    print(f"transcoded {vin.name} -> {vout.name}: {codec}, stored {new_tot / 1e9:.4f} GB, ratio {new_tot / raw_tot:.4f}, every matrix verified bit-identical, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]; codec = a[a.index("--codec") + 1] if "--codec" in a else "rans4"
    main(a[0], a[1], codec)
