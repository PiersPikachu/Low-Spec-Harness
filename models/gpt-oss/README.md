# gpt-oss vaults

Compressed versions of OpenAI's **gpt-oss-20b** and **gpt-oss-120b** for the Cube Harness. The vaults are **exact**: the attention
and other bf16 weights are Huffman-coded losslessly, and the MXFP4 expert weights are stored in MXF2, a lossless repacking of the
released MXFP4 blocks and scales. The engine's outputs were checked against a NumPy reference of the original model (relative L2
4.5 × 10⁻⁷ on real layers, same top-5).

| Vault | Size | Measured on a Ryzen 7 5800H, 16 GB RAM, RTX 3060 Laptop 6 GB |
|---|---|---|
| `vault_gptoss_20b_mxf2` | 12.3 GB | ~3.7–4.0 tokens/s, context up to 131,072 tokens (fp16 KV cache) |
| `vault_gptoss_120b_mxf2` | 61.6 GB | ~2.1 tokens/s (experts streamed from the NVMe SSD, cached in RAM + VRAM) |

## Get them

The large `vault.bin` files are too big for a git repository, so they are attached to the GitHub release
**gpt-oss-vaults-v1** as parts under 2 GiB. Don't download those by hand - run:

```
models\gpt-oss\fetch.bat          gpt-oss-20b
models\gpt-oss\fetch.bat 120b     gpt-oss-120b
```

It downloads one part at a time, checks each against its SHA-256 in `manifest.json`, appends it, checks the whole file and finally
runs the engine's check on every stream. An interrupted download resumes where it stopped. The small files in this folder (vault
index, tokenizer, chat template) are copied next to it. The vault lands in the harness's install folder (`data\vaults` unless you
chose another one in the **Model folders** tab); keep it on an SSD.

## Files here

- `manifest.json` - for each vault: size and SHA-256 of `vault.bin` and of every release part
- `vault_gptoss_*/` - the vault's small files
- `fetch.bat`, `fetch_models.py` - download and rebuild
- `make_manifest.py` - how the manifest was made (publisher side)
- `NOTICE.md` - origin and licence

The models are © OpenAI, released under the Apache License 2.0 - see `NOTICE.md`.
