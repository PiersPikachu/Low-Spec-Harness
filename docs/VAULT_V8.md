# Cube vault format v8

A vault is a folder. v8 replaces the text index (`vault.idx`), the JSON side file (`vault.json`), `resident.bin` and the loose
tokenizer files with one checksummed binary manifest, `vault.cv8`. The large weight streams stay in `vault.bin`, in the same
layout as vaults 3-7, so an existing vault becomes a v8 vault by writing one small file next to it (`cube convert`): nothing is
re-encoded, re-downloaded or copied.

```
vault_x/
  vault.cv8     manifest: header + sections (model description, stream table, resident tensors, tokenizer, chat template)
  vault.bin     weight streams (unchanged from v3-v7), each stream sector-aligned (4096) for direct I/O
```

A single-file vault (`flags.EMBEDDED_DATA`) carries the streams in a `DATA` section of `vault.cv8` itself; nothing else changes.

## Why the bitstreams are kept

The weight codecs are already at their measured limit: huffman4 is within 0.1-0.2% of the order-0 entropy of bf16 exponents, and
sign+mantissa bytes are incompressible (log E1, X16). A new codec would cost a repack of every vault for well under 1%. What v8
fixes is everything around the streams: typed metadata, one file instead of four, checksums on every section, data the engine
needs at load time in one read, and an explicit codec per tensor.

## Header (64 bytes, little endian)

| off | type | field |
|---|---|---|
| 0 | [u8; 8] | magic `CUBEVLT8` |
| 8 | u32 | version = 8 |
| 12 | u32 | flags: bit 0 `EMBEDDED_DATA`, bit 1 `CONVERTED` (made from a v3-v7 vault) |
| 16 | u64 | section table offset |
| 24 | u32 | section count |
| 28 | u32 | reserved (0) |
| 32 | u64 | total length of `vault.cv8` |
| 40 | u64 | xxh3-64 of the section table |
| 48 | [u8; 16] | reserved (0) |

## Section table

One 32-byte entry per section: `tag [u8;4]`, `flags u32`, `offset u64`, `length u64`, `xxh3 u64` (of the section bytes).
Sections start on 64-byte boundaries; `DATA` starts on a 4096-byte boundary.

| tag | contents |
|---|---|
| `META` | UTF-8 JSON, `ModelMeta` (below) |
| `STRM` | stream table: per stream 64 bytes: `offset u64, size u64, padded u64, raw u64, hash u64, hash_kind u8, role u8, nsub u16, layer u32, expert u32, pad [u8;12]` |
| `RESI` | resident tensors, f32 little endian (norms, biases, attention sinks, router biases), in `META.resident` order |
| `ROPE` | optional f32 table: inverse frequencies (gpt-oss YaRN) |
| `TOKJ` | `tokenizer.json` |
| `TOKC` | `tokenizer_config.json` |
| `CHAT` | chat template (jinja) |
| `GENC` | `generation_config.json` |
| `SRCC` | the source model's `config.json`, verbatim |
| `DATA` | streams, only with `EMBEDDED_DATA` |

Unknown tags are skipped, so later versions can add sections.

`hash_kind`: 1 = xxh3-64 of the padded stream (new packs), 2 = the v3-v7 FNV-style `hash64` (converted vaults keep their
original hashes, so a converted vault still verifies against the hashes made when it was packed).

`role`: 0 layer (attention + dense MLP or router), 1 embedding, 2 lm_head, 3 expert, 4 extra.

## Streams (unchanged from v3-v7)

A stream is a super-blob: `u32 magic 0x534C5657, u32 n`, then n `u64` sub-blob offsets (from the stream start), sub-blobs
64-aligned, the stream zero-padded to 4096. Sub-blob codecs:

| codec | layout |
|---|---|
| `huff4` | bf16 matrix: `u32 rows, u32 cols, u8 len[256]` (Huffman code lengths of the exponent byte, max 12), `u32 off[4*rows+1]` (bit-stream offsets, 4 interleaved column streams per row), exponent bit-streams, 8 bytes pad, `u8 sm[rows*cols]` (sign bit 7, mantissa bits 0-6) |
| `rans4` | bf16 matrix, exponents rANS-coded (M = 2^12, R rows per chain, 4 interleaved chains): `u32 rows, u32 cols, u8 mbits, u8 R, u16 0, u16 freq[256], u32 off[4*nblk+1]`, chains, 8 pad, `sm` |
| `mxf4` | MXFP4 as released: `"MXF4", u32 rows, u32 cols`, `u8 scales[rows][cols/32]`, `u8 blocks[rows][cols/2]` |
| `mxf2` | MXFP4 with 2-bit packed block scales: `"MXF2", rows, cols, pb, nexc, blocks_off`, `u32 desc[rows]`, `u8 packed[rows][pb]`, exception rows, blocks |
| `f32` | `"F32 ", u32 n, f32[n]` |

## ModelMeta (META)

```json
{
  "format": 8,
  "arch": "dense" | "qwen_moe" | "gptoss",
  "hidden": 1024, "inter": 3072, "layers": 28, "heads": 16, "kv_heads": 8, "head_dim": 128, "vocab": 151936,
  "tie_embeddings": true, "eos": [151645, 151643], "rms_eps": 1e-6,
  "rope": {"kind": "default" | "llama3" | "table", "theta": 1000000.0, "factor": 0, "low": 0, "high": 0, "orig": 0, "scale": 1.0},
  "layer_resident": {"len": 2304, "ln_in": 0, "ln_post": 1024, "q_norm": 2048, "k_norm": 2176, "bq": null, ...},
  "moe": {"experts": 32, "top_k": 4, "inter": 2880, "norm_topk": false, "expert_stream0": 26} | null,
  "gptoss": {"sliding_window": 128, "layer_sliding": [1, 0, ...], "swiglu_limit": 7.0, "swiglu_alpha": 1.702} | null,
  "streams": {"layers": [0, 28), "embed": 28, "lm_head": 28, "experts": [..]},
  "tensors": [{"name": "...", "shape": [..], "codec": "huff4", "stream": 3, "sub": 1} | {"name": "...", "resident": [off, n]}],
  "lossy": false,
  "source": {"repo": "Qwen/Qwen3-0.6B", "packer": "cube-pack 0.1", "converted_from": 3}
}
```
