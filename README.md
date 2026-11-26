# Cube Harness (Rust)

A Rust rewrite of the Low-Spec Harness: run large language models on a modest PC straight from **lossless compressed vaults**,
streaming weights from the SSD, with a native window (egui) for chat, tools, artifacts, skills and model installs.

```
build.bat          build everything (cargo) + unit tests
run.bat            start the window (target\release\cube-harness.exe)
gpu_check.bat V    GPU engine vs CPU engine on a dense vault
target\release\cube.exe ...   command line (below)
data\              everything the harness writes (created on first run)
```

## Crates

| crate | what |
|---|---|
| `cube-vault` | the vault format: v8 manifest (`vault.cv8`, spec in `docs/VAULT_V8.md`) + `vault.bin`; reads v3–v7 vaults (`vault.idx`) directly and converts them in place without touching the weights; huffman4 / rans4 / MXF4 / MXF2 codecs |
| `cube-pack` | lossless packer for Hugging Face checkpoints (dense Qwen2/3, Llama, Mistral; Qwen3-MoE; gpt-oss with MXF2 experts) |
| `cube-engine` | CPU engine: streaming with pinned-RAM budget and one-ahead prefetch, MoE expert LRU cache, fp32/fp16 KV, sliding-window rings; AVX2/FMA kernels; the `nvme_engine_u` serve protocol |
| `cube-gpu` | CUDA engine (cudarc): weights stay compressed in VRAM; GPU-resident mode and split mode (first N layers on the GPU) |
| `cube-core` | tokenizer + chat templates, reply parsing (harmony / `<think>` / `<tool_call>`), tools, skills, conversations, the reply loop, Hugging Face hub, model folders |
| `cube-cli` | `cube.exe` |
| `cube-app` | `cube-harness.exe`, the window |

## Command line

```
cube info|check|convert <vault>          describe / verify every checksum / write vault.cv8 next to a v3-v7 vault
cube pack <model_dir> <vault> [--rans4] [--mxf4]
cube verify-source <vault> <model_dir>   every tensor of a vault vs its source checkpoint
cube serve <vault> [--ctx N] [--kv16] [--gpu | --gpu-layers N] ...   nvme_engine_u protocol on stdin/stdout
cube chat <vault> [--prompt TEXT] [--tools a,b] [--think]
cube bench <vault> [--prompt N] [--tokens N]
cube gpu-check <vault>
```

## What is verified

* **Vault codecs and packer**: byte-identical to the Python packers on every stream; `verify-source` 0 mismatches.
* **CPU engine**: bit-identical to `nvme_engine_u.c` (logits and the hidden state after every layer) on huffman4, rans4,
  Qwen3-MoE, gpt-oss MXF2/MXF4, fp32 and fp16 KV, sliding-window rings, batch sizes 7/16/32, with and without AVX2
  (`tools/parity_vs_c.py`); passes `verify_engine.py synth` and `real` (Qwen3-0.6B) against the numpy reference.
* **Chat templates**: token-identical to `transformers.apply_chat_template` (Qwen3, Qwen3-MoE, gpt-oss; tools, tool calls, thinking).
* **Window**: chat with tool calls and artifact creation end to end on Qwen3-0.6B.
* **GPU engine: compiled, not yet run** (no GPU where it was written). Run `gpu_check.bat <dense vault>` first; it compares
  logits, hidden states, greedy text and split mode against the CPU engine and prints PASS / FAIL.

## Not ported yet

* Speculative decoding with a draft model (`spec.py`).
* The VRAM expert tier and next-layer expert prediction for MoE models (MoE runs on the CPU with the RAM expert cache).
* Rank-1 weight patches (`--patch`).
* Single-file v8 vaults (manifest and weights in one file): specified, not implemented.

## Fixed relative to the Python harness

* Live saving of a file written by a `<tool_call>` model (Qwen3, Hermes) went to an artifact named `artifact`
  (the tool's own name was read as the artifact name).
* A disabled plugin could still be run if the model called it by name.
