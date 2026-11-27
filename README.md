# Cube Harness (Rust)

A Rust rewrite of the Low-Spec Harness: run large language models on a modest PC straight from **lossless compressed vaults**,
streaming weights from the SSD, with a native window (egui) for chat, tools, artifacts, skills and model installs.

```
build.bat          build everything (cargo) + unit tests
run.bat            start the window (target\release\cube-harness.exe)
gpu_check.bat V    GPU engine vs CPU engine on a dense vault
searxng_setup.bat  your own SearXNG search engine in WSL for the web plugin (start / stop: searxng_start.bat, searxng_stop.bat)
web_check.bat      live test of the web plugin: search, read, loop protection, unsafe-site blocking
bench_gptoss.bat V gpt-oss on real chat replies: output layer on the CPU vs on the GPU (both with RAM+VRAM expert tiers)
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
* **GPU engine** (RTX 3060 Laptop): Qwen3-0.6B GPU-resident logits relL2 1.9e-6, greedy 32/32 identical, 47.5 tok/s;
  Qwen3-4B split relL2 1.4e-6.
* **gpt-oss VRAM expert tier**: experts evicted from the RAM cache drop into VRAM (stored as in the vault, MXF4/MXF2) and run there.
  Tiering logic checked without a GPU (`--dev-experts-cpu N`: same tiers, host memory): serve output bit-identical to
  `nvme_engine_u` with 3 and 1000 device slots. On the RTX 3060 (gpt-oss-20b, every expert from the device tier): prompt logits relL2 3.1e-7, greedy 16/16 identical.
  Real chat (`cube chat-bench`, warm): 3.64 tok/s CPU-only -> 4.81 tok/s with the VRAM tier (cache hits 85% -> 99%).
* **Output layer on the GPU** (gpt-oss, with the expert tiers): its pinned RAM goes to the expert cache, so the number of
  cached experts stays about the same (345 RAM + 371 VRAM -> 431 RAM + 309 VRAM). Real chat, warm: 4.65 -> 5.92 tok/s;
  gpu-check logits relL2 3.7e-7, greedy 16/16. Host stand-in (`--dev-head-cpu`) bit-identical to `nvme_engine_u` on every test vault.

## Web plugin and skill

`data/plugins/web.py` (tool `web`) and `data/skills/web-research.md` are installed by the app (and updated with it,
unless you edit them). Turn the tool on in the Plugins tab; the skill is added automatically while it is on.

* `search` -> results S1.. from a configured API first (`searxng_url`, `tavily_api_key` or `brave_api_key` in
  `data/plugins/web_config.json`), else keyless pages: DuckDuckGo HTML, DuckDuckGo lite, Bing. An engine that
  answers with a bot check is left alone for 30 minutes. Keyless search is best effort (DuckDuckGo challenges
  automated clients often); an API or your own SearXNG is the reliable route. `searxng_setup.bat` installs SearXNG
  in a WSL distro named `searxng` (127.0.0.1:8888, this PC only, JSON output on) and sets `searxng_url` /
  `searxng_wsl`; the plugin then starts it on demand. SearXNG's Tor engines (ahmia, torch) are removed; its
  wikipedia / wikidata / openstreetmap engines need a contact for those services - add `"searxng_contact_url":
  "mailto:you@example.com"` (or a web page) to web_config.json and re-run searxng_setup.bat to enable them. Your own SearXNG address is exempt from the address checks;
  its results are not.
  `open` (target S#/L#/URL + a required `why`) -> page P# with its links L#, `more` -> another part of a page read,
  `status` -> what was read.
* Safety: http/https only, no credentials in URLs, no private / local / reserved addresses (checked again after DNS
  and after every redirect), no unusual ports or raw-IP hosts, look-alike (mixed-script) domains blocked, no file
  downloads (by extension, content type and Content-Disposition), URLhaus malware and phishing-filter domain lists
  (refreshed daily), your own `web_blocklist.txt`. Page text is fenced as data, not instructions.
* No loops: a research session keeps a ledger - pages read are never fetched again (normalized URLs: www, tracking
  parameters, fragments, slashes) and mirrors with identical text are recognised; repeated revisits tell the model to
  answer. Searching again is fine: the same query shows the next results (8 at a time from what was fetched, then the
  engine's next page). Budgets per session: 10 engine requests, 8 pages, 3 per site, 6 'more' (web_config.json:
  max_searches, results_per_search, max_results, max_opens, max_per_site, max_more).
* Context: older web results keep only their header and matched passages (`TOOL_DETAILS` marker in the harness).
* With the web tool on, a reply may make at least 12 tool calls.

## Not ported yet

* Speculative decoding with a draft model (`spec.py`).
* Next-layer expert prediction / prefetch; the VRAM expert tier for Qwen3-MoE (gpt-oss has it).
* Rank-1 weight patches (`--patch`).
* Single-file v8 vaults (manifest and weights in one file): specified, not implemented.

## Fixed relative to the Python harness

* Live saving of a file written by a `<tool_call>` model (Qwen3, Hermes) went to an artifact named `artifact`
  (the tool's own name was read as the artifact name).
* A disabled plugin could still be run if the model called it by name.
