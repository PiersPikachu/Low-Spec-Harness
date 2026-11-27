# Cube Harness

A local chat harness that runs large language models from **losslessly compressed vaults** on an ordinary laptop: weights stay
compressed on disk and in memory and are decoded on the fly, so the output is the same as the original model's. Mixture-of-experts
models (gpt-oss-20b / 120b, Qwen3-30B-A3B) run with an expert cache in RAM and VRAM; dense models stream from the SSD, optionally
with a small draft model (speculative decoding).

## Start

Double-click **cube-harness.bat**. It finds Python, adapts the harness to the folder and machine it is on (first-time setup if
packages are missing), then opens the window. The folder can be moved or copied anywhere; see *Moving the harness* below.

Requirements: Windows, Python 3.10+ with tkinter, ~16 GB RAM recommended, an NVMe SSD for the models you run. A CUDA GPU is optional.
**gcc is not needed to run the harness** - only to rebuild the CPU engine (see *gcc: when you need it* below).

## gcc: when you need it

**gcc** is the GNU C compiler (on Windows: **MinGW-w64**). The harness uses it for one thing: compiling the CPU engine
`engine\nvme_engine_u.exe` from its C source, `engine\nvme_engine_u.c`.

- **You do not need it** if the engine that ships in `engine\` runs on your PC. It is prebuilt, so most people never install gcc.
- **You need it** when the shipped engine does not run on your CPU. It is compiled with `-march=native`, i.e. for the CPU it was
  built on (a Zen 3 Ryzen, using AVX2), so an older or different CPU can refuse it. `cube-harness.bat` test-runs the engine at
  every start (`app\launch_check.py`); if it fails and gcc is installed, it rebuilds the engine automatically. Without gcc it
  tells you so in `data\logs\launch.txt`.
- **You also need it** if you change the engine source and want to rebuild it yourself: `engine\build.bat`.

Install it (any one of these), then open a new terminal and check that `gcc --version` prints a version:

- winget: `winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT` (the WinLibs build of GCC + MinGW-w64)
- MSYS2 (https://www.msys2.org): `pacman -S mingw-w64-ucrt-x86_64-gcc`, then add `C:\msys64\ucrt64\bin` to PATH

The build line, in `engine\build.bat`:

```
gcc -O3 -march=native -ffp-contract=off -fopenmp -static nvme_engine_u.c -o nvme_engine_u.exe -lm -lpsapi
```

| Flag | Why |
|---|---|
| `-O3` | full optimisation |
| `-march=native` | use every instruction this CPU has (AVX2 kernels) - the reason a copy built elsewhere may not run here |
| `-ffp-contract=off` | no fused multiply-add: results stay bit-exact across builds (the output must match the original model) |
| `-fopenmp` | multithreading |
| `-static` | no MinGW DLLs needed next to the .exe |
| `-lm -lpsapi` | maths library; Windows process-memory API |

The GPU engine (`engine\gpu\gpu_engine.exe`) is a separate build with CUDA's nvcc plus Visual Studio Build Tools - gcc is
not involved there.

## Models

- **gpt-oss-20b and gpt-oss-120b** (exact compressed vaults, 12.3 GB / 61.6 GB) are published on this repository's release
  `gpt-oss-vaults-v1`. Get them with `models\gpt-oss\fetch.bat` (20b) or `models\gpt-oss\fetch.bat 120b` - it downloads,
  verifies and installs them. See `models/gpt-oss/README.md`.
- **Any other compatible model**: open the **Models** tab, search Hugging Face, and press Install (download, compress, check, tune).

## The window

| Tab | What it is for |
|---|---|
| Messages | Chats. Reasoning shows live and collapses to a preview; a live activity tree shows each step and tool call. Stop / Force stop. |
| Artifacts | Notes, pages, sites, code and data the model writes with the `artifact` tool; watch them being written live, edit (autosaved), revert any change, open HTML in the browser, export .zip. |
| Models | Search Hugging Face, see a decision tree of whether each model runs on this machine and how fast, then install it (download, compress, check, tune). |
| Model folders | The folders that hold compressed models - several, on any drive. Drive type and read speed, install target, download folder, move a model between folders. |
| Config | Model and generation settings; **Model defaults** fills in the model's recommended (and tuned) values. |
| Plugins | Tools the model can call; plugins the model wrote wait here for your review before they load. |
| Skills | Instruction snippets (`/name`); model-written ones are marked. |

## Folder layout

```
cube-harness/
  cube-harness.bat     start here
  README.md
  app/                 the harness: harness_app.py (window), harness_core.py, harness_paths.py (all paths), chat.py (engine
                       launcher), planner.py, model_hub.py, storage.py, launch_check.py, setup_harness.py, spec.py
  engine/              nvme_engine_u.exe (CPU/MoE engine), gpu_experts.dll, gpu/gpu_engine.exe; sources + build.bat
  tools/               pack_universal_v3.py / pack_gptoss.py (compressors), vault tools; research/, patches/, bat/
  models/gpt-oss/      manifests and small files of the published gpt-oss vaults, fetch.bat to download them
  tests/               test scripts (tests/data/ test vaults are not published)
  data/                everything the harness writes - vaults/ (built-in model folder), conversations/, artifacts/,
                       skills/, plugins/, workspace/, hub_cache/, logs/, harness_config.json, hw_profile.json
```

## Moving the harness

Everything inside the folder is stored relative to it. When `cube-harness.bat` starts the harness from a new location it re-bases
model folders that pointed inside the old location. On a different computer it sets aside the old hardware measurements (the
planner re-measures), test-runs the engine and rebuilds it if it does not run on this CPU (needs gcc - see
*gcc: when you need it*), and reports model folders
that are not reachable (for example an unplugged USB drive). The result is written to `data/logs/launch.txt`.

## Rebuilding the engine

`engine\build.bat` regenerates `nvme_engine_u.c` from the patch chain (`make_engine_u.py` + `make_engine_*.py`), compiles it with
gcc (MinGW-w64, see *gcc: when you need it*) and builds the GPU engine with nvcc (`gpu\nvbuild.bat`, needs Visual Studio Build Tools). `engine\gpu\build_gpu_experts.bat`
builds `gpu_experts.dll`.
