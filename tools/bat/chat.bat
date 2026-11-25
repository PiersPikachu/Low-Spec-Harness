@echo off
rem chat.bat <vault> [options]   talk to a model in the harness from its compressed vault
rem   chat.bat vault_hermes14b_huff
rem   chat.bat vault_06b_rans --greedy
rem   chat.bat vault_llama1b_rans          (base model: plain text continuation)
rem options: --mode auto^|cpu^|gpu^|split  --gpu-layers N  --ctx 2048  --max-new 512  --temp 0.7 --top-p 0.8 --top-k 20  --greedy  --seed N
rem          --system "text"  --think (let Qwen3-style models think before answering)
rem          --draft auto^|none^|<vault>  --k 8   speculative decoding: big streamed models get a small same-vocabulary draft on the GPU
rem                                  (greedy, identical replies, ~2.7x faster on Hermes-4-14B); --draft none --temp 0.7 to sample
rem in chat: /reset  /stats  /exit
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo usage: chat.bat ^<vault^> [options]
  echo vaults here:
  for /d %%v in (vault_*) do echo   %%v
  exit /b 2
)
set PYTHONIOENCODING=utf-8
python chat.py %*
