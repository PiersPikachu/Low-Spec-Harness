@echo off
rem spec_14b.bat ["prompt"] [max_new] [k]
rem Hermes-4-14B (rans4 vault, CPU) with Qwen3-0.6B drafting on the GPU (huffman4 vault in the D: archive).
rem Greedy output is identical to running Hermes alone; the draft only proposes tokens the 14B then checks in one pass.
setlocal
cd /d "%~dp0"
set P=%~1
if "%P%"=="" set P=In one sentence, what is lossless compression?
set N=%~2
if "%N%"=="" set N=64
set K=%~3
if "%K%"=="" set K=8
set PYTHONIOENCODING=utf-8
python spec.py vault_hermes14b_huff vault_06b_huff --prompt "%P%" --max-new %N% --k %K% --resident-mb 5500
pause
