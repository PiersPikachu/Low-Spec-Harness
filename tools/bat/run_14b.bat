@echo off
rem run_14b.bat [max_new_tokens]  - Hermes-4-14B from the rans4 vault on C:, split mode (first layers on the GPU, rest on the CPU)
setlocal
cd /d "%~dp0"
set N=%~1
if "%N%"=="" set N=32
set PYTHONIOENCODING=utf-8
python run_14b.py split %N% "%~dp0vaults\vault_hermes14b_huff"
pause
