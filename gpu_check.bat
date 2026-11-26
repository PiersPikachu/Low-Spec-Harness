@echo off
rem gpu_check.bat ^<vault folder^> - GPU engine vs CPU engine on one dense vault (logits, hidden states, greedy text, split mode, speed)
rem   example: gpu_check.bat D:\models\vault_qwen3_0_6b_huff
cd /d "%~dp0"
if "%~1"=="" (echo usage: gpu_check.bat ^<vault folder^> & exit /b 2)
target\release\cube.exe gpu-check "%~1" --ctx 512 --tokens 32 > gpu_check_log.txt 2>&1
type gpu_check_log.txt
