@echo off
rem gpu_check.bat [vault folder] - GPU engine vs CPU engine on one dense vault
rem   (logits, hidden states, greedy text, split mode, speed). Writes gpu_check_log.txt next to this file.
rem   Double-click: checks data\vaults\vault_qwen3-0_6b_huff (fits in VRAM: tests GPU-resident AND split),
rem   else D:\models\vault_q4b_huff (split only). Or: gpu_check.bat D:\models\other_vault (or drag a vault folder onto it)
setlocal
cd /d "%~dp0"
set "LOG=%~dp0gpu_check_log.txt"
set "VAULT=%~1"
if "%VAULT%"=="" if exist "data\vaults\vault_qwen3-0_6b_huff\vault.bin" set "VAULT=%~dp0data\vaults\vault_qwen3-0_6b_huff"
if "%VAULT%"=="" set "VAULT=D:\models\vault_q4b_huff"
echo gpu-check %VAULT%  (%DATE% %TIME%) > "%LOG%"
if not exist "target\release\cube.exe" (
  echo target\release\cube.exe not found - run build.bat first >> "%LOG%"
  goto :done
)
if not exist "%VAULT%\vault.bin" (
  echo no vault at %VAULT% - pass a vault folder: gpu_check.bat D:\models\name >> "%LOG%"
  goto :done
)
echo Checking %VAULT% - takes a few minutes ...
"target\release\cube.exe" gpu-check "%VAULT%" --ctx 512 --tokens 32 >> "%LOG%" 2>&1
echo exit code %ERRORLEVEL% >> "%LOG%"
:done
type "%LOG%"
echo.
echo Log saved to %LOG%
pause
