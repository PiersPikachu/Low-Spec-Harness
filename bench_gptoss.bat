@echo off
rem bench_gptoss.bat [vault folder] - gpt-oss speed on real chat replies, as the app runs them:
rem   CPU only (RAM expert cache) vs RAM + VRAM expert tiers. 6 prompts x 2 rounds x 96 tokens per mode (~20 min total).
rem   Writes bench_gptoss_log.txt next to this file. Double-click: data\vaults\vault_gpt-oss-20b_mxf2.
setlocal
cd /d "%~dp0"
set "LOG=%~dp0bench_gptoss_log.txt"
set "VAULT=%~1"
if "%VAULT%"=="" set "VAULT=%~dp0data\vaults\vault_gpt-oss-20b_mxf2"
set "ARGS=--rounds 2 --max-new 96 --threads 16"
echo bench_gptoss %VAULT%  (%DATE% %TIME%) > "%LOG%"
echo Building (picks up source changes; quick when nothing changed) ...
cargo build --release -p cube-cli >> "%LOG%" 2>&1
if errorlevel 1 (
  echo build failed - see above >> "%LOG%"
  goto :done
)
if not exist "%VAULT%\vault.bin" (
  echo no vault at %VAULT% - pass a vault folder: bench_gptoss.bat D:\models\name >> "%LOG%"
  goto :done
)
echo 1/2 CPU only (RAM expert cache) - about 10 minutes ...
echo. >> "%LOG%"
echo ===== 1/2 chat-bench, CPU only >> "%LOG%"
"target\release\cube.exe" chat-bench "%VAULT%" --mode cpu %ARGS% >> "%LOG%" 2>&1
echo exit code %ERRORLEVEL% >> "%LOG%"
echo 2/2 RAM + VRAM expert tiers - about 10 minutes ...
echo. >> "%LOG%"
echo ===== 2/2 chat-bench, RAM + VRAM expert tiers >> "%LOG%"
"target\release\cube.exe" chat-bench "%VAULT%" --mode auto %ARGS% >> "%LOG%" 2>&1
echo exit code %ERRORLEVEL% >> "%LOG%"
:done
type "%LOG%"
echo.
echo Log saved to %LOG%
pause
