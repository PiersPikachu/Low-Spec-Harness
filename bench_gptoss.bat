@echo off
rem bench_gptoss.bat [vault folder] - gpt-oss: VRAM expert tier check + speed, CPU-only vs RAM+VRAM expert tiers.
rem   Writes bench_gptoss_log.txt next to this file. Double-click: data\vaults\vault_gpt-oss-20b_mxf2.
setlocal
cd /d "%~dp0"
set "LOG=%~dp0bench_gptoss_log.txt"
set "VAULT=%~1"
if "%VAULT%"=="" set "VAULT=%~dp0data\vaults\vault_gpt-oss-20b_mxf2"
set "THREADS=16"
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
echo 1/3 correctness: experts computed on the GPU vs the CPU engine ...
echo. >> "%LOG%"
echo ===== 1/3 gpu-check (VRAM expert tier vs CPU) >> "%LOG%"
"target\release\cube.exe" gpu-check "%VAULT%" --ctx 512 --tokens 16 >> "%LOG%" 2>&1
echo exit code %ERRORLEVEL% >> "%LOG%"
echo 2/3 speed, CPU only (RAM expert cache) ...
echo. >> "%LOG%"
echo ===== 2/3 bench CPU only >> "%LOG%"
"target\release\cube.exe" bench "%VAULT%" --threads %THREADS% --resident-mb auto --prompt 8 --tokens 64 2>&1 | findstr /v /b /c:"T " >> "%LOG%"
echo 3/3 speed, RAM + VRAM expert tiers ...
echo. >> "%LOG%"
echo ===== 3/3 bench RAM + VRAM expert tiers >> "%LOG%"
"target\release\cube.exe" bench "%VAULT%" --threads %THREADS% --resident-mb auto --prompt 8 --tokens 64 --gpu 2>&1 | findstr /v /b /c:"T " >> "%LOG%"
:done
type "%LOG%"
echo.
echo Log saved to %LOG%
pause
