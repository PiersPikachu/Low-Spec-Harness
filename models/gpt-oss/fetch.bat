@echo off
rem fetch.bat [20b] [120b] - download the compressed gpt-oss vaults from this repository's GitHub release, verify them and put
rem them where the harness installs models (data\vaults unless changed in the Model folders tab). Default: 20b.
rem   fetch.bat            gpt-oss-20b   (~12.3 GB, needs ~15 GB free)
rem   fetch.bat 120b       gpt-oss-120b  (~61.6 GB, needs ~65 GB free; runs at ~2 tok/s with 16 GB RAM and a 6 GB GPU)
rem   fetch.bat 20b 120b   both
rem Interrupted? Run it again: verified parts are kept.
setlocal
cd /d "%~dp0..\.."
where python >nul 2>nul || (echo Python 3.10+ is required & pause & exit /b 1)
python "%~dp0fetch_models.py" %*
pause
