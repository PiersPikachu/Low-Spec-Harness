@echo off
rem build.bat - rebuild both engines from source
rem   nvme_engine_u.exe : CPU engine (huffman4 + rans4 vaults, split mode). Derived from JEPA\_stage_memory\nvme_engine.c by make_engine_u.py
rem   gpu\gpu_engine.exe: GPU engine / split-mode worker (huffman4 + rans4 vaults)
rem   gcc = the GNU C compiler (MinGW-w64 on Windows). Only this rebuild needs it - running the harness does not.
rem   Install: winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT   (README.md: "gcc: when you need it")
setlocal
cd /d "%~dp0"
where gcc >nul 2>&1 || (
  echo gcc was not found on PATH. It is needed to compile nvme_engine_u.exe from nvme_engine_u.c.
  echo Install MinGW-w64, e.g.: winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT
  echo then open a new terminal and run this again. See README.md, "gcc: when you need it".
  goto :fail
)
python make_engine_u.py || goto :fail
gcc -O3 -march=native -ffp-contract=off -fopenmp -static nvme_engine_u.c -o nvme_engine_u.exe -lm -lpsapi || goto :fail
nvme_engine_u.exe selftest | findstr /C:"SELFTEST"
call gpu\nvbuild.bat gpu_engine.cu || goto :fail
echo BUILD OK
exit /b 0
:fail
echo BUILD FAILED
exit /b 1
