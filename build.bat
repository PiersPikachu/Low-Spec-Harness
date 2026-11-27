@echo off
rem build.bat - build the Rust Cube Harness (cube-harness.exe = the window, cube.exe = command line). Writes build_log.txt.
rem   nvcc compiles the GPU kernels to PTX when the VS2019 C++ tools are loaded; without them NVRTC compiles them on first use
setlocal
cd /d "%~dp0"
set "LOG=%~dp0build_log.txt"
tasklist /fi "imagename eq cube-harness.exe" 2>nul | find /i "cube-harness.exe" >nul
if not errorlevel 1 (
  echo Cube Harness is still open - close every Cube Harness window first ^(Windows cannot replace a running .exe^).
  echo build not started > "%LOG%"
  goto :fail
)
set VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if exist "%VC%" call "%VC%" >nul
echo build %DATE% %TIME% > "%LOG%"
echo Building ...
cargo build --release >> "%LOG%" 2>&1 || goto :fail
echo Testing ...
cargo test --release -p cube-vault -p cube-engine -p cube-core >> "%LOG%" 2>&1 || goto :fail
echo BUILD OK >> "%LOG%"
echo.
echo BUILD OK - start the window with run.bat  (log: %LOG%)
pause
exit /b 0
:fail
echo BUILD FAILED >> "%LOG%"
type "%LOG%"
echo.
echo BUILD FAILED - log: %LOG%
pause
exit /b 1
