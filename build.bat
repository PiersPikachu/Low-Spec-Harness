@echo off
rem build.bat - build the Rust Cube Harness (cube-harness.exe = the window, cube.exe = command line)
rem   nvcc compiles the GPU kernels to PTX when the VS2019 C++ tools are loaded; without them NVRTC compiles them on first use
setlocal
cd /d "%~dp0"
set VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if exist "%VC%" call "%VC%" >nul
cargo build --release || goto :fail
cargo test --release -p cube-vault -p cube-engine -p cube-core || goto :fail
echo.
echo BUILD OK - start the window with run.bat
exit /b 0
:fail
echo BUILD FAILED
exit /b 1
