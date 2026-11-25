@echo off
rem build helper: nvcc needs MSVC cl.exe on PATH; this machine has VS2019 BuildTools (MSVC 14.29, supported by CUDA 12.0)
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || (echo vcvars64 failed & exit /b 1)
for %%f in (%*) do (
  echo nvcc %%f
  nvcc -arch=sm_86 -O3 -lineinfo -Xcompiler "/O2" %%f -o %%~nf.exe || exit /b 1
)
exit /b 0
