@echo off
rem builds gpu_experts.dll (GPU compute for VRAM-resident MXFP4 experts) next to nvme_engine_u.exe
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || (echo vcvars64 failed & exit /b 1)
nvcc -arch=sm_86 -O3 -shared -cudart shared -Xcompiler "/O2 /MD" gpu_experts.cu -o ..\gpu_experts.dll || exit /b 1
echo built ..\gpu_experts.dll
exit /b 0
