@echo off
rem verify_rans4.bat - full check of the rans4 harness
rem  1 codec + packer + engine selftests   2 synthetic models: CPU, GPU and split engines vs numpy reference
rem  3 real models (Qwen3-0.6B, Llama-3.2-1B): CPU, GPU, split vs numpy   4 GEN parity GPU vs CPU   5 Hugging Face fp32, token for token
setlocal
cd /d "%~dp0"
set Q=C:\Users\1fred\Desktop\AI\JEPA\_stage_memory\hf_model\Qwen_Qwen3-0.6B
set L=D:\models\cube-harness_archive_20261121\models\Llama-3.2-1B
set PACK_CODEC=rans4

echo ===== 1/5 selftests =====
python rans4.py selftest || goto :fail
python pack_universal_v3.py selftest | findstr /C:"SELFTEST" /C:"FAIL"
nvme_engine_u.exe selftest | findstr /C:"SELFTEST"

echo ===== 2/5 synthetic: CPU / GPU / split =====
python verify_engine.py synth || goto :fail
set ENGINE_EXE=%~dp0gpu\gpu_engine.exe
python verify_engine.py synth || goto :fail
set ENGINE_EXE=
set ENGINE_EXTRA_ARGS=--gpu-layers 2
python verify_engine.py synth || goto :fail
set ENGINE_EXTRA_ARGS=

echo ===== 3/5 real models: CPU / GPU / split =====
python verify_engine.py real "%Q%" vaults\vault_06b_rans || goto :fail
python verify_engine.py real "%L%" vaults\vault_llama1b_rans || goto :fail
set ENGINE_EXE=%~dp0gpu\gpu_engine.exe
python verify_engine.py real "%Q%" vaults\vault_06b_rans || goto :fail
python verify_engine.py real "%L%" vaults\vault_llama1b_rans || goto :fail
set ENGINE_EXE=
set ENGINE_EXTRA_ARGS=--gpu-layers 14
python verify_engine.py real "%Q%" vaults\vault_06b_rans || goto :fail
set ENGINE_EXTRA_ARGS=--gpu-layers 8
python verify_engine.py real "%L%" vaults\vault_llama1b_rans || goto :fail
set ENGINE_EXTRA_ARGS=

echo ===== 4/5 GEN parity: GPU engine vs CPU engine =====
python gen_compare.py vaults\vault_06b_rans 151936 || goto :fail

echo ===== 5/5 Hugging Face fp32, token for token (takes ~10 min) =====
set PYTHONIOENCODING=utf-8
python hf_compare.py 0.6b-rans 1b-rans || goto :fail
echo ALL PASSED
pause
exit /b 0
:fail
echo FAILED - send me the output above
pause
exit /b 1
