@echo off
rem pack_rans4.bat <hf_model_dir> <vault_dir> [--delete-consumed]
rem Packs any HF safetensors model (bf16, dense decoder) into a rans4 vault, verifies every matrix bit-exact while packing,
rem re-checks the whole vault against the source, then prints the execution plan for this machine.
rem --delete-consumed removes each source shard once all its matrices are packed and verified (for disks that cannot hold both).
setlocal
cd /d "%~dp0"
if "%~2"=="" (echo usage: pack_rans4.bat ^<hf_model_dir^> ^<vault_dir^> [--delete-consumed] & exit /b 2)
python pack_universal_v3.py pack "%~1" "%~2" --codec rans4 %3 || goto :fail
if /I not "%~3"=="--delete-consumed" (python pack_universal_v3.py check "%~1" "%~2" || goto :fail)
nvme_engine_u.exe check "%~2" --log "%~2\check_log.txt" || goto :fail
python planner.py plan "%~2"
echo PACK OK
pause
exit /b 0
:fail
echo PACK FAILED - send me the output above
pause
exit /b 1
