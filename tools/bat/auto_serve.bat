@echo off
rem auto_serve.bat <vault_dir> [engine options]  - measures the machine, picks the execution mode, prints which one, launches the best engine that exists
cd /d "%~dp0"
python planner.py serve %*
