@echo off
rem cube-harness.bat - starts the Cube Harness from wherever this folder is.
rem   1 finds Python   2 app\launch_check.py adapts the harness to this folder and this machine (runs first-time setup if needed)
rem   3 opens the window (app\harness_app.py) without a console
setlocal
cd /d "%~dp0"
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  echo Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause & exit /b 1
)
%PY% "%~dp0app\launch_check.py"
if errorlevel 2 (
  echo.
  echo First-time setup: installing what is missing ...
  %PY% "%~dp0app\setup_harness.py"
  %PY% "%~dp0app\launch_check.py"
)
if errorlevel 1 (
  echo.
  echo The harness cannot start - see data\logs\launch.txt
  pause & exit /b 1
)
where pythonw >nul 2>nul && (start "" pythonw "%~dp0app\harness_app.py") || (start "" %PY% "%~dp0app\harness_app.py")
exit /b 0
