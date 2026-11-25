@echo off
rem setup.bat - first-run setup for the Cube Harness: packages, folders, engines, unpack vault archives,
rem move vaults into vaults\, check them, write harness_config.json (relative paths), create "Cube Harness.bat".
rem Run it again any time you add vaults or move the folder.  setup.bat --yes  accepts every default.
setlocal
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python 3.10+ was not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH".
  pause & exit /b 1
)
%PY% "%~dp0setup_harness.py" %*
echo.
pause
