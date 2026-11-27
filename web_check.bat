@echo off
rem web_check.bat - live test of the web plugin (search, read, loop protection, unsafe-site blocking, blocklists).
rem   Writes web_check_log.txt next to this file. Needs Python (the same one the harness uses for plugins).
setlocal
cd /d "%~dp0"
set "LOG=%~dp0web_check_log.txt"
set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY where py >nul 2>&1 && set "PY=py -3"
echo web check %DATE% %TIME% > "%LOG%"
if not defined PY (
  echo Python not found - install Python 3 and tick "Add to PATH" >> "%LOG%"
  goto :done
)
echo Testing the web plugin (needs internet; about a minute) ...
%PY% -X utf8 "crates\cube-app\assets\web.py" --selftest >> "%LOG%" 2>&1
echo exit code %ERRORLEVEL% >> "%LOG%"
:done
type "%LOG%"
echo.
echo Log saved to %LOG%
pause
