@echo off
rem web_check.bat - live test of the web plugin (search, read, loop protection, unsafe-site blocking, blocklists).
rem   Uses your settings in data\plugins\web_config.json. With your own SearXNG set up, it stops SearXNG first,
rem   so the test also shows the plugin starting it on demand. Writes web_check_log.txt next to this file.
setlocal EnableDelayedExpansion
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
set "CUBE_WEB_CONFIG=%~dp0data\plugins\web_config.json"
if exist "%CUBE_WEB_CONFIG%" (
  echo settings: %CUBE_WEB_CONFIG% >> "%LOG%"
  findstr /c:"searxng_wsl" "%CUBE_WEB_CONFIG%" >nul && (
    echo stopping SearXNG so the test shows the plugin starting it >> "%LOG%"
    wsl --terminate searxng >> "%LOG%" 2>&1
  )
)
echo Testing the web plugin (needs internet; about a minute) ...
%PY% -X utf8 "crates\cube-app\assets\web.py" --selftest >> "%LOG%" 2>&1
echo exit code !errorlevel! >> "%LOG%"
:done
type "%LOG%"
echo.
echo Log saved to %LOG%
pause
