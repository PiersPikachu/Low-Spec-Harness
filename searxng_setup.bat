@echo off
rem searxng_setup.bat - your own SearXNG search engine for the web plugin, in a WSL distro named "searxng".
rem   Installs Ubuntu 24.04 into WSL under that name (about 1.5 GB on C:), installs SearXNG in it, listens on
rem   127.0.0.1:8888 (this PC only), tests it from Windows, and points data\plugins\web_config.json at it.
rem   Re-run to update SearXNG. Writes searxng_setup_log.txt. Remove everything: wsl --unregister searxng
setlocal EnableDelayedExpansion
rem wsl.exe reports errors as negative exit codes, which 'if errorlevel 1' does not see: every check is '!errorlevel! neq 0'
cd /d "%~dp0"
set "LOG=%~dp0searxng_setup_log.txt"
echo searxng setup %DATE% %TIME% > "%LOG%"
where wsl >nul 2>&1
if !errorlevel! neq 0 (echo WSL is not available on this PC >> "%LOG%" & goto :done)
wsl -d searxng -u root -- true <nul >nul 2>&1
if !errorlevel! neq 0 (
  echo Installing Ubuntu 24.04 into WSL as "searxng" ^(download, a few minutes^) ...
  echo ===== wsl --install Ubuntu-24.04 --name searxng --no-launch >> "%LOG%"
  wsl --install Ubuntu-24.04 --name searxng --no-launch <nul >> "%LOG%" 2>&1
  if !errorlevel! neq 0 (echo WSL install failed - see above >> "%LOG%" & goto :done)
  wsl -d searxng -u root -- true <nul >> "%LOG%" 2>&1
  if !errorlevel! neq 0 (echo the new searxng distro does not start - see above >> "%LOG%" & goto :done)
)
rem optional: "searxng_contact_url" in data\plugins\web_config.json (a URL or mailto: where Wikimedia / OSM can reach you)
rem enables SearXNG's wikipedia, wikidata and openstreetmap engines
set "CONTACT="
if exist "data\plugins\web_config.json" for /f "usebackq delims=" %%c in (`powershell -NoProfile -Command "(Get-Content 'data\plugins\web_config.json' -Raw | ConvertFrom-Json).searxng_contact_url"`) do set "CONTACT=%%c"
echo Installing SearXNG inside it ^(packages, source, Python environment - several minutes^) ...
echo ===== setup.sh >> "%LOG%"
set "WSLENV=CONTACT"
wsl -d searxng -u root --cd "%~dp0tools\searxng" -- bash -c "tr -d '\r' < setup.sh > /tmp/setup.sh && bash /tmp/setup.sh" <nul >> "%LOG%" 2>&1
if !errorlevel! neq 0 (echo SearXNG setup failed - see above >> "%LOG%" & goto :done)
echo Testing from Windows and configuring the web plugin ...
echo ===== Windows side >> "%LOG%"
rem started the way the web plugin starts it: a hidden wsl.exe that lives as long as SearXNG runs
powershell -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','searxng','-u','searxng','--','/usr/local/searxng/run.sh' -WindowStyle Hidden"
powershell -NoProfile -Command "foreach ($i in 1..60) { try { (New-Object Net.Sockets.TcpClient('127.0.0.1',8888)).Close(); break } catch { Start-Sleep 1 } }; Start-Sleep 2"
set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY where py >nul 2>&1 && set "PY=py -3"
%PY% -X utf8 -c "import json,os,urllib.request as u; r=json.load(u.urlopen('http://127.0.0.1:8888/search?q=rust+programming+language&format=json',timeout=30)); n=len(r.get('results',[])); print('JSON search from Windows: %%d results' %% n); p=os.path.join('data','plugins','web_config.json'); os.makedirs(os.path.dirname(p),exist_ok=True); c=json.load(open(p,encoding='utf-8')) if os.path.exists(p) else {}; c.update(searxng_url='http://127.0.0.1:8888', searxng_wsl='searxng'); json.dump(c,open(p,'w',encoding='utf-8'),indent=1); print('web_config.json:',json.dumps(c)); assert n>0" >> "%LOG%" 2>&1
if !errorlevel! neq 0 (echo Windows cannot reach SearXNG on 127.0.0.1:8888 - see above >> "%LOG%" & goto :done)
echo SEARXNG READY - the web plugin searches through it and starts it when needed >> "%LOG%"
:done
type "%LOG%"
echo.
echo Log saved to %LOG%
pause
