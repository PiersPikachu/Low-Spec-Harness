@echo off
rem sandbox_setup.bat - the WSL sandbox "cube-sandbox" for running artifact code cut off from Windows.
rem   Installs Ubuntu 24.04 into WSL under that name (about 2.5 GB on C: with the toolchains), installs Python, C/C++,
rem   Node.js, Go, Java, Rust and Graphviz in it, and isolates it: no Windows drives mounted, no starting Windows programs,
rem   code runs as an unprivileged user. Then checks the isolation. Re-run to update. Writes sandbox_setup_log.txt.
rem   Use it: Artifacts tab > "in the sandbox", or Config > "Run artifact code in the WSL sandbox".
rem   Remove it: wsl --unregister cube-sandbox
setlocal EnableDelayedExpansion
rem wsl.exe reports errors as negative exit codes: every check is '!errorlevel! neq 0'
cd /d "%~dp0"
set "LOG=%~dp0sandbox_setup_log.txt"
echo sandbox setup %DATE% %TIME% > "%LOG%"
where wsl >nul 2>&1
if !errorlevel! neq 0 (echo WSL is not available on this PC >> "%LOG%" & goto :done)
wsl -d cube-sandbox -u root --exec true <nul >nul 2>&1
if !errorlevel! neq 0 (
  echo Installing Ubuntu 24.04 into WSL as "cube-sandbox" ^(download, a few minutes^) ...
  echo ===== wsl --install Ubuntu-24.04 --name cube-sandbox --no-launch >> "%LOG%"
  wsl --install Ubuntu-24.04 --name cube-sandbox --no-launch <nul >> "%LOG%" 2>&1
  if !errorlevel! neq 0 (echo WSL install failed - see above >> "%LOG%" & goto :done)
  wsl -d cube-sandbox -u root --exec true <nul >> "%LOG%" 2>&1
  if !errorlevel! neq 0 (echo the new cube-sandbox distro does not start - see above >> "%LOG%" & goto :done)
)
echo Installing the toolchains inside it ^(several minutes^) ...
echo ===== setup.sh >> "%LOG%"
rem the script goes in through stdin: works also once Windows drives are no longer mounted in there
wsl -d cube-sandbox -u root --exec bash -c "tr -d '\r' > /tmp/sandbox_setup.sh && bash /tmp/sandbox_setup.sh" < "tools\sandbox\setup.sh" >> "%LOG%" 2>&1
if !errorlevel! neq 0 (echo sandbox setup failed - see above >> "%LOG%" & goto :done)
echo Restarting it with the isolation settings and checking them ...
wsl --terminate cube-sandbox >> "%LOG%" 2>&1
echo ===== isolation check >> "%LOG%"
set "LEAK="
for /f "usebackq delims=" %%L in (`wsl -d cube-sandbox -u runner --exec sh -c "ls -d /mnt/c /mnt/d 2>/dev/null; command -v cmd.exe powershell.exe 2>/dev/null; true" ^<nul`) do (
  echo LEAK: %%L >> "%LOG%"
  set "LEAK=1"
)
wsl -d cube-sandbox -u runner --exec sh -c "echo user: $(id -un); echo windows drives: $(ls -d /mnt/[a-z] 2>/dev/null | tr '\n' ' ')none; python3 --version" <nul >> "%LOG%" 2>&1
if defined LEAK (echo NOT ISOLATED - Windows drives or programs are still reachable from the sandbox, see above >> "%LOG%" & goto :done)
echo SANDBOX READY - isolated: no Windows drives, no Windows programs, user runner >> "%LOG%"
:done
type "%LOG%"
echo.
echo Log saved to %LOG%
pause
