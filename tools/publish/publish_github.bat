@echo off
rem publish_github.bat - publish the harness to https://github.com/PiersPikachu/Low-Spec-Harness
rem   1  code, engine, tools, tests and models\gpt-oss (small files) -> git push to main   (.gitignore keeps your chats, logs,
rem      artifacts, settings, test vaults and data\vaults out)
rem   2  the two gpt-oss vault.bin files (12.3 GB + 61.6 GB) -> 38 parts on the release "gpt-oss-vaults-v1" (GitHub's limit is
rem      2 GiB per release file; releases have no total size or bandwidth limit). Safe to run again: finished parts are skipped.
rem Needs: git (with your GitHub login) and the GitHub CLI (gh), logged in once with "gh auth login".
setlocal
cd /d "%~dp0..\.."
where gh >nul 2>nul || (
  echo The GitHub CLI is not installed. Install it with:   winget install --id GitHub.cli
  echo then open a new terminal, run:   gh auth login   and run this script again.
  pause & exit /b 1
)
gh auth status >nul 2>nul || gh auth login || (pause & exit /b 1)

echo.
echo [1/2] pushing the harness code ...
if not exist .git (
  git init -b main || goto :fail
  git remote add origin https://github.com/PiersPikachu/Low-Spec-Harness.git || goto :fail
)
git fetch origin main || goto :fail
git reset origin/main >nul
git checkout -- LICENSE 2>nul
git add -A || goto :fail
git commit -m "Cube Harness: app, engine, tools, tests, gpt-oss vault manifests" || echo (nothing new to commit)
git push -u origin main || goto :fail

echo.
echo [2/2] uploading the gpt-oss vault parts to the release (74 GB - this takes hours; it can be stopped and restarted) ...
python tools\publish\upload_release.py || goto :fail
echo.
echo DONE
pause
exit /b 0
:fail
echo.
echo FAILED - see the message above. It is safe to run this script again.
pause
exit /b 1
