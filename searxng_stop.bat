@echo off
rem searxng_stop.bat - stop SearXNG and its WSL distro (frees its memory)
wsl --terminate searxng
echo SearXNG stopped.
pause
