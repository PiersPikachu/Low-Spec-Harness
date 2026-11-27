@echo off
rem searxng_start.bat - start your SearXNG (normally the web plugin starts it on its own). Stop: searxng_stop.bat
powershell -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','searxng','-u','searxng','--','/usr/local/searxng/run.sh' -WindowStyle Hidden"
powershell -NoProfile -Command "foreach ($i in 1..60) { try { (New-Object Net.Sockets.TcpClient('127.0.0.1',8888)).Close(); Write-Host 'SearXNG is running on http://127.0.0.1:8888'; exit } catch { Start-Sleep 1 } }; Write-Host 'SearXNG did not start - run searxng_setup.bat'"
pause
