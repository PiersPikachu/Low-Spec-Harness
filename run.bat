@echo off
rem run.bat - start the Cube Harness window (data folder: .\data)
cd /d "%~dp0"
start "" "target\release\cube-harness.exe"
