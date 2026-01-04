@echo off
setlocal
cd /d "%~dp0.."

REM One-click launcher (default: PAPER mode, no real clicks)
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_kvb.ps1" %*

if errorlevel 1 (
  echo.
  echo Launcher failed. Press any key to close...
  pause >nul
)

endlocal
