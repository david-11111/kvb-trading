# Live test launcher (SAFE DEFAULTS)
#
# Goal: verify real-web execution (CDP attach, prices/positions reading, close-by-X)
# without allowing the bot to open new positions.
#
# Usage:
#   1) Run `start_chrome_debug.bat` and login, keep the tab open at https://mykvb.com/trade
#   2) Run this script:
#        powershell -ExecutionPolicy Bypass -File .\start_live_test.ps1
#
# Safety:
#   - Disables live OPEN clicks (can’t buy/sell)
#   - Allows live CLOSE clicks (so you can test X close)
#   - Creates trade_data\STOP_TRADING to block any live clicks until you remove it
#
# To allow closing after you confirm everything is ready:
#   - Delete trade_data\STOP_TRADING
#

param(
  [string]$Symbol = "ETHUSD",
  [int]$IntervalSec = 6
)

Set-Location $PSScriptRoot

if (-not $env:SMART_TRADING_CDP_URL) {
  $env:SMART_TRADING_CDP_URL = "http://localhost:9222"
}

# Safety toggles for live mode
$env:SMART_TRADING_ALLOW_LIVE_OPEN = "0"
$env:SMART_TRADING_ALLOW_LIVE_CLOSE = "1"

# Extra kill switch (blocks both open/close when present)
$stopFile = Join-Path $PSScriptRoot "trade_data\STOP_TRADING"
New-Item -ItemType File -Force -Path $stopFile | Out-Null

Write-Host "CDP: $env:SMART_TRADING_CDP_URL" -ForegroundColor Cyan
Write-Host "Mode: LIVE (close-only), AUTO disabled" -ForegroundColor Yellow
Write-Host "Kill switch file created: $stopFile" -ForegroundColor Yellow
Write-Host "Delete STOP_TRADING to allow CLOSE clicks." -ForegroundColor Yellow
Write-Host ""

python .\auto_trader.py --live --no-auto --symbols $Symbol --interval $IntervalSec

