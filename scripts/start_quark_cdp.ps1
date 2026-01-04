param(
  [int]$Port = 9222,
  [string]$QuarkExe = "D:\\Users\\Lenovo\\AppData\\Local\\Programs\\Quark\\quark.exe",
  [string]$ProfileDir = "",
  [string]$Url = "https://mykvb.com/trade",
  [int]$WaitSec = 12,
  [switch]$KillExisting
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path $QuarkExe)) {
  throw "QuarkExe not found: $QuarkExe"
}

if (-not $ProfileDir) {
  $ProfileDir = Join-Path $repoRoot "quark_cdp_profile"
}

New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null

Write-Host ""
Write-Host "Starting Quark with CDP..."
Write-Host "  QuarkExe: $QuarkExe"
Write-Host "  ProfileDir: $ProfileDir"
Write-Host "  Port: $Port"
Write-Host "  Url: $Url"
Write-Host ""
Write-Host "If you already have Quark running, close ALL Quark windows first (Task Manager: quark.exe)."
if ($KillExisting) { Write-Host "KillExisting: ON (will force close all quark.exe)" }
Write-Host ""

if ($KillExisting) {
  try {
    Get-Process -Name "quark" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 800
  } catch {}
}

$args = @(
  "--user-data-dir=$ProfileDir",
  "--remote-debugging-address=127.0.0.1",
  "--remote-debugging-port=$Port",
  "--remote-allow-origins=*",
  "--new-window",
  $Url
)

Start-Process -FilePath $QuarkExe -ArgumentList $args | Out-Null

$cdp = "http://127.0.0.1:$Port/json/version"
$deadline = (Get-Date).AddSeconds($WaitSec)
$ok = $false
while ((Get-Date) -lt $deadline) {
  try {
    $resp = irm $cdp -TimeoutSec 1
    if ($resp) { $ok = $true; break }
  } catch {}
  Start-Sleep -Milliseconds 500
}

if (-not $ok) {
  Write-Host ""
  Write-Host "CDP not reachable at $cdp"
  Write-Host "This usually means Quark did NOT start with remote-debugging (flags ignored) or the port is blocked."
  Write-Host ""
  Write-Host "Fallback (no CDP attach):"
  Write-Host "  cd `"$repoRoot`""
  Write-Host "  .\\scripts\\run_kvb.ps1 -Live -BrowserExe `"$QuarkExe`" -ProfileDir `".\\browser_data`""
  exit 2
}

Write-Host ""
Write-Host "CDP OK: $cdp"
Write-Host "Now start the bot (attach mode):"
Write-Host "  cd `"$repoRoot`""
Write-Host "  .\\scripts\\run_kvb.ps1 -Live -CdpUrl `\"http://127.0.0.1:$Port`\""
