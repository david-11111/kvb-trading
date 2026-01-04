param(
  [int]$Port = 9222,
  [string]$ProfileDir = "",
  [string]$Url = "https://mykvb.com/trade",
  [int]$WaitSec = 12
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Resolve-EdgeExe {
  $candidates = @(
    "$env:ProgramFiles\\Microsoft\\Edge\\Application\\msedge.exe",
    "$env:ProgramFiles(x86)\\Microsoft\\Edge\\Application\\msedge.exe"
  )
  foreach ($p in $candidates) { if ($p -and (Test-Path $p)) { return $p } }
  return ""
}

$edge = Resolve-EdgeExe
if (-not $edge) { throw "msedge.exe not found. Install Microsoft Edge or provide a different browser." }

if (-not $ProfileDir) {
  $ProfileDir = Join-Path $repoRoot "edge_cdp_profile"
}
New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null

$args = @(
  "--user-data-dir=$ProfileDir",
  "--remote-debugging-address=127.0.0.1",
  "--remote-debugging-port=$Port",
  "--new-window",
  $Url
)

Write-Host ""
Write-Host "Starting Edge with CDP..."
Write-Host "  EdgeExe: $edge"
Write-Host "  ProfileDir: $ProfileDir"
Write-Host "  Port: $Port"
Write-Host "  Url: $Url"
Write-Host ""

Start-Process -FilePath $edge -ArgumentList $args | Out-Null

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
  Write-Host "Try a different port: .\\start_edge_cdp.ps1 -Port 9223"
  exit 2
}

Write-Host ""
Write-Host "CDP OK: $cdp"
Write-Host "Now start the bot (attach mode):"
Write-Host "  cd `"$repoRoot`""
Write-Host "  .\\scripts\\run_kvb.ps1 -Live -CdpUrl `\"http://127.0.0.1:$Port`\""
