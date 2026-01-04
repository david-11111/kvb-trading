param(
  [string]$Symbols = "USOIL,XAUUSD,ETHUSD",
  [int]$Interval = 5,
  [int]$Duration = 0,
  [switch]$Live,
  [switch]$Paper,
  [switch]$Headless,
  [switch]$Mobile,
  [string]$BrowserExe = "",
  [string]$CdpUrl = "",
  [switch]$Login,
  [switch]$UseQuarkProfile,
  [string]$ProfileDir = "",
  [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Resolve-Python {
  param([string]$Preferred)

  if ($Preferred -and (Get-Command $Preferred -ErrorAction SilentlyContinue)) { return $Preferred }
  if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
  if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
  throw "Python not found. Install Python or add it to PATH."
}

function Default-ProfileDir {
  $quark = Join-Path $env:LOCALAPPDATA "Quark\User Data"
  if (Test-Path $quark) { return $quark }
  return ""
}

if (-not $ProfileDir -and $UseQuarkProfile) { $ProfileDir = Default-ProfileDir }
if ($ProfileDir) {
  $env:SMART_TRADING_PROFILE_DIR = $ProfileDir
} else {
  Remove-Item Env:SMART_TRADING_PROFILE_DIR -ErrorAction SilentlyContinue
}

$resolvedCdpUrl = ""
if ($CdpUrl) {
  $resolvedCdpUrl = $CdpUrl
  $env:SMART_TRADING_CDP_URL = $resolvedCdpUrl
} else {
  Remove-Item Env:SMART_TRADING_CDP_URL -ErrorAction SilentlyContinue
}

$resolvedBrowserExe = ""
if ($BrowserExe) {
  $resolvedBrowserExe = $BrowserExe
  if (Test-Path $resolvedBrowserExe) {
    $env:SMART_TRADING_BROWSER_EXE = $resolvedBrowserExe
  } else {
    Write-Host "  Warning: BrowserExe not found: $resolvedBrowserExe"
    Remove-Item Env:SMART_TRADING_BROWSER_EXE -ErrorAction SilentlyContinue
    $resolvedBrowserExe = ""
  }
} else {
  Remove-Item Env:SMART_TRADING_BROWSER_EXE -ErrorAction SilentlyContinue
}

$py = Resolve-Python -Preferred $Python

Write-Host ""
Write-Host "KVB Auto Trader launcher"
Write-Host "  Repo: $repoRoot"
Write-Host "  Symbols: $Symbols"
Write-Host "  Interval: $Interval sec"
Write-Host "  Duration: $Duration sec (0=run until Ctrl+C)"
Write-Host "  Mode: $(
  if ($Paper) { 'PAPER (no clicks)' }
  elseif ($Live) { 'LIVE (real clicks)' }
  else { 'SAFE (paper by default)' }
)"
if ($env:SMART_TRADING_PROFILE_DIR) {
  Write-Host "  Profile: $env:SMART_TRADING_PROFILE_DIR"
} elseif ($UseQuarkProfile) {
  Write-Host "  Profile: (Quark User Data not found)"
}
if ($resolvedBrowserExe) {
  Write-Host "  BrowserExe: $resolvedBrowserExe"
}
if ($resolvedCdpUrl) {
  Write-Host "  CDP: $resolvedCdpUrl (attach mode)"
}
if ($Mobile) {
  Write-Host "  Mobile: ON (lighter UI)"
  $env:SMART_TRADING_MOBILE = "1"
} else {
  Remove-Item Env:SMART_TRADING_MOBILE -ErrorAction SilentlyContinue
}

if ($UseQuarkProfile -and $env:SMART_TRADING_PROFILE_DIR) {
  try {
    $quarkProc = Get-Process -Name "quark" -ErrorAction SilentlyContinue
    if ($quarkProc) {
      Write-Host "  Note: 检测到 Quark 正在运行，可能占用同一 Profile 导致 Playwright 启动失败；建议先关闭 Quark 再运行。"
    }
  } catch {}
}
Write-Host ""

if ($Login) {
  & $py ".\login_once.py"
  exit $LASTEXITCODE
}

$argsList = @(
  ".\auto_trader.py",
  "--symbols", $Symbols,
  "--interval", "$Interval",
  "--duration", "$Duration"
)

if ($Paper -or (-not $Live)) { $argsList += "--paper" }
if ($Live -and (-not $Paper)) { $argsList += "--live" }
if ($Headless) { $argsList += "--headless" }

& $py @argsList
exit $LASTEXITCODE
