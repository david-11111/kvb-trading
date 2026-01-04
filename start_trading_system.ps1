# KVB 智能交易系统完整启动脚本
# 同时启动：智能交易浏览器 + 后台交易引擎

param(
    [switch]$BrowserOnly,  # 只启动浏览器
    [switch]$Live          # 实盘模式
)

$Host.UI.RawUI.WindowTitle = "KVB 智能交易系统"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  KVB 智能交易系统" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# 切换到脚本目录
Set-Location $PSScriptRoot

# 检查并安装依赖
Write-Host "[1/3] 检查依赖..." -ForegroundColor Yellow

$packages = @("PyQt5", "PyQtWebEngine", "aiohttp", "aiohttp-cors", "playwright")
foreach ($pkg in $packages) {
    $installed = pip show $pkg 2>$null
    if (-not $installed) {
        Write-Host "  安装 $pkg..." -ForegroundColor Gray
        pip install $pkg -q
    }
}

Write-Host "[2/3] 准备启动..." -ForegroundColor Yellow
Write-Host ""

if ($BrowserOnly) {
    Write-Host "模式: 仅浏览器（不启动交易引擎）" -ForegroundColor Magenta
    Write-Host ""
    python trading_browser.py
} else {
    Write-Host "模式: 完整系统（浏览器 + 交易引擎）" -ForegroundColor Green
    if ($Live) {
        Write-Host "交易模式: 实盘" -ForegroundColor Red
    } else {
        Write-Host "交易模式: 模拟" -ForegroundColor Yellow
    }
    Write-Host ""

    # 启动交易引擎（后台）
    Write-Host "[3/3] 启动交易引擎..." -ForegroundColor Yellow

    $tradeArgs = @("-c", "import sys; sys.path.insert(0, '.'); from run_kvb import main; main()")
    if ($Live) {
        # 设置环境变量启用实盘
        $env:KVB_LIVE_TRADE = "1"
    }

    $tradeProcess = Start-Process -FilePath "python" -ArgumentList $tradeArgs -PassThru -WindowStyle Normal

    Write-Host "  交易引擎已启动 (PID: $($tradeProcess.Id))" -ForegroundColor Green
    Write-Host ""

    # 等待一下让交易引擎初始化
    Start-Sleep -Seconds 3

    # 启动浏览器（前台）
    Write-Host "启动智能交易浏览器..." -ForegroundColor Yellow
    python trading_browser.py

    # 浏览器关闭后，询问是否停止交易引擎
    Write-Host ""
    Write-Host "浏览器已关闭。" -ForegroundColor Yellow
    $response = Read-Host "是否停止交易引擎？(Y/N)"
    if ($response -eq "Y" -or $response -eq "y") {
        Stop-Process -Id $tradeProcess.Id -Force -ErrorAction SilentlyContinue
        Write-Host "交易引擎已停止。" -ForegroundColor Green
    } else {
        Write-Host "交易引擎继续在后台运行 (PID: $($tradeProcess.Id))" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "系统已退出。" -ForegroundColor Gray
