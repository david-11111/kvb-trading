@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ================================================================================
echo   KVB 智能交易浏览器 - 正在启动...
echo ================================================================================
echo.
echo   功能：
echo   - 左侧：智能交易镜像面板（持仓监控、一键平仓）
echo   - 中间：内嵌 KVB 网站浏览器
echo   - 右侧：实时交易日志
echo.
echo   提示：请确保 auto_trader 正在运行，否则平仓命令无法执行
echo ================================================================================
echo.

REM 检查PyQt5
python -c "import PyQt5" 2>nul
if errorlevel 1 (
    echo 首次运行，正在安装必要组件...
    echo 这可能需要几分钟，请耐心等待...
    echo.
    pip install PyQt5 PyQtWebEngine
    echo.
    echo 安装完成！
    echo.
)

python trading_browser.py

if errorlevel 1 (
    echo.
    echo 启动失败！
    echo 请尝试手动运行: pip install PyQt5 PyQtWebEngine
    pause
)
