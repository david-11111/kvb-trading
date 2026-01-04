@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动实盘模式...
echo 提示：请先在环境变量中设置账号密码（不要写进脚本）：
echo   set SMART_TRADING_PHONE=你的手机号
echo   set SMART_TRADING_PASSWORD=你的密码
python auto_trader.py --live
pause
