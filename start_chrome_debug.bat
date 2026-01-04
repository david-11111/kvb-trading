@echo off
chcp 65001 >nul

rem Proxy is optional. If your local proxy is not running, keeping --proxy-server will break login/page load.
rem Enable by setting SMART_TRADING_PROXY=http://127.0.0.1:7890 (or your proxy URL) before running.
set "PROXY_ARG="
if not "%SMART_TRADING_PROXY%"=="" set "PROXY_ARG=--proxy-server=%SMART_TRADING_PROXY%"

echo 正在以调试模式启动Chrome...
if not "%PROXY_ARG%"=="" (
  echo 已启用代理: %SMART_TRADING_PROXY%
) else (
  echo 未启用代理（推荐先这样登录成功后再考虑代理）
)
echo.
echo 请在打开的Chrome中:
echo 1. 等待 mykvb.com/trade 加载完成
echo 2. 登录您的账户（如果需要）
echo 3. 确认看到交易界面和持仓列表
echo.
echo 然后运行 start_live.bat 启动自动交易
echo.
start "" "C:\Users\Lenovo\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe" --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1 --user-data-dir="%TEMP%\chrome_debug_profile" %PROXY_ARG% https://mykvb.com/trade
pause
