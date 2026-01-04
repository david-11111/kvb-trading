@echo off
chcp 65001 >nul
title KVB 智能交易浏览器

echo ================================================================================
echo   KVB 智能交易浏览器
echo ================================================================================
echo.

REM 检查并安装依赖
echo [1/2] 检查依赖...
pip show PyQt5 >nul 2>&1
if errorlevel 1 (
    echo 正在安装 PyQt5...
    pip install PyQt5 PyQtWebEngine -q
)

pip show PyQtWebEngine >nul 2>&1
if errorlevel 1 (
    echo 正在安装 PyQtWebEngine...
    pip install PyQtWebEngine -q
)

echo [2/2] 启动浏览器...
echo.

cd /d "%~dp0"
python trading_browser.py

if errorlevel 1 (
    echo.
    echo 启动失败！请检查错误信息。
    pause
)
