"""
启动 Chromium 浏览器并开启 CDP 远程调试
使用方法:
1. 运行此脚本: python start_cdp.py
2. 在打开的浏览器中手动登录 KVB
3. 然后运行: run_kvb.ps1 -Live -CdpUrl "http://127.0.0.1:9222"
"""
import subprocess
import sys
import time
from pathlib import Path

def main():
    # 找到 Playwright 安装的 Chromium
    try:
        from playwright._impl._driver import compute_driver_executable
        driver_path = compute_driver_executable()
        browsers_path = Path(driver_path).parent.parent / "chromium-1169"

        # 尝试多个可能的路径
        possible_paths = [
            browsers_path / "chrome-win" / "chrome.exe",
            browsers_path / "chrome.exe",
            Path.home() / "AppData" / "Local" / "ms-playwright" / "chromium-1169" / "chrome-win" / "chrome.exe",
        ]

        chrome_exe = None
        for p in possible_paths:
            if p.exists():
                chrome_exe = p
                break

        if not chrome_exe:
            # 使用 playwright 命令获取路径
            import playwright
            pw_path = Path(playwright.__file__).parent
            # 搜索 chromium
            for p in pw_path.rglob("chrome.exe"):
                chrome_exe = p
                break
    except:
        chrome_exe = None

    if not chrome_exe:
        print("无法找到 Playwright 的 Chromium，尝试使用系统浏览器...")
        # 尝试系统 Chrome
        possible_system = [
            Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
            Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        ]
        for p in possible_system:
            if p.exists():
                chrome_exe = p
                break

    if not chrome_exe:
        print("错误: 找不到任何可用的浏览器")
        print("请手动启动 Chrome/Edge 并开启远程调试:")
        print('  chrome.exe --remote-debugging-port=9222 --user-data-dir="%TEMP%\\chrome_cdp"')
        return 1

    print(f"使用浏览器: {chrome_exe}")

    # 用户数据目录（放在项目根目录下，便于统一管理；已被 .gitignore 忽略）
    repo_root = Path(__file__).resolve().parent.parent
    user_data_dir = repo_root / "cdp_profile"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # 启动参数
    args = [
        str(chrome_exe),
        f"--user-data-dir={user_data_dir}",
        "--remote-debugging-port=9222",
        "--remote-debugging-address=127.0.0.1",
        "--no-first-run",
        "--disable-session-crashed-bubble",
        "https://mykvb.com/trade"
    ]

    print(f"启动浏览器...")
    print(f"  用户数据: {user_data_dir}")
    print(f"  CDP 端口: 9222")
    print()

    # 启动浏览器
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 等待 CDP 就绪
    import urllib.request
    cdp_url = "http://127.0.0.1:9222/json/version"
    for _ in range(20):
        try:
            urllib.request.urlopen(cdp_url, timeout=1)
            print("CDP 已就绪!")
            print()
            print("=" * 60)
            print("现在请在浏览器中登录 KVB，然后运行:")
            print()
            print('  .\\scripts\\run_kvb.ps1 -Live -CdpUrl "http://127.0.0.1:9222"')
            print("=" * 60)
            return 0
        except:
            time.sleep(0.5)

    print("警告: CDP 可能未就绪，请检查浏览器是否启动")
    return 1

if __name__ == "__main__":
    sys.exit(main())
