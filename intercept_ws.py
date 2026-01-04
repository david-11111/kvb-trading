"""
拦截 WebSocket 连接信息
"""

import asyncio
import os
from playwright.async_api import async_playwright


async def intercept_ws():
    print('启动浏览器拦截 WebSocket...')

    playwright = await async_playwright().start()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    user_data_dir = os.path.join(script_dir, 'browser_data')

    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=True,
    )

    page = context.pages[0] if context.pages else await context.new_page()

    ws_info = []

    # 监听 WebSocket 连接
    def on_ws(ws):
        print(f'[WS] 新连接: {ws.url}')
        ws_info.append({
            'url': ws.url,
        })

        # 监听消息
        ws.on('framereceived', lambda payload: print(f'[WS] 收到: {payload[:100]}...'))
        ws.on('framesent', lambda payload: print(f'[WS] 发送: {payload[:100]}...'))

    page.on('websocket', on_ws)

    # 拦截请求获取 token
    tokens = []

    def handle_request(request):
        auth = request.headers.get('authorization', '')
        if auth and auth not in tokens:
            tokens.append(auth)
            print(f'[AUTH] 发现: {auth[:50]}...')

    page.on('request', handle_request)

    print('访问交易页面...')
    try:
        await page.goto('https://mykvb.com/trade', wait_until='load', timeout=60000)
    except Exception as e:
        print(f'页面加载警告: {e}')

    print('等待 WebSocket 连接...')
    await asyncio.sleep(10)

    print('\n' + '='*60)
    print('=== 汇总信息 ===')
    print('='*60)

    print('\n[WebSocket 连接]')
    for info in ws_info:
        print(f'  URL: {info["url"]}')

    print('\n[Authorization Headers]')
    for token in tokens:
        print(f'  {token[:80]}...' if len(token) > 80 else f'  {token}')

    # 从 localStorage 获取更多信息
    print('\n[LocalStorage 认证信息]')
    try:
        storage = await page.evaluate('''() => {
            let result = {};
            for (let i = 0; i < localStorage.length; i++) {
                let key = localStorage.key(i);
                if (key.toLowerCase().includes('token') ||
                    key.toLowerCase().includes('auth') ||
                    key.toLowerCase().includes('user') ||
                    key.toLowerCase().includes('session')) {
                    result[key] = localStorage.getItem(key);
                }
            }
            return result;
        }''')
        for k, v in storage.items():
            v_str = str(v)
            print(f'  {k}: {v_str[:60]}...' if len(v_str) > 60 else f'  {k}: {v_str}')
    except Exception as e:
        print(f'  获取失败: {e}')

    print('\n[Cookies]')
    try:
        cookies = await context.cookies()
        for cookie in cookies:
            if 'token' in cookie['name'].lower() or 'auth' in cookie['name'].lower():
                print(f'  {cookie["name"]}: {cookie["value"][:40]}...')
    except Exception as e:
        print(f'  获取失败: {e}')

    print('='*60)

    await context.close()
    await playwright.stop()


if __name__ == "__main__":
    asyncio.run(intercept_ws())
