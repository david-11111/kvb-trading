"""
切换交易品种脚本
"""

import asyncio
import logging
import argparse

from data_fetcher import DataFetcher


async def main():
    parser = argparse.ArgumentParser(description="切换交易品种")
    parser.add_argument(
        "symbol",
        type=str,
        help="要切换到的品种代码，如 XAUUSD, USOIL, ETHUSD",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
        help="切换后保持浏览器打开的秒数（默认 120 秒）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
    )
    logger = logging.getLogger("SwitchSymbol")

    fetcher = DataFetcher(headless=False)
    ok = await fetcher.connect()
    if not ok:
        logger.error("无法打开交易页面")
        return

    # 等待页面完全加载
    logger.info("等待页面加载...")
    await asyncio.sleep(3)

    # 获取当前品种
    current = await fetcher.get_symbol()
    logger.info(f"当前品种: {current}")

    # 切换品种
    target_symbol = args.symbol.upper()
    logger.info(f"正在切换到: {target_symbol}")

    # 尝试多种方式切换
    page = fetcher.page
    success = False

    # 方法1: 直接点击自选列表中的品种
    try:
        item = await page.query_selector(f'.custom-list-item[data-item="{target_symbol}"]')
        if item:
            await item.click()
            await asyncio.sleep(1)
            success = True
            logger.info(f"在自选列表中找到并点击了 {target_symbol}")
    except Exception as e:
        logger.debug(f"方法1失败: {e}")

    # 方法2: 尝试通过搜索框添加品种
    if not success:
        try:
            # 查找添加品种的按钮（通常是一个 + 号）
            add_btn = await page.query_selector('.add-symbol-btn, button:has-text("+"), .custom-list-add')
            if add_btn:
                await add_btn.click()
                await asyncio.sleep(0.5)

            # 查找搜索输入框
            search_selectors = [
                'input[placeholder*="搜索"]',
                'input[placeholder*="Search"]',
                '.symbol-search input',
                '.search-input input',
                'input.search',
            ]
            for sel in search_selectors:
                search_input = await page.query_selector(sel)
                if search_input:
                    await search_input.fill(target_symbol)
                    await asyncio.sleep(1)
                    # 点击搜索结果
                    result = await page.query_selector(f'text="{target_symbol}"')
                    if result:
                        await result.click()
                        await asyncio.sleep(1)
                        success = True
                        logger.info(f"通过搜索添加了 {target_symbol}")
                    break
        except Exception as e:
            logger.debug(f"方法2失败: {e}")

    # 方法3: 直接在页面上点击包含品种名称的元素
    if not success:
        try:
            elem = await page.query_selector(f'text="{target_symbol}"')
            if elem:
                await elem.click()
                await asyncio.sleep(1)
                success = True
                logger.info(f"通过文本匹配点击了 {target_symbol}")
        except Exception as e:
            logger.debug(f"方法3失败: {e}")

    # 获取当前品种确认是否成功
    await asyncio.sleep(1)
    new_symbol = await fetcher.get_symbol()

    if success or (new_symbol and target_symbol in new_symbol.upper()):
        logger.info(f"切换成功！当前品种: {new_symbol}")
    else:
        logger.warning(f"自动切换失败，请手动操作：")
        logger.info(f"1. 在左侧品种列表中点击 '+' 添加品种")
        logger.info(f"2. 搜索 '{target_symbol}' 并点击添加")
        logger.info(f"3. 然后在列表中点击 '{target_symbol}' 切换")
        logger.info(f"浏览器将保持打开 {args.wait_seconds} 秒...")

    # 等待一段时间让用户确认或手动操作
    try:
        start = asyncio.get_running_loop().time()
        while True:
            elapsed = asyncio.get_running_loop().time() - start
            if elapsed >= args.wait_seconds:
                break
            try:
                if page.is_closed():
                    logger.info("检测到浏览器已关闭")
                    break
            except:
                break
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，准备退出...")
    finally:
        await fetcher.disconnect()
        logger.info("已关闭浏览器")


if __name__ == "__main__":
    asyncio.run(main())
