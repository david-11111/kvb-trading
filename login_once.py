"""
一次性登录脚本（手动输入账号密码）

用途：
- 打开 KVB WebTrade 页面
- 你在弹出的浏览器里手动完成登录
- 登录态会保存到 smart_trading/browser_data/，后续自动化会复用

注意：本脚本不下单、不做任何交易操作。
"""

import asyncio
import logging
import argparse

from data_fetcher import DataFetcher


async def main():
    parser = argparse.ArgumentParser(description="打开 KVB WebTrade 让你手动登录并保存登录态")
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=900,
        help="保持浏览器打开的秒数（默认 900 秒=15 分钟）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
    )
    logger = logging.getLogger("LoginOnce")

    fetcher = DataFetcher(headless=False)
    ok = await fetcher.connect()
    if not ok:
        logger.error("无法打开交易页面，请检查网络/是否能访问 https://mykvb.com/trade")
        return

    logger.info("浏览器已打开：请在页面中手动完成登录。")
    logger.info(f"本脚本将保持浏览器打开 {args.wait_seconds} 秒；登录完成后可直接关闭浏览器窗口或等待脚本自动结束。")
    logger.info("如需提前结束，在此窗口按 Ctrl+C。")

    try:
        start = asyncio.get_running_loop().time()
        while True:
            elapsed = asyncio.get_running_loop().time() - start
            if elapsed >= args.wait_seconds:
                break

            page = getattr(fetcher, "page", None)
            try:
                if page is None or page.is_closed():
                    logger.info("检测到浏览器窗口已关闭，准备退出…")
                    break
            except Exception:
                logger.info("无法确认页面状态（可能已关闭），准备退出…")
                break

            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，准备退出…")
    finally:
        await fetcher.disconnect()
        logger.info("已关闭浏览器。后续实盘运行会复用 browser_data/ 登录态。")


if __name__ == "__main__":
    asyncio.run(main())
