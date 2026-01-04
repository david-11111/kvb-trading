"""
紧急平仓脚本 - 平掉平台上所有持仓
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import DataFetcher

async def close_all():
    fetcher = DataFetcher()

    print("正在连接到交易平台...")
    connected = await fetcher.connect()
    if not connected:
        print("连接失败！")
        return

    print("连接成功，尝试平掉所有持仓...")

    # 使用 close_all_positions 方法
    closed = await fetcher.close_all_positions()
    print(f"通过平仓按钮平掉 {closed} 个仓位")

    if closed == 0:
        print("\n未找到平仓按钮，请手动在平台上平仓！")
        print("请在KVB平台上找到'持仓'或'订单'面板，逐个点击平仓按钮。")

    await fetcher.disconnect()

if __name__ == "__main__":
    asyncio.run(close_all())
