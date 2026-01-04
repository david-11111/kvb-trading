"""
多品种波动监控 - 启动脚本

监控 USOIL(原油)、XAUUSD(黄金)、ETHUSD(以太坊) 的波动率
发现高波动交易机会
"""

import asyncio
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List
import numpy as np

from data_fetcher import DataFetcher


class VolatilityMonitor:
    """波动率监控器"""

    def __init__(self, symbols: List[str], interval: float = 5.0):
        self.symbols = symbols
        self.interval = interval

        self.fetcher = DataFetcher(headless=False)

        # 价格历史
        self.price_history: Dict[str, List[float]] = {s: [] for s in symbols}
        self.max_history = 60  # 保存60条数据

        # 最新价格
        self.latest_prices: Dict[str, dict] = {}

        self._is_running = False
        self.logger = logging.getLogger("VolatilityMonitor")

    async def start(self) -> bool:
        """启动监控"""
        self.logger.info("正在启动波动监控...")

        ok = await self.fetcher.connect()
        if not ok:
            self.logger.error("无法连接到交易页面")
            return False

        await asyncio.sleep(3)
        self._is_running = True
        self.logger.info(f"波动监控已启动! 监控品种: {self.symbols}")
        return True

    async def stop(self):
        """停止监控"""
        self._is_running = False
        await self.fetcher.disconnect()
        self.logger.info("波动监控已停止")

    async def fetch_symbol_prices(self) -> Dict[str, dict]:
        """获取所有品种的价格"""
        page = self.fetcher.page
        if not page:
            return {}

        result = {}

        for symbol in self.symbols:
            try:
                # 从自选列表获取价格
                item_selector = f'.custom-list-item[data-item="{symbol}"]'
                item = await page.query_selector(item_selector)

                if not item:
                    continue

                # 获取买卖价
                bid_elem = await item.query_selector('.item-bid span')
                ask_elem = await item.query_selector('.item-ask span')

                bid = 0.0
                ask = 0.0

                if bid_elem:
                    text = await bid_elem.inner_text()
                    bid = float(text.strip())

                if ask_elem:
                    text = await ask_elem.inner_text()
                    ask = float(text.strip())

                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    result[symbol] = {
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'spread': ask - bid,
                        'timestamp': time.time()
                    }

                    # 更新历史
                    self.price_history[symbol].append(mid)
                    if len(self.price_history[symbol]) > self.max_history:
                        self.price_history[symbol].pop(0)

            except Exception as e:
                self.logger.debug(f"获取 {symbol} 失败: {e}")

        self.latest_prices = result
        return result

    def calc_volatility(self, symbol: str) -> float:
        """计算波动率(%)"""
        history = self.price_history.get(symbol, [])
        if len(history) < 2:
            return 0.0

        prices = np.array(history)
        returns = np.diff(prices) / prices[:-1] * 100
        return float(np.std(returns))

    def calc_price_change(self, symbol: str) -> float:
        """计算价格变动(%)"""
        history = self.price_history.get(symbol, [])
        if len(history) < 2:
            return 0.0

        return (history[-1] - history[0]) / history[0] * 100

    def get_ranking(self) -> List[dict]:
        """获取波动率排名"""
        ranking = []

        for symbol in self.symbols:
            price_info = self.latest_prices.get(symbol, {})
            if not price_info:
                continue

            ranking.append({
                'symbol': symbol,
                'bid': price_info.get('bid', 0),
                'ask': price_info.get('ask', 0),
                'spread': price_info.get('spread', 0),
                'volatility': self.calc_volatility(symbol),
                'change': self.calc_price_change(symbol),
                'history_count': len(self.price_history.get(symbol, [])),
            })

        # 按波动率排序
        ranking.sort(key=lambda x: x['volatility'], reverse=True)
        return ranking

    def print_status(self):
        """打印状态"""
        ranking = self.get_ranking()

        # 清屏效果
        print("\033[2J\033[H", end="")

        print("="*70)
        print(f"  多品种波动监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        if not ranking:
            print("\n  等待数据... 请确保以下品种在自选列表中:")
            for s in self.symbols:
                print(f"    - {s}")
            print("\n  提示: 在 KVB 页面左侧点击 '+' 添加品种")
        else:
            print(f"\n  {'品种':<10} {'买价':<12} {'卖价':<12} {'波动率%':<10} {'涨跌%':<10} {'数据量':<8}")
            print("-"*70)

            for item in ranking:
                symbol = item['symbol']
                bid = f"{item['bid']:.4f}"
                ask = f"{item['ask']:.4f}"
                vol = f"{item['volatility']:.4f}"
                change = f"{item['change']:+.3f}"
                count = f"{item['history_count']}"

                # 高波动标记
                flag = " <<<" if item['volatility'] > 0.1 else ""

                print(f"  {symbol:<10} {bid:<12} {ask:<12} {vol:<10} {change:<10} {count:<8}{flag}")

            print("-"*70)

            # 推荐
            if ranking[0]['volatility'] > 0.05:
                best = ranking[0]
                print(f"\n  >>> 推荐交易: {best['symbol']} (波动率 {best['volatility']:.4f}%)")
            else:
                print(f"\n  >>> 当前波动较小，等待机会...")

            # 未找到的品种
            found = [r['symbol'] for r in ranking]
            missing = [s for s in self.symbols if s not in found]
            if missing:
                print(f"\n  [!] 未找到品种: {', '.join(missing)} (请添加到自选)")

        print("\n" + "="*70)
        print("  按 Ctrl+C 停止监控 | 浏览器窗口可最小化")
        print("="*70)

    async def run(self, duration: int = None):
        """运行监控"""
        if not await self.start():
            return

        start_time = time.time()
        check_count = 0

        print("\n  开始监控，每 {:.1f} 秒刷新一次...\n".format(self.interval))

        try:
            while self._is_running:
                # 获取数据
                await self.fetch_symbol_prices()
                check_count += 1

                # 打印状态
                self.print_status()

                # 检查时长
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n  已运行 {duration} 秒，停止监控")
                    break

                # 等待
                await asyncio.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\n  收到停止信号...")
        finally:
            await self.stop()


async def main():
    parser = argparse.ArgumentParser(description="多品种波动监控")
    parser.add_argument(
        "--symbols",
        type=str,
        default="USOIL,XAUUSD,ETHUSD",
        help="监控的品种，逗号分隔 (默认: USOIL,XAUUSD,ETHUSD)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="刷新间隔秒数 (默认: 5)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="运行时长秒数，0=无限 (默认: 0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,  # 只显示警告和错误
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    duration = args.duration if args.duration > 0 else None

    print("\n" + "="*70)
    print("  KVB 多品种波动监控")
    print("="*70)
    print(f"  监控品种: {', '.join(symbols)}")
    print(f"  刷新间隔: {args.interval} 秒")
    print("="*70)

    monitor = VolatilityMonitor(symbols, args.interval)
    await monitor.run(duration)


if __name__ == "__main__":
    asyncio.run(main())
