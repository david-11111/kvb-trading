"""
多品种监控器

监控多个交易品种，计算波动率，发现交易机会
支持品种: USOIL(原油), XAUUSD(黄金), ETHUSD(以太坊) 等
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from data_fetcher import DataFetcher


@dataclass
class SymbolData:
    """品种数据"""
    symbol: str                      # 品种代码
    bid: float                       # 卖价
    ask: float                       # 买价
    spread: int                      # 点差
    mid_price: float                 # 中间价
    timestamp: float                 # 时间戳

    # 波动率相关
    price_history: List[float] = field(default_factory=list)  # 价格历史
    volatility: float = 0.0          # 波动率(%)
    atr: float = 0.0                 # ATR
    price_change: float = 0.0        # 价格变动(%)

    def update_volatility(self):
        """更新波动率"""
        if len(self.price_history) < 2:
            return

        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1] * 100  # 收益率(%)

        # 波动率 = 收益率标准差
        self.volatility = float(np.std(returns)) if len(returns) > 0 else 0.0

        # 价格变动 = (最新价 - 第一个价) / 第一个价
        self.price_change = (prices[-1] - prices[0]) / prices[0] * 100


@dataclass
class MultiMonitorConfig:
    """多品种监控配置"""
    symbols: List[str] = field(default_factory=lambda: ["USOIL", "XAUUSD", "ETHUSD"])
    check_interval: float = 5.0      # 检查间隔（秒）
    history_length: int = 60         # 保存多少条历史数据
    volatility_threshold: float = 0.5  # 波动率阈值(%)，超过此值认为有交易机会
    headless: bool = False


class MultiSymbolMonitor:
    """
    多品种监控器

    功能：
    1. 同时监控多个品种的价格
    2. 计算每个品种的波动率
    3. 对波动率排名
    4. 发现高波动交易机会
    5. 支持切换到目标品种
    """

    def __init__(self, config: MultiMonitorConfig = None):
        self.config = config or MultiMonitorConfig()

        # 数据获取器
        self.fetcher = DataFetcher(headless=self.config.headless)

        # 品种数据缓存
        self.symbols_data: Dict[str, SymbolData] = {}

        # 监控状态
        self._is_running = False
        self._stop_event = asyncio.Event()

        self.logger = logging.getLogger("MultiMonitor")

    async def start(self) -> bool:
        """启动监控"""
        self.logger.info("正在启动多品种监控...")

        connected = await self.fetcher.connect()
        if not connected:
            self.logger.error("无法连接到交易页面")
            return False

        await asyncio.sleep(3)

        # 初始化品种数据
        for symbol in self.config.symbols:
            self.symbols_data[symbol] = SymbolData(
                symbol=symbol,
                bid=0.0, ask=0.0, spread=0,
                mid_price=0.0, timestamp=time.time()
            )

        self._is_running = True
        self._stop_event.clear()

        self.logger.info(f"多品种监控已启动! 监控品种: {self.config.symbols}")
        return True

    async def stop(self):
        """停止监控"""
        self._stop_event.set()
        self._is_running = False
        await self.fetcher.disconnect()
        self.logger.info("多品种监控已停止")

    async def fetch_all_symbols(self) -> Dict[str, SymbolData]:
        """
        获取所有品种的价格数据
        从自选列表中读取
        """
        page = self.fetcher.page
        if not page:
            return {}

        result = {}

        for symbol in self.config.symbols:
            try:
                # 查找品种项
                item_selector = f'.custom-list-item[data-item="{symbol}"]'
                item = await page.query_selector(item_selector)

                if not item:
                    self.logger.debug(f"品种 {symbol} 不在自选列表中")
                    continue

                # 获取买卖价
                bid_elem = await item.query_selector('.item-bid span')
                ask_elem = await item.query_selector('.item-ask span')
                spread_elem = await item.query_selector('.item-displaySpread span')

                bid = 0.0
                ask = 0.0
                spread = 0

                if bid_elem:
                    bid_text = await bid_elem.inner_text()
                    bid = float(bid_text.strip())

                if ask_elem:
                    ask_text = await ask_elem.inner_text()
                    ask = float(ask_text.strip())

                if spread_elem:
                    spread_text = await spread_elem.inner_text()
                    spread = int(spread_text.strip())

                mid_price = (bid + ask) / 2 if bid and ask else 0.0

                # 更新数据
                if symbol in self.symbols_data:
                    data = self.symbols_data[symbol]
                    data.bid = bid
                    data.ask = ask
                    data.spread = spread
                    data.mid_price = mid_price
                    data.timestamp = time.time()

                    # 添加到价格历史
                    if mid_price > 0:
                        data.price_history.append(mid_price)
                        # 保持历史长度
                        if len(data.price_history) > self.config.history_length:
                            data.price_history.pop(0)
                        # 更新波动率
                        data.update_volatility()

                    result[symbol] = data

            except Exception as e:
                self.logger.debug(f"获取 {symbol} 数据失败: {e}")

        return result

    def get_volatility_ranking(self) -> List[SymbolData]:
        """
        获取波动率排名（从高到低）
        """
        valid_data = [d for d in self.symbols_data.values() if d.volatility > 0]
        return sorted(valid_data, key=lambda x: x.volatility, reverse=True)

    def get_best_opportunity(self) -> Optional[SymbolData]:
        """
        获取最佳交易机会（波动率最高的品种）
        """
        ranking = self.get_volatility_ranking()
        if ranking and ranking[0].volatility >= self.config.volatility_threshold:
            return ranking[0]
        return None

    async def switch_to_symbol(self, symbol: str) -> bool:
        """
        切换到指定品种
        """
        return await self.fetcher.switch_symbol(symbol)

    async def run(self, duration: int = None):
        """
        运行监控循环

        Args:
            duration: 运行时长（秒），None表示无限运行
        """
        if not self._is_running:
            started = await self.start()
            if not started:
                return

        self.logger.info(f"开始监控，间隔: {self.config.check_interval}秒")

        start_time = time.time()
        check_count = 0

        while self._is_running and not self._stop_event.is_set():
            try:
                # 获取所有品种数据
                await self.fetch_all_symbols()
                check_count += 1

                # 每5次检查打印一次状态
                if check_count % 5 == 0:
                    self._print_status()

                # 检查是否到达运行时长
                if duration and (time.time() - start_time) >= duration:
                    self.logger.info(f"已运行 {duration} 秒，停止监控")
                    break

            except Exception as e:
                self.logger.error(f"监控检查异常: {e}")

            # 等待下一次检查
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.check_interval
                )
            except asyncio.TimeoutError:
                pass

    def _print_status(self):
        """打印当前状态"""
        print("\n" + "="*60)
        print(f"多品种监控状态 - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        print(f"{'品种':<10} {'买价':<12} {'卖价':<12} {'点差':<6} {'波动率%':<8} {'涨跌%':<8}")
        print("-"*60)

        ranking = self.get_volatility_ranking()

        for data in ranking:
            symbol = data.symbol
            bid = f"{data.bid:.2f}" if data.bid else "N/A"
            ask = f"{data.ask:.2f}" if data.ask else "N/A"
            spread = f"{data.spread}" if data.spread else "N/A"
            vol = f"{data.volatility:.3f}" if data.volatility else "0.000"
            change = f"{data.price_change:+.2f}" if data.price_change else "0.00"

            # 高波动标记
            flag = " ***" if data.volatility >= self.config.volatility_threshold else ""

            print(f"{symbol:<10} {bid:<12} {ask:<12} {spread:<6} {vol:<8} {change:<8}{flag}")

        # 显示无数据的品种
        for symbol in self.config.symbols:
            if symbol not in [d.symbol for d in ranking]:
                print(f"{symbol:<10} {'N/A':<12} {'N/A':<12} {'N/A':<6} {'N/A':<8} {'N/A':<8} (未在自选)")

        print("-"*60)

        # 交易机会提示
        best = self.get_best_opportunity()
        if best:
            print(f">>> 发现交易机会: {best.symbol} (波动率 {best.volatility:.3f}%)")
        else:
            print(">>> 暂无高波动交易机会")

        print("="*60 + "\n")

    def get_summary(self) -> dict:
        """获取监控摘要"""
        ranking = self.get_volatility_ranking()
        best = self.get_best_opportunity()

        return {
            "timestamp": time.time(),
            "symbols": {
                symbol: {
                    "bid": data.bid,
                    "ask": data.ask,
                    "spread": data.spread,
                    "volatility": data.volatility,
                    "price_change": data.price_change,
                    "history_count": len(data.price_history),
                }
                for symbol, data in self.symbols_data.items()
            },
            "ranking": [d.symbol for d in ranking],
            "best_opportunity": best.symbol if best else None,
        }


# ==================== 便捷函数 ====================

async def quick_scan(symbols: List[str] = None, duration: int = 60) -> dict:
    """
    快速扫描多个品种的波动情况

    Args:
        symbols: 要监控的品种列表
        duration: 扫描时长（秒）

    Returns:
        扫描结果
    """
    config = MultiMonitorConfig(
        symbols=symbols or ["USOIL", "XAUUSD", "ETHUSD"],
        check_interval=3.0,
    )

    monitor = MultiSymbolMonitor(config)
    await monitor.run(duration=duration)

    summary = monitor.get_summary()
    await monitor.stop()

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多品种波动监控")
    parser.add_argument(
        "--symbols",
        type=str,
        default="USOIL,XAUUSD,ETHUSD",
        help="要监控的品种，逗号分隔",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="检查间隔（秒）",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="监控时长（秒），0表示无限",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
    )

    symbols = [s.strip() for s in args.symbols.split(",")]

    config = MultiMonitorConfig(
        symbols=symbols,
        check_interval=args.interval,
    )

    monitor = MultiSymbolMonitor(config)

    duration = args.duration if args.duration > 0 else None

    asyncio.run(monitor.run(duration=duration))
