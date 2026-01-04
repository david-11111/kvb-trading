"""
交易监控器

整合数据获取、策略分析、风控管理、自动交易
"""

import asyncio
import logging
import time
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np

from data_fetcher import DataFetcher, PriceData
from strategy import TradingStrategy, SignalType, Signal
from indicators import Indicators
from trade_logger import JsonlLogger, data_dir_for, dated_jsonl_path, now_event


@dataclass
class MonitorConfig:
    """监控器配置"""
    url: str = "https://mykvb.com/trade"
    headless: bool = False
    check_interval: float = 5.0      # 检查间隔（秒）
    total_capital: float = 500.0     # 总资金
    max_loss_percent: float = 0.05   # 最大亏损5%
    profit_confirm_percent: float = 0.05  # 盈利确认5%
    lot_size: float = 0.01           # 每次交易手数
    max_add_count: int = 3           # 最大加仓次数
    auto_trade: bool = False         # 是否自动执行交易


class TradingMonitor:
    """
    交易监控器

    功能：
    1. 实时获取价格数据
    2. 计算技术指标
    3. 分析交易信号
    4. 风控监控（5%止损）
    5. 自动/手动交易执行
    """

    def __init__(self, config: MonitorConfig = None):
        self.config = config or MonitorConfig()

        # 数据获取器
        self.fetcher = DataFetcher(
            url=self.config.url,
            headless=self.config.headless
        )

        # 交易策略
        self.strategy = TradingStrategy(
            total_capital=self.config.total_capital,
            max_loss_percent=self.config.max_loss_percent,
            profit_confirm_percent=self.config.profit_confirm_percent,
            max_add_count=self.config.max_add_count
        )

        # 价格历史（用于计算指标）
        self._price_history: list = []
        self._max_history = 200

        # 监控状态
        self._is_running = False
        self._stop_event = asyncio.Event()

        # 回调函数
        self.on_signal: Optional[Callable[[Signal], None]] = None
        self.on_trade: Optional[Callable[[str, float, float], None]] = None
        self.on_status: Optional[Callable[[dict], None]] = None

        self.logger = logging.getLogger("TradingMonitor")
        self.data_dir = data_dir_for(__file__)
        self.data_dir.mkdir(exist_ok=True)
        self._event_logger = JsonlLogger(dated_jsonl_path(self.data_dir, "monitor_events"))

    async def start(self):
        """启动监控"""
        self.logger.info("正在启动交易监控...")

        # 连接网页
        connected = await self.fetcher.connect()
        if not connected:
            self.logger.error("无法连接到交易页面")
            return False

        # 等待页面加载
        await asyncio.sleep(3)

        # 获取初始账户信息
        account = await self.fetcher.fetch_account()
        if account:
            self.config.total_capital = account.free_margin
            self.strategy.update_capital(account.free_margin)
            self.logger.info(f"账户资金: {account.free_margin}")

        # 获取当前品种
        symbol = await self.fetcher.get_symbol()
        self.logger.info(f"交易品种: {symbol}")

        self._is_running = True
        self._stop_event.clear()

        self.logger.info("交易监控已启动!")
        return True

    async def stop(self):
        """停止监控"""
        self._stop_event.set()
        self._is_running = False
        await self.fetcher.disconnect()
        self.logger.info("交易监控已停止")

    async def run(self):
        """运行监控循环"""
        if not self._is_running:
            started = await self.start()
            if not started:
                return

        self.logger.info(f"开始监控，间隔: {self.config.check_interval}秒")

        while self._is_running and not self._stop_event.is_set():
            try:
                await self._check_once()
            except Exception as e:
                self.logger.error(f"监控检查异常: {e}")

            # 等待下一次检查
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.check_interval
                )
            except asyncio.TimeoutError:
                pass  # 正常超时，继续循环

    async def _check_once(self):
        """执行一次检查"""
        # 1. 获取价格数据
        price_data = await self.fetcher.fetch_price()
        if not price_data:
            self.logger.warning("无法获取价格数据")
            return

        # 2. 更新价格历史
        self._price_history.append(price_data)
        if len(self._price_history) > self._max_history:
            self._price_history.pop(0)

        # 3. 检查是否有足够的历史数据
        if len(self._price_history) < 30:
            self.logger.debug(f"收集数据中... {len(self._price_history)}/30")
            return

        # 4. 准备指标计算数据
        high = np.array([p.high for p in self._price_history])
        low = np.array([p.low for p in self._price_history])
        close = np.array([p.close for p in self._price_history])

        # 5. 获取买卖价
        buy_price = await self.fetcher.get_buy_price()
        sell_price = await self.fetcher.get_sell_price()
        current_price = price_data.close

        # 6. 分析信号
        signal = self.strategy.analyze(high, low, close, current_price)

        # 7. 处理信号
        await self._handle_signal(signal, buy_price, sell_price)

        # 8. 更新状态回调
        if self.on_status:
            status = self._get_status(price_data, buy_price, sell_price, signal)
            self.on_status(status)

    async def _handle_signal(self, signal: Signal, buy_price: float, sell_price: float):
        """处理交易信号"""
        if signal.type == SignalType.NONE:
            return

        # 触发信号回调
        if self.on_signal:
            self.on_signal(signal)

        self._event_logger.log(now_event(
            "signal",
            signal=signal,
            buy_price=buy_price,
            sell_price=sell_price,
            auto_trade=self.config.auto_trade,
        ))

        # 如果不是自动交易模式，只记录信号
        if not self.config.auto_trade:
            self.logger.info(f"[信号] {signal.type.value}: {signal.reason}")
            self._event_logger.log(now_event("skip_trade", reason="auto_trade=false", signal=signal))
            return

        # 自动交易执行
        if signal.type == SignalType.BUY:
            await self._execute_buy(buy_price)

        elif signal.type == SignalType.SELL:
            await self._execute_sell(sell_price)

        elif signal.type == SignalType.ADD_LONG:
            await self._execute_buy(buy_price, is_add=True)

        elif signal.type == SignalType.ADD_SHORT:
            await self._execute_sell(sell_price, is_add=True)

        elif signal.type == SignalType.CLOSE:
            await self._execute_close()

    async def _execute_buy(self, price: float, is_add: bool = False):
        """执行买入"""
        self.logger.info(f"{'加仓' if is_add else '开仓'}买入 @ {price}")

        success = await self.fetcher.click_buy()
        if success:
            await asyncio.sleep(1)
            # 确认订单（点击确认按钮）
            confirm_btn = await self.fetcher.page.query_selector('button:has-text("市价买入")')
            if confirm_btn:
                await confirm_btn.click()
                await asyncio.sleep(1)

                # 更新策略持仓
                cost = price * self.config.lot_size * 10  # 估算保证金
                if is_add:
                    self.strategy.add_position(price, self.config.lot_size, cost)
                else:
                    self.strategy.open_position("long", price, self.config.lot_size, cost)

                if self.on_trade:
                    self.on_trade("buy", price, self.config.lot_size)

                self.logger.info(f"买入成功 @ {price}")
                self._event_logger.log(now_event(
                    "trade_executed",
                    side="buy",
                    price=price,
                    lots=self.config.lot_size,
                    is_add=is_add,
                ))
            else:
                self._event_logger.log(now_event(
                    "trade_failed",
                    side="buy",
                    price=price,
                    lots=self.config.lot_size,
                    is_add=is_add,
                    reason="confirm_button_not_found",
                ))
        else:
            self._event_logger.log(now_event(
                "trade_failed",
                side="buy",
                price=price,
                lots=self.config.lot_size,
                is_add=is_add,
                reason="click_buy_failed",
            ))

    async def _execute_sell(self, price: float, is_add: bool = False):
        """执行卖出"""
        self.logger.info(f"{'加仓' if is_add else '开仓'}卖出 @ {price}")

        success = await self.fetcher.click_sell()
        if success:
            await asyncio.sleep(1)
            confirm_btn = await self.fetcher.page.query_selector('button:has-text("市价卖出")')
            if confirm_btn:
                await confirm_btn.click()
                await asyncio.sleep(1)

                cost = price * self.config.lot_size * 10
                if is_add:
                    self.strategy.add_position(price, self.config.lot_size, cost)
                else:
                    self.strategy.open_position("short", price, self.config.lot_size, cost)

                if self.on_trade:
                    self.on_trade("sell", price, self.config.lot_size)

                self.logger.info(f"卖出成功 @ {price}")
                self._event_logger.log(now_event(
                    "trade_executed",
                    side="sell",
                    price=price,
                    lots=self.config.lot_size,
                    is_add=is_add,
                ))
            else:
                self._event_logger.log(now_event(
                    "trade_failed",
                    side="sell",
                    price=price,
                    lots=self.config.lot_size,
                    is_add=is_add,
                    reason="confirm_button_not_found",
                ))
        else:
            self._event_logger.log(now_event(
                "trade_failed",
                side="sell",
                price=price,
                lots=self.config.lot_size,
                is_add=is_add,
                reason="click_sell_failed",
            ))

    async def _execute_close(self):
        """执行平仓"""
        self.logger.info("执行平仓...")

        page = getattr(self.fetcher, "page", None)
        if not page:
            self.logger.warning("页面对象不存在，无法执行网页平仓；仅更新本地策略持仓")
            return self._record_close_only()

        # 1) 尝试直接点击“平仓/关闭”按钮（如果页面当前就展示）
        if await self._click_close_flow():
            return self._record_close_only()

        # 2) 切换到“持仓”Tab并选择一条持仓行，再尝试平仓
        await self._ensure_positions_tab()
        await asyncio.sleep(0.3)

        symbol = await self.fetcher.get_symbol() or ""
        row_clicked = await self._select_position_row(prefer_symbol=symbol)
        if not row_clicked:
            self.logger.warning("未能定位到持仓行（可能当前无持仓/表格未加载）")

        if await self._click_close_flow():
            return self._record_close_only()

        # 3) 右键/双击行触发上下文操作，再尝试平仓（不同页面版本交互可能不同）
        if row_clicked:
            try:
                await self._selected_row.click(button="right")
                await asyncio.sleep(0.2)
            except Exception:
                pass
            if await self._click_close_flow():
                return self._record_close_only()

            try:
                await self._selected_row.dblclick()
                await asyncio.sleep(0.2)
            except Exception:
                pass
            if await self._click_close_flow():
                return self._record_close_only()

        self.logger.error("网页平仓失败：未找到可点击的平仓入口或确认按钮；未执行反向下单兜底")
        return False

    async def _ensure_positions_tab(self) -> bool:
        """确保底部面板切换到“持仓”"""
        page = self.fetcher.page
        # 页面上“持仓(1)”等可能带数字，使用正则前缀匹配
        selectors = [
            'text=/^持仓/',
            'text=持仓',
        ]
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0 and await loc.is_visible():
                    await loc.click()
                    return True
            except Exception:
                continue
        return False

    async def _select_position_row(self, prefer_symbol: str = "") -> bool:
        """选择一条持仓行（优先当前品种），用于后续触发平仓入口"""
        page = self.fetcher.page
        self._selected_row = None

        # 兼容不同表格实现：fmui-table 或原生 table
        candidates = page.locator("div.fmui-table-body-tr, tbody tr")

        try:
            count = await candidates.count()
        except Exception:
            count = 0

        if count <= 0:
            return False

        # 先尝试匹配当前品种
        if prefer_symbol:
            try:
                row = candidates.filter(has_text=prefer_symbol).first
                if await row.count() > 0:
                    await row.click()
                    self._selected_row = row
                    return True
            except Exception:
                pass

        # 再尝试选择第一行
        try:
            row = candidates.first
            await row.click()
            self._selected_row = row
            return True
        except Exception:
            return False

    async def _click_close_flow(self) -> bool:
        """
        执行“点击平仓入口 -> 点击确认”的完整流程。
        返回是否至少点击到确认（或平仓入口后无需确认也成功）。
        """
        page = self.fetcher.page

        # 可能的平仓入口：按钮/菜单项/文本
        close_entry_selectors = [
            'button:has-text("平仓")',
            'button:has-text("关闭")',
            'text=平仓',
            'text=关闭',
            'text=Close',
        ]

        clicked_entry = await self._click_first_visible(close_entry_selectors)
        if not clicked_entry:
            return False

        await asyncio.sleep(0.4)

        # 可能的确认按钮：不同版本可能为“市价平仓/市价卖出/市价买入/确认/确定”
        confirm_selectors = [
            'button:has-text("市价平仓")',
            'button:has-text("市价卖出")',
            'button:has-text("市价买入")',
            'button:has-text("确认")',
            'button:has-text("确定")',
            'button:has-text("OK")',
            'button:has-text("Confirm")',
        ]

        # 有些情况下点击平仓后直接成交（无确认弹窗），因此确认按钮点不到也算部分成功
        await self._click_first_visible(confirm_selectors)

        # 给页面一个处理时间（toast/弹窗/持仓刷新）
        await asyncio.sleep(1.0)
        return True

    async def _click_first_visible(self, selectors) -> bool:
        """依次尝试点击第一个可见元素"""
        page = self.fetcher.page
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0 and await loc.is_visible():
                    await loc.click()
                    return True
            except Exception:
                continue
        return False

    def _record_close_only(self) -> bool:
        """仅在本地策略中记录平仓（用于网页已平仓/或无法操作网页时保持内部状态一致）"""
        if self.strategy.position.direction:
            price = self._price_history[-1].close if self._price_history else 0
            realized_pnl = self.strategy.close_position(price)
            self.logger.info(f"平仓记录完成，实现盈亏: {realized_pnl:.2f}")
            return True
        return False

    def _get_status(self, price_data: PriceData, buy_price: float,
                    sell_price: float, signal: Signal) -> dict:
        """获取当前状态"""
        return {
            "timestamp": time.time(),
            "price": {
                "open": price_data.open,
                "high": price_data.high,
                "low": price_data.low,
                "close": price_data.close,
                "buy": buy_price,
                "sell": sell_price,
            },
            "position": self.strategy.get_position_summary(),
            "signal": {
                "type": signal.type.value,
                "reason": signal.reason,
                "strength": signal.strength,
            },
            "history_count": len(self._price_history),
        }

    # ==================== 手动交易接口 ====================

    async def manual_buy(self, lots: float = None) -> bool:
        """手动买入"""
        lots = lots or self.config.lot_size
        buy_price = await self.fetcher.get_buy_price()

        if not buy_price:
            return False

        success = await self.fetcher.click_buy()
        if success:
            await asyncio.sleep(1)
            confirm_btn = await self.fetcher.page.query_selector('button:has-text("市价买入")')
            if confirm_btn:
                await confirm_btn.click()
                await asyncio.sleep(2)

                # 记录持仓
                if not self.strategy.position.direction:
                    cost = buy_price * lots * 10
                    self.strategy.open_position("long", buy_price, lots, cost)
                else:
                    cost = buy_price * lots * 10
                    self.strategy.add_position(buy_price, lots, cost)

                return True
        return False

    async def manual_sell(self, lots: float = None) -> bool:
        """手动卖出"""
        lots = lots or self.config.lot_size
        sell_price = await self.fetcher.get_sell_price()

        if not sell_price:
            return False

        success = await self.fetcher.click_sell()
        if success:
            await asyncio.sleep(1)
            confirm_btn = await self.fetcher.page.query_selector('button:has-text("市价卖出")')
            if confirm_btn:
                await confirm_btn.click()
                await asyncio.sleep(2)

                if not self.strategy.position.direction:
                    cost = sell_price * lots * 10
                    self.strategy.open_position("short", sell_price, lots, cost)
                else:
                    cost = sell_price * lots * 10
                    self.strategy.add_position(sell_price, lots, cost)

                return True
        return False

    async def get_current_status(self) -> dict:
        """获取当前状态（单次查询）"""
        price = await self.fetcher.fetch_price()
        buy_price = await self.fetcher.get_buy_price()
        sell_price = await self.fetcher.get_sell_price()
        account = await self.fetcher.fetch_account()

        return {
            "price": {
                "current": price.close if price else None,
                "buy": buy_price,
                "sell": sell_price,
            },
            "account": {
                "equity": account.equity if account else None,
                "free_margin": account.free_margin if account else None,
                "profit": account.profit if account else None,
            },
            "position": self.strategy.get_position_summary(),
        }


# ==================== 便捷函数 ====================

async def create_monitor(capital: float = 500.0, auto_trade: bool = False) -> TradingMonitor:
    """
    创建并启动监控器

    Args:
        capital: 总资金
        auto_trade: 是否自动交易

    Returns:
        TradingMonitor实例
    """
    config = MonitorConfig(
        total_capital=capital,
        auto_trade=auto_trade
    )
    monitor = TradingMonitor(config)
    await monitor.start()
    return monitor
