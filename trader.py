"""
交易执行模块
通过 Playwright 自动化控制网页进行交易操作
保存开仓价格用于风控
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from config import WEB_CONFIG, TRADE_CONFIG
from trade_logger import data_dir_for


@dataclass
class TradeRecord:
    """交易记录"""
    trade_id: str                # 交易ID
    action: str                  # "buy" / "sell" / "close"
    direction: str               # "long" / "short"
    price: float                 # 成交价格
    quantity: float              # 成交数量
    timestamp: float             # 成交时间戳
    pnl: float = 0.0             # 盈亏（平仓时）
    notes: str = ""              # 备注


@dataclass
class OpenPosition:
    """持仓信息"""
    direction: str               # "long" / "short"
    entry_price: float           # 开仓价格
    quantity: float              # 持仓数量
    entry_time: float            # 开仓时间
    take_profit: float           # 止盈价格
    stop_loss: float             # 止损价格
    trade_id: str                # 关联的交易ID


class Trader:
    """
    交易执行器

    功能：
    1. 通过 Playwright 点击网页按钮执行交易
    2. 保存所有交易记录
    3. 保存开仓价格用于风控
    """

    def __init__(self, page=None, initial_capital: float = 10000):
        """
        初始化交易执行器

        Args:
            page: Playwright页面对象
            initial_capital: 初始资金
        """
        self.page = page
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # 当前持仓
        self.current_position: Optional[OpenPosition] = None

        # 交易历史
        self.trade_history: List[TradeRecord] = []

        # 元素选择器 - 针对 mykvb.com/trade 页面
        # KVB按钮格式: "买入 价格"(绿色) / "卖出 价格"(红色)
        self.selectors = {
            "buy_button": "//*[contains(text(), '买入')]",
            "sell_button": "//*[contains(text(), '卖出')]",
            "close_button": "//*[contains(text(), '平仓') or contains(text(), '关闭')]",
            "quantity_input": "input[type='number'], input[placeholder*='手数'], input[placeholder*='数量']",
            "confirm_button": "//*[contains(text(), '确认') or contains(text(), '确定') or contains(text(), 'OK')]",
        }

        # 数据文件路径
        self.data_dir = data_dir_for(__file__)
        self.data_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger("Trader")

        # 加载历史数据
        self._load_data()

    def set_page(self, page):
        """设置页面对象"""
        self.page = page

    async def buy(self, price: float, quantity: float,
                  take_profit_percent: float = None,
                  position_size: float = None) -> Optional[TradeRecord]:
        """
        买入（做多）

        Args:
            price: 当前价格
            quantity: 交易数量
            take_profit_percent: 止盈比例
            position_size: 仓位比例

        Returns:
            TradeRecord对象
        """
        # 检查是否已有持仓
        if self.current_position:
            if self.current_position.direction == "long":
                self.logger.warning("已有多头持仓，不能重复开多")
                return None
            else:
                # 有空头持仓，先平仓
                await self.close_position(price)

        # 计算仓位
        if position_size:
            actual_quantity = (self.current_capital * position_size) / price
        else:
            actual_quantity = quantity

        # 执行买入操作
        success = await self._execute_buy()

        if success or self.page is None:  # 无页面时模拟成功
            trade_id = f"T{int(time.time() * 1000)}"

            # 计算止盈止损价格
            tp_percent = take_profit_percent or TRADE_CONFIG["oscillation"]["take_profit_percent"]
            take_profit_price = price * (1 + tp_percent)
            stop_loss_price = price * 0.95  # 5%止损

            # 创建持仓
            self.current_position = OpenPosition(
                direction="long",
                entry_price=price,
                quantity=actual_quantity,
                entry_time=time.time(),
                take_profit=take_profit_price,
                stop_loss=stop_loss_price,
                trade_id=trade_id
            )

            # 创建交易记录
            record = TradeRecord(
                trade_id=trade_id,
                action="buy",
                direction="long",
                price=price,
                quantity=actual_quantity,
                timestamp=time.time(),
                notes=f"开多仓，止盈: {take_profit_price:.4f}, 止损: {stop_loss_price:.4f}"
            )

            self.trade_history.append(record)
            self._save_data()

            self.logger.info(f"买入成功: 价格={price}, 数量={actual_quantity}, ID={trade_id}")
            return record

        return None

    async def sell(self, price: float, quantity: float,
                   take_profit_percent: float = None,
                   position_size: float = None) -> Optional[TradeRecord]:
        """
        卖出（做空）

        Args:
            price: 当前价格
            quantity: 交易数量
            take_profit_percent: 止盈比例
            position_size: 仓位比例

        Returns:
            TradeRecord对象
        """
        # 检查是否已有持仓
        if self.current_position:
            if self.current_position.direction == "short":
                self.logger.warning("已有空头持仓，不能重复开空")
                return None
            else:
                # 有多头持仓，先平仓
                await self.close_position(price)

        # 计算仓位
        if position_size:
            actual_quantity = (self.current_capital * position_size) / price
        else:
            actual_quantity = quantity

        # 执行卖出操作
        success = await self._execute_sell()

        if success or self.page is None:
            trade_id = f"T{int(time.time() * 1000)}"

            # 计算止盈止损价格
            tp_percent = take_profit_percent or TRADE_CONFIG["oscillation"]["take_profit_percent"]
            take_profit_price = price * (1 - tp_percent)
            stop_loss_price = price * 1.05  # 5%止损

            # 创建持仓
            self.current_position = OpenPosition(
                direction="short",
                entry_price=price,
                quantity=actual_quantity,
                entry_time=time.time(),
                take_profit=take_profit_price,
                stop_loss=stop_loss_price,
                trade_id=trade_id
            )

            # 创建交易记录
            record = TradeRecord(
                trade_id=trade_id,
                action="sell",
                direction="short",
                price=price,
                quantity=actual_quantity,
                timestamp=time.time(),
                notes=f"开空仓，止盈: {take_profit_price:.4f}, 止损: {stop_loss_price:.4f}"
            )

            self.trade_history.append(record)
            self._save_data()

            self.logger.info(f"卖出成功: 价格={price}, 数量={actual_quantity}, ID={trade_id}")
            return record

        return None

    async def close_position(self, current_price: float) -> Optional[TradeRecord]:
        """
        平仓

        Args:
            current_price: 当前价格

        Returns:
            TradeRecord对象
        """
        if not self.current_position:
            self.logger.warning("无持仓可平")
            return None

        pos = self.current_position

        # 计算盈亏
        if pos.direction == "long":
            pnl = (current_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - current_price) * pos.quantity

        # 执行平仓操作
        success = await self._execute_close()

        if success or self.page is None:
            trade_id = f"T{int(time.time() * 1000)}"

            # 更新资金
            self.current_capital += pnl

            # 创建交易记录
            record = TradeRecord(
                trade_id=trade_id,
                action="close",
                direction=pos.direction,
                price=current_price,
                quantity=pos.quantity,
                timestamp=time.time(),
                pnl=pnl,
                notes=f"平{pos.direction}仓，开仓价: {pos.entry_price:.4f}, 盈亏: {pnl:.2f}"
            )

            self.trade_history.append(record)

            # 清除持仓
            self.current_position = None

            self._save_data()

            pnl_status = "盈利" if pnl > 0 else "亏损"
            self.logger.info(f"平仓成功: {pnl_status} {abs(pnl):.2f}, 当前资金: {self.current_capital:.2f}")

            return record

        return None

    async def _execute_buy(self) -> bool:
        """执行买入点击"""
        if not self.page:
            self.logger.info("模拟模式：执行买入")
            return True

        try:
            # 使用Playwright的locator查找包含"买入"文字的按钮
            buy_btn = self.page.locator("text=买入").first
            if await buy_btn.count() > 0:
                await buy_btn.click()
                self.logger.info("已点击买入按钮")
                await asyncio.sleep(0.5)

                # 点击确认按钮（如果有）
                await asyncio.sleep(0.3)
                for confirm_text in ["确认", "确定", "OK", "Confirm"]:
                    confirm_btn = self.page.locator(f"text={confirm_text}").first
                    if await confirm_btn.count() > 0:
                        await confirm_btn.click()
                        self.logger.info(f"已点击确认按钮: {confirm_text}")
                        break

                return True
            else:
                self.logger.error("找不到买入按钮")
                return False
        except Exception as e:
            self.logger.error(f"执行买入失败: {e}")
            return False

    async def _execute_sell(self) -> bool:
        """执行卖出点击"""
        if not self.page:
            self.logger.info("模拟模式：执行卖出")
            return True

        try:
            # 使用Playwright的locator查找包含"卖出"文字的按钮
            sell_btn = self.page.locator("text=卖出").first
            if await sell_btn.count() > 0:
                await sell_btn.click()
                self.logger.info("已点击卖出按钮")
                await asyncio.sleep(0.5)

                # 点击确认按钮（如果有）
                await asyncio.sleep(0.3)
                for confirm_text in ["确认", "确定", "OK", "Confirm"]:
                    confirm_btn = self.page.locator(f"text={confirm_text}").first
                    if await confirm_btn.count() > 0:
                        await confirm_btn.click()
                        self.logger.info(f"已点击确认按钮: {confirm_text}")
                        break

                return True
            else:
                self.logger.error("找不到卖出按钮")
                return False
        except Exception as e:
            self.logger.error(f"执行卖出失败: {e}")
            return False

    async def _execute_close(self) -> bool:
        """执行平仓点击"""
        if not self.page:
            self.logger.info("模拟模式：执行平仓")
            return True

        try:
            # 使用Playwright的locator查找平仓按钮
            close_btn = None
            for close_text in ["平仓", "关闭", "Close"]:
                btn = self.page.locator(f"text={close_text}").first
                if await btn.count() > 0:
                    close_btn = btn
                    break

            if close_btn:
                await close_btn.click()
                self.logger.info("已点击平仓按钮")
                await asyncio.sleep(0.5)

                # 点击确认按钮（如果有）
                await asyncio.sleep(0.3)
                for confirm_text in ["确认", "确定", "OK", "Confirm"]:
                    confirm_btn = self.page.locator(f"text={confirm_text}").first
                    if await confirm_btn.count() > 0:
                        await confirm_btn.click()
                        self.logger.info(f"已点击确认按钮: {confirm_text}")
                        break

                return True
            else:
                self.logger.error("找不到平仓按钮")
                return False
        except Exception as e:
            self.logger.error(f"执行平仓失败: {e}")
            return False

    def get_position_info(self) -> Optional[Dict]:
        """获取当前持仓信息"""
        if not self.current_position:
            return None

        pos = self.current_position
        return {
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "entry_time": datetime.fromtimestamp(pos.entry_time).isoformat(),
            "take_profit": pos.take_profit,
            "stop_loss": pos.stop_loss,
            "trade_id": pos.trade_id
        }

    def get_trade_summary(self) -> Dict:
        """获取交易统计摘要"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "current_capital": self.current_capital
            }

        closed_trades = [t for t in self.trade_history if t.action == "close"]
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl < 0]
        total_pnl = sum(t.pnl for t in closed_trades)

        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "total_pnl": total_pnl,
            "win_rate": len(winning) / len(closed_trades) * 100 if closed_trades else 0,
            "current_capital": self.current_capital,
            "return_percent": (self.current_capital - self.initial_capital) / self.initial_capital * 100
        }

    def _save_data(self):
        """保存数据到文件"""
        # 保存持仓
        position_file = self.data_dir / "position.json"
        if self.current_position:
            pos_data = {
                "direction": self.current_position.direction,
                "entry_price": self.current_position.entry_price,
                "quantity": self.current_position.quantity,
                "entry_time": self.current_position.entry_time,
                "take_profit": self.current_position.take_profit,
                "stop_loss": self.current_position.stop_loss,
                "trade_id": self.current_position.trade_id
            }
            position_file.write_text(json.dumps(pos_data, indent=2))
        else:
            if position_file.exists():
                position_file.unlink()

        # 保存交易历史
        history_file = self.data_dir / "trade_history.json"
        history_data = [{
            "trade_id": t.trade_id,
            "action": t.action,
            "direction": t.direction,
            "price": t.price,
            "quantity": t.quantity,
            "timestamp": t.timestamp,
            "pnl": t.pnl,
            "notes": t.notes
        } for t in self.trade_history]
        history_file.write_text(json.dumps(history_data, indent=2))

        # 保存资金状态
        capital_file = self.data_dir / "capital.json"
        capital_file.write_text(json.dumps({
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital
        }, indent=2))

    def _load_data(self):
        """加载历史数据"""
        # 加载持仓
        position_file = self.data_dir / "position.json"
        if position_file.exists():
            try:
                pos_data = json.loads(position_file.read_text())
                self.current_position = OpenPosition(**pos_data)
                self.logger.info(f"加载持仓: {self.current_position.direction} @ {self.current_position.entry_price}")
            except Exception as e:
                self.logger.error(f"加载持仓失败: {e}")

        # 加载交易历史
        history_file = self.data_dir / "trade_history.json"
        if history_file.exists():
            try:
                history_data = json.loads(history_file.read_text())
                self.trade_history = [TradeRecord(**t) for t in history_data]
                self.logger.info(f"加载 {len(self.trade_history)} 条交易记录")
            except Exception as e:
                self.logger.error(f"加载交易历史失败: {e}")

        # 加载资金状态
        capital_file = self.data_dir / "capital.json"
        if capital_file.exists():
            try:
                capital_data = json.loads(capital_file.read_text())
                self.current_capital = capital_data.get("current_capital", self.initial_capital)
                self.logger.info(f"加载资金: {self.current_capital}")
            except Exception as e:
                self.logger.error(f"加载资金状态失败: {e}")

    def update_selectors(self, selectors: Dict[str, str]):
        """更新元素选择器"""
        self.selectors.update(selectors)
