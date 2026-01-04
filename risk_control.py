"""
风控模块
核心功能：
1. 亏损达到总资金5%时立即止损
2. 盈利时不干预，让利润奔跑
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from config import RISK_CONFIG


class RiskAction(Enum):
    """风控动作"""
    NONE = "none"                  # 无需动作
    STOP_LOSS = "stop_loss"        # 止损平仓
    TAKE_PROFIT = "take_profit"    # 止盈平仓（可选）
    WARNING = "warning"            # 风险警告


@dataclass
class Position:
    """持仓信息"""
    direction: str          # "long" 或 "short"
    entry_price: float      # 开仓价格
    quantity: float         # 持仓数量
    entry_time: float       # 开仓时间戳


@dataclass
class RiskStatus:
    """风控状态"""
    action: RiskAction       # 建议动作
    current_pnl: float       # 当前盈亏金额
    pnl_percent: float       # 盈亏百分比
    is_profit: bool          # 是否盈利
    message: str             # 状态消息


class RiskController:
    """
    风控控制器

    核心规则：
    1. 盈利时的波动不处理
    2. 亏损达到总资金5%时立即平仓
    """

    def __init__(self, total_capital: float,
                 max_loss_percent: float = None,
                 on_stop_loss: Optional[Callable] = None):
        """
        初始化风控控制器

        Args:
            total_capital: 总资金
            max_loss_percent: 最大允许亏损比例，默认5%
            on_stop_loss: 止损时的回调函数
        """
        self.total_capital = total_capital
        self.max_loss_percent = max_loss_percent or RISK_CONFIG["max_loss_percent"]
        self.max_loss_amount = total_capital * self.max_loss_percent
        self.on_stop_loss = on_stop_loss

        self.current_position: Optional[Position] = None
        self.is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.logger = logging.getLogger("RiskController")

    def set_position(self, direction: str, entry_price: float, quantity: float):
        """
        设置当前持仓

        Args:
            direction: "long" 或 "short"
            entry_price: 开仓价格
            quantity: 持仓数量
        """
        self.current_position = Position(
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=time.time()
        )
        self.logger.info(f"设置持仓: {direction} @ {entry_price}, 数量: {quantity}")

    def clear_position(self):
        """清除当前持仓"""
        self.current_position = None
        self.logger.info("持仓已清除")

    def update_capital(self, new_capital: float):
        """更新总资金"""
        self.total_capital = new_capital
        self.max_loss_amount = new_capital * self.max_loss_percent
        self.logger.info(f"更新总资金: {new_capital}, 最大亏损限额: {self.max_loss_amount}")

    def calculate_pnl(self, current_price: float) -> tuple:
        """
        计算当前盈亏

        Args:
            current_price: 当前价格

        Returns:
            (盈亏金额, 盈亏百分比)
        """
        if self.current_position is None:
            return 0.0, 0.0

        pos = self.current_position
        price_change = current_price - pos.entry_price

        # 根据方向计算盈亏
        if pos.direction == "long":
            pnl = price_change * pos.quantity
        else:  # short
            pnl = -price_change * pos.quantity

        pnl_percent = pnl / self.total_capital if self.total_capital else 0.0

        return pnl, pnl_percent

    def check_risk(self, current_price: float) -> RiskStatus:
        """
        检查风险状态

        核心逻辑：
        - 盈利时：不处理，让利润奔跑
        - 亏损达到5%：立即止损

        Args:
            current_price: 当前价格

        Returns:
            RiskStatus对象
        """
        if self.current_position is None:
            return RiskStatus(
                action=RiskAction.NONE,
                current_pnl=0.0,
                pnl_percent=0.0,
                is_profit=True,
                message="无持仓"
            )

        pnl, pnl_percent = self.calculate_pnl(current_price)
        is_profit = pnl >= 0

        # 盈利时不处理
        if is_profit:
            return RiskStatus(
                action=RiskAction.NONE,
                current_pnl=pnl,
                pnl_percent=pnl_percent,
                is_profit=True,
                message=f"盈利中: {pnl:.2f} ({pnl_percent*100:.2f}%)，继续持有"
            )

        # 亏损检查
        loss_amount = abs(pnl)
        loss_percent = abs(pnl_percent)

        # 亏损达到5%，触发止损
        if loss_percent >= self.max_loss_percent:
            return RiskStatus(
                action=RiskAction.STOP_LOSS,
                current_pnl=pnl,
                pnl_percent=pnl_percent,
                is_profit=False,
                message=f"触发止损！亏损: {loss_amount:.2f} ({loss_percent*100:.2f}%)，已达到{self.max_loss_percent*100}%限额"
            )

        # 亏损但未达到止损线
        warning_threshold = self.max_loss_percent * 0.8  # 80%警戒线
        if loss_percent >= warning_threshold:
            return RiskStatus(
                action=RiskAction.WARNING,
                current_pnl=pnl,
                pnl_percent=pnl_percent,
                is_profit=False,
                message=f"风险警告！亏损: {loss_amount:.2f} ({loss_percent*100:.2f}%)，接近止损线"
            )

        return RiskStatus(
            action=RiskAction.NONE,
            current_pnl=pnl,
            pnl_percent=pnl_percent,
            is_profit=False,
            message=f"亏损中: {loss_amount:.2f} ({loss_percent*100:.2f}%)，继续监控"
        )

    def start_monitoring(self, price_getter: Callable[[], float],
                         check_interval: float = None):
        """
        启动风控监控

        Args:
            price_getter: 获取当前价格的函数
            check_interval: 检查间隔（秒）
        """
        if self.is_monitoring:
            self.logger.warning("风控监控已在运行")
            return

        check_interval = check_interval or RISK_CONFIG["check_interval"]
        self._stop_event.clear()
        self.is_monitoring = True

        def monitor_loop():
            while not self._stop_event.is_set():
                try:
                    current_price = price_getter()
                    status = self.check_risk(current_price)

                    # 记录状态
                    if status.action == RiskAction.STOP_LOSS:
                        self.logger.critical(status.message)
                        if self.on_stop_loss:
                            self.on_stop_loss(status)
                        # 止损后停止监控
                        break
                    elif status.action == RiskAction.WARNING:
                        self.logger.warning(status.message)
                    else:
                        self.logger.debug(status.message)

                except Exception as e:
                    self.logger.error(f"风控检查异常: {e}")

                self._stop_event.wait(check_interval)

            self.is_monitoring = False

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info(f"风控监控已启动，检查间隔: {check_interval}秒")

    def stop_monitoring(self):
        """停止风控监控"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.is_monitoring = False
        self.logger.info("风控监控已停止")

    def get_risk_summary(self) -> dict:
        """获取风控摘要"""
        return {
            "total_capital": self.total_capital,
            "max_loss_percent": self.max_loss_percent,
            "max_loss_amount": self.max_loss_amount,
            "has_position": self.current_position is not None,
            "position_direction": self.current_position.direction if self.current_position else None,
            "position_entry_price": self.current_position.entry_price if self.current_position else None,
            "is_monitoring": self.is_monitoring,
        }


class RiskManager:
    """
    风控管理器 - 高级接口

    提供更简洁的风控管理接口
    """

    def __init__(self, total_capital: float):
        self.controller = RiskController(total_capital)
        self._price_history = []

    def open_position(self, direction: str, price: float, quantity: float):
        """开仓"""
        self.controller.set_position(direction, price, quantity)

    def close_position(self):
        """平仓"""
        self.controller.clear_position()

    def check(self, current_price: float) -> RiskStatus:
        """检查风险"""
        self._price_history.append(current_price)
        return self.controller.check_risk(current_price)

    def should_stop_loss(self, current_price: float) -> bool:
        """是否应该止损"""
        status = self.check(current_price)
        return status.action == RiskAction.STOP_LOSS

    def get_pnl(self, current_price: float) -> tuple:
        """获取盈亏"""
        return self.controller.calculate_pnl(current_price)

    def start_auto_monitor(self, price_getter: Callable[[], float],
                          on_stop_loss: Callable = None):
        """启动自动监控"""
        self.controller.on_stop_loss = on_stop_loss
        self.controller.start_monitoring(price_getter)

    def stop_auto_monitor(self):
        """停止自动监控"""
        self.controller.stop_monitoring()
