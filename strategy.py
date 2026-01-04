"""
交易策略模块

开仓策略: MACD+KDJ共振
加仓策略: 盈利5%后 + 布林带下轨反弹
止损策略: 亏损达到5%强制平仓
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from indicators import Indicators, CrossSignal, MACDResult, KDJResult, BollingerResult


class SignalType(Enum):
    """信号类型"""
    NONE = "none"           # 无信号
    BUY = "buy"             # 买入信号
    SELL = "sell"           # 卖出信号
    ADD_LONG = "add_long"   # 加仓做多
    ADD_SHORT = "add_short" # 加仓做空
    CLOSE = "close"         # 平仓信号


@dataclass
class Signal:
    """交易信号"""
    type: SignalType
    strength: float         # 信号强度 0-1
    reason: str             # 信号原因
    price: float            # 触发价格
    timestamp: float        # 时间戳


@dataclass
class PositionState:
    """持仓状态"""
    direction: str = None       # "long" / "short" / None
    entry_price: float = 0.0    # 开仓均价
    quantity: float = 0.0       # 持仓量
    total_cost: float = 0.0     # 总成本
    unrealized_pnl: float = 0.0 # 浮动盈亏
    pnl_percent: float = 0.0    # 盈亏百分比
    add_count: int = 0          # 加仓次数
    direction_confirmed: bool = False  # 方向是否确认（盈利5%）


class TradingStrategy:
    """
    交易策略类

    开仓条件: MACD金叉 + KDJ金叉 同时满足
    加仓条件: 盈利≥5% + 价格回调到布林带下轨反弹
    止损条件: 亏损≥5%
    """

    def __init__(self,
                 total_capital: float,
                 max_loss_percent: float = 0.05,
                 profit_confirm_percent: float = 0.05,
                 max_add_count: int = 3):
        """
        初始化策略

        Args:
            total_capital: 总资金
            max_loss_percent: 最大亏损比例 (默认5%)
            profit_confirm_percent: 方向确认盈利比例 (默认5%)
            max_add_count: 最大加仓次数
        """
        self.total_capital = total_capital
        self.max_loss_percent = max_loss_percent
        self.profit_confirm_percent = profit_confirm_percent
        self.max_add_count = max_add_count

        self.position = PositionState()
        self.signals_history: List[Signal] = []

        # 布林带反弹状态追踪
        self._touched_lower_band = False
        self._last_percent_b = 0.5

        self.logger = logging.getLogger("TradingStrategy")

    def update_capital(self, capital: float):
        """更新总资金"""
        self.total_capital = capital

    def analyze(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                current_price: float) -> Signal:
        """
        分析市场并生成信号

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            current_price: 当前价格

        Returns:
            Signal对象
        """
        # 计算技术指标
        macd = Indicators.macd(close)
        kdj = Indicators.kdj(high, low, close)
        bb = Indicators.bollinger_bands(close)

        # 更新持仓盈亏
        self._update_position_pnl(current_price)

        # 1. 检查是否需要止损
        if self._should_stop_loss():
            return self._create_signal(
                SignalType.CLOSE,
                1.0,
                f"止损平仓！亏损达到{abs(self.position.pnl_percent)*100:.1f}%",
                current_price
            )

        # 2. 有持仓时，检查是否可以加仓
        if self.position.direction:
            add_signal = self._check_add_position(macd, kdj, bb, current_price)
            if add_signal:
                return add_signal

        # 3. 无持仓时，检查是否可以开仓
        else:
            open_signal = self._check_open_position(macd, kdj, current_price)
            if open_signal:
                return open_signal

        return self._create_signal(SignalType.NONE, 0, "无交易信号", current_price)

    def _check_open_position(self, macd: MACDResult, kdj: KDJResult,
                             current_price: float) -> Optional[Signal]:
        """
        检查开仓条件: MACD金叉 + KDJ金叉 共振
        """
        # MACD金叉 + KDJ金叉 = 做多信号
        if macd.signal == CrossSignal.GOLDEN and kdj.signal == CrossSignal.GOLDEN:
            # 额外条件：KDJ不在超买区（J<80）
            if not kdj.is_overbought:
                strength = (macd.signal_strength + 0.5) / 1.5  # 综合强度
                return self._create_signal(
                    SignalType.BUY,
                    strength,
                    "MACD金叉 + KDJ金叉共振，做多信号",
                    current_price
                )

        # MACD死叉 + KDJ死叉 = 做空信号
        if macd.signal == CrossSignal.DEATH and kdj.signal == CrossSignal.DEATH:
            # 额外条件：KDJ不在超卖区（J>20）
            if not kdj.is_oversold:
                strength = (macd.signal_strength + 0.5) / 1.5
                return self._create_signal(
                    SignalType.SELL,
                    strength,
                    "MACD死叉 + KDJ死叉共振，做空信号",
                    current_price
                )

        return None

    def _check_add_position(self, macd: MACDResult, kdj: KDJResult,
                            bb: BollingerResult, current_price: float) -> Optional[Signal]:
        """
        检查加仓条件: 盈利≥5% + 布林带下轨反弹
        """
        # 条件1: 已达到加仓上限
        if self.position.add_count >= self.max_add_count:
            return None

        # 条件2: 必须盈利≥5%才能加仓（方向确认）
        if self.position.pnl_percent < self.profit_confirm_percent:
            return None

        # 标记方向已确认
        if not self.position.direction_confirmed:
            self.position.direction_confirmed = True
            self.logger.info(f"方向确认！盈利达到{self.position.pnl_percent*100:.1f}%")

        # 条件3: 布林带下轨反弹
        # 做多时：价格触及下轨后反弹（percent_b从<0.2回升到>0.2）
        if self.position.direction == "long":
            # 追踪是否触及下轨
            if bb.percent_b < 0.2:
                self._touched_lower_band = True

            # 触及下轨后开始反弹
            if self._touched_lower_band and bb.percent_b > 0.2 and bb.percent_b > self._last_percent_b:
                self._touched_lower_band = False  # 重置
                return self._create_signal(
                    SignalType.ADD_LONG,
                    0.7,
                    f"布林带下轨反弹加仓（第{self.position.add_count + 1}次）",
                    current_price
                )

        # 做空时：价格触及上轨后回落（percent_b从>0.8回落到<0.8）
        elif self.position.direction == "short":
            if bb.percent_b > 0.8:
                self._touched_lower_band = True  # 复用变量

            if self._touched_lower_band and bb.percent_b < 0.8 and bb.percent_b < self._last_percent_b:
                self._touched_lower_band = False
                return self._create_signal(
                    SignalType.ADD_SHORT,
                    0.7,
                    f"布林带上轨回落加仓（第{self.position.add_count + 1}次）",
                    current_price
                )

        self._last_percent_b = bb.percent_b
        return None

    def _should_stop_loss(self) -> bool:
        """检查是否应该止损"""
        if not self.position.direction:
            return False
        return self.position.pnl_percent <= -self.max_loss_percent

    def _update_position_pnl(self, current_price: float):
        """更新持仓盈亏"""
        if not self.position.direction:
            return

        price_change = current_price - self.position.entry_price

        if self.position.direction == "long":
            self.position.unrealized_pnl = price_change * self.position.quantity
        else:  # short
            self.position.unrealized_pnl = -price_change * self.position.quantity

        if self.position.total_cost > 0:
            self.position.pnl_percent = self.position.unrealized_pnl / self.position.total_cost

    def _create_signal(self, signal_type: SignalType, strength: float,
                       reason: str, price: float) -> Signal:
        """创建信号"""
        signal = Signal(
            type=signal_type,
            strength=strength,
            reason=reason,
            price=price,
            timestamp=time.time()
        )

        if signal_type != SignalType.NONE:
            self.signals_history.append(signal)
            self.logger.info(f"[{signal_type.value}] {reason} @ {price}")

        return signal

    # ==================== 持仓管理 ====================

    def open_position(self, direction: str, price: float, quantity: float, cost: float):
        """
        记录开仓

        Args:
            direction: "long" 或 "short"
            price: 开仓价格
            quantity: 数量
            cost: 占用保证金
        """
        self.position = PositionState(
            direction=direction,
            entry_price=price,
            quantity=quantity,
            total_cost=cost,
            add_count=0,
            direction_confirmed=False
        )
        self._touched_lower_band = False
        self.logger.info(f"开仓: {direction} @ {price}, 数量: {quantity}")

    def add_position(self, price: float, quantity: float, cost: float):
        """
        记录加仓

        Args:
            price: 加仓价格
            quantity: 加仓数量
            cost: 新增保证金
        """
        if not self.position.direction:
            return

        # 计算新的平均价格
        total_value = self.position.entry_price * self.position.quantity + price * quantity
        new_quantity = self.position.quantity + quantity
        new_avg_price = total_value / new_quantity

        self.position.entry_price = new_avg_price
        self.position.quantity = new_quantity
        self.position.total_cost += cost
        self.position.add_count += 1

        self.logger.info(f"加仓: {price}, 新均价: {new_avg_price:.3f}, 总量: {new_quantity}")

    def close_position(self, price: float) -> float:
        """
        记录平仓

        Args:
            price: 平仓价格

        Returns:
            实现盈亏
        """
        if not self.position.direction:
            return 0.0

        self._update_position_pnl(price)
        realized_pnl = self.position.unrealized_pnl

        self.logger.info(f"平仓: @ {price}, 盈亏: {realized_pnl:.2f}")

        # 重置持仓
        self.position = PositionState()
        self._touched_lower_band = False
        self._last_percent_b = 0.5

        return realized_pnl

    def get_position_summary(self) -> dict:
        """获取持仓摘要"""
        return {
            "direction": self.position.direction,
            "entry_price": self.position.entry_price,
            "quantity": self.position.quantity,
            "unrealized_pnl": self.position.unrealized_pnl,
            "pnl_percent": self.position.pnl_percent * 100,
            "add_count": self.position.add_count,
            "direction_confirmed": self.position.direction_confirmed,
            "max_add_count": self.max_add_count,
        }

    def get_strategy_summary(self) -> dict:
        """获取策略摘要"""
        return {
            "total_capital": self.total_capital,
            "max_loss_percent": self.max_loss_percent * 100,
            "profit_confirm_percent": self.profit_confirm_percent * 100,
            "total_signals": len(self.signals_history),
            "position": self.get_position_summary(),
        }
