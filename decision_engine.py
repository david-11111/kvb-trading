"""
信号决策引擎
整合所有模块，实现自动化交易决策
"""

import logging
import time
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

from indicators import Indicators, CrossSignal
from philosophy import InvestmentPhilosophy, MarketState, TradeAction, TradeAdvice
from risk_control import RiskManager, RiskAction, RiskStatus


class DecisionResult(Enum):
    """决策结果"""
    NO_ACTION = "no_action"      # 无操作
    BUY = "buy"                  # 买入
    SELL = "sell"               # 卖出
    CLOSE_LONG = "close_long"   # 平多
    CLOSE_SHORT = "close_short" # 平空
    STOP_LOSS = "stop_loss"     # 止损


@dataclass
class Decision:
    """决策详情"""
    result: DecisionResult       # 决策结果
    confidence: float            # 置信度 (0-1)
    market_state: str           # 市场状态
    macd_signal: str            # MACD信号
    take_profit: float          # 建议止盈
    position_size: float        # 建议仓位
    reasoning: str              # 决策理由
    risk_status: Optional[RiskStatus] = None  # 风控状态


class DecisionEngine:
    """
    决策引擎

    工作流程：
    1. 获取价格数据
    2. 计算MACD指标
    3. 检测金叉/死叉信号
    4. 通过投资哲学体系分析市场状态
    5. 检查风控状态
    6. 生成交易决策
    """

    def __init__(self, initial_capital: float = 10000):
        """
        初始化决策引擎

        Args:
            initial_capital: 初始资金
        """
        self.indicators = Indicators()
        self.philosophy = InvestmentPhilosophy()
        self.risk_manager = RiskManager(initial_capital)

        self.current_position: Optional[str] = None  # "long" / "short" / None

        # 回调函数
        self.on_decision: Optional[Callable[[Decision], None]] = None

        self.logger = logging.getLogger("DecisionEngine")

    def process(self, high, low, close, current_price: float) -> Decision:
        """
        处理数据并生成决策

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            current_price: 当前价格

        Returns:
            Decision对象
        """
        # 1. 首先检查风控
        risk_status = self._check_risk(current_price)
        if risk_status and risk_status.action == RiskAction.STOP_LOSS:
            return self._create_stop_loss_decision(risk_status)

        # 2. 计算MACD指标
        macd_result = self.indicators.macd(close)

        # 3. 检查是否有交叉信号
        if macd_result.signal == CrossSignal.NONE:
            return Decision(
                result=DecisionResult.NO_ACTION,
                confidence=0.0,
                market_state="N/A",
                macd_signal="无信号",
                take_profit=0,
                position_size=0,
                reasoning="无MACD交叉信号，继续观望",
                risk_status=risk_status
            )

        # 4. 有信号时，进入哲学问询循环
        self.logger.info(f"检测到MACD{'金叉' if macd_result.signal == CrossSignal.GOLDEN else '死叉'}信号")

        # 执行哲学问询
        inquiry_result = self.philosophy.philosophical_inquiry(high, low, close)

        # 打印问询过程
        self._log_inquiry(inquiry_result)

        # 5. 生成交易建议
        advice = self.philosophy.generate_advice(high, low, close, self.current_position)

        # 6. 转换为决策
        decision = self._advice_to_decision(advice, risk_status)

        # 触发回调
        if self.on_decision:
            self.on_decision(decision)

        return decision

    def _check_risk(self, current_price: float) -> Optional[RiskStatus]:
        """检查风控状态"""
        if self.current_position is None:
            return None

        return self.risk_manager.check(current_price)

    def _create_stop_loss_decision(self, risk_status: RiskStatus) -> Decision:
        """创建止损决策"""
        if self.current_position == "long":
            result = DecisionResult.CLOSE_LONG
        else:
            result = DecisionResult.CLOSE_SHORT

        return Decision(
            result=DecisionResult.STOP_LOSS,
            confidence=1.0,
            market_state="风控触发",
            macd_signal="N/A",
            take_profit=0,
            position_size=0,
            reasoning=f"触发止损！{risk_status.message}",
            risk_status=risk_status
        )

    def _advice_to_decision(self, advice: TradeAdvice,
                           risk_status: Optional[RiskStatus]) -> Decision:
        """将交易建议转换为决策"""
        # 映射动作
        action_map = {
            TradeAction.BUY: DecisionResult.BUY,
            TradeAction.SELL: DecisionResult.SELL,
            TradeAction.CLOSE_LONG: DecisionResult.CLOSE_LONG,
            TradeAction.CLOSE_SHORT: DecisionResult.CLOSE_SHORT,
            TradeAction.HOLD: DecisionResult.NO_ACTION,
        }

        result = action_map.get(advice.action, DecisionResult.NO_ACTION)

        # 映射市场状态
        state_map = {
            MarketState.OSCILLATION: "震荡",
            MarketState.TRENDING_UP: "上涨趋势",
            MarketState.TRENDING_DOWN: "下跌趋势",
            MarketState.UNCERTAIN: "不确定",
        }

        # 映射信号
        signal_map = {
            CrossSignal.GOLDEN: "金叉",
            CrossSignal.DEATH: "死叉",
            CrossSignal.NONE: "无信号",
        }

        return Decision(
            result=result,
            confidence=advice.confidence,
            market_state=state_map.get(advice.market_state, "未知"),
            macd_signal=signal_map.get(advice.signal, "未知"),
            take_profit=advice.take_profit,
            position_size=advice.position_size,
            reasoning=advice.reasoning,
            risk_status=risk_status
        )

    def _log_inquiry(self, inquiry_result: dict):
        """记录哲学问询过程"""
        self.logger.info("=" * 50)
        self.logger.info("【投资哲学问询循环】")
        for q, a in zip(inquiry_result["questions"], inquiry_result["answers"]):
            self.logger.info(f"Q: {q}")
            self.logger.info(f"A: {a}")
        if inquiry_result["conclusion"]:
            self.logger.info("-" * 30)
            self.logger.info(f"结论: {inquiry_result['conclusion']['description']}")
            self.logger.info(f"策略: {inquiry_result['conclusion']['strategy']}")
        self.logger.info("=" * 50)

    def update_position(self, direction: Optional[str], entry_price: float = 0):
        """
        更新持仓状态

        Args:
            direction: "long" / "short" / None
            entry_price: 开仓价格
        """
        self.current_position = direction

        if direction and entry_price > 0:
            # 计算数量（简化处理，实际应该根据仓位比例计算）
            quantity = 1.0
            self.risk_manager.open_position(direction, entry_price, quantity)
        else:
            self.risk_manager.close_position()

    def get_status(self) -> dict:
        """获取引擎状态"""
        return {
            "has_position": self.current_position is not None,
            "position_direction": self.current_position,
            "risk_summary": self.risk_manager.controller.get_risk_summary()
        }


class TradingBrain:
    """
    交易大脑 - 顶层接口

    整合所有功能，提供简洁的接口供主程序调用
    """

    def __init__(self, initial_capital: float = 10000):
        """初始化交易大脑"""
        self.engine = DecisionEngine(initial_capital)
        self.initial_capital = initial_capital

        self.last_decision: Optional[Decision] = None
        self.decision_history = []

        self.logger = logging.getLogger("TradingBrain")

    def think(self, high, low, close, current_price: float) -> Decision:
        """
        思考并做出决策

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            current_price: 当前价格

        Returns:
            Decision对象
        """
        decision = self.engine.process(high, low, close, current_price)

        self.last_decision = decision
        self.decision_history.append({
            "timestamp": time.time(),
            "price": current_price,
            "decision": decision.result.value,
            "confidence": decision.confidence
        })

        # 记录决策
        self._log_decision(decision, current_price)

        return decision

    def _log_decision(self, decision: Decision, price: float):
        """记录决策"""
        if decision.result == DecisionResult.NO_ACTION:
            self.logger.debug(f"价格: {price:.4f} | 决策: 观望 | {decision.reasoning}")
        else:
            self.logger.info(f"""
╔════════════════════════════════════════════════════════════╗
║ 交易决策                                                    ║
╠════════════════════════════════════════════════════════════╣
║ 当前价格: {price:<10.4f}                                    ║
║ 决策结果: {decision.result.value:<12}                       ║
║ 市场状态: {decision.market_state:<12}                       ║
║ MACD信号: {decision.macd_signal:<12}                        ║
║ 置信度:   {decision.confidence*100:.1f}%                    ║
║ 建议止盈: {decision.take_profit*100:.1f}%                   ║
║ 建议仓位: {decision.position_size*100:.0f}%                 ║
╠════════════════════════════════════════════════════════════╣
║ 分析理由:                                                   ║
║ {decision.reasoning[:55]:<55} ║
╚════════════════════════════════════════════════════════════╝
""")

    def notify_trade_executed(self, direction: Optional[str], price: float):
        """
        通知交易已执行

        Args:
            direction: "long" / "short" / None (平仓时为None)
            price: 成交价格
        """
        self.engine.update_position(direction, price)

    def should_take_profit(self, current_price: float, entry_price: float,
                          take_profit_percent: float) -> bool:
        """
        检查是否应该止盈

        Args:
            current_price: 当前价格
            entry_price: 入场价格
            take_profit_percent: 止盈比例

        Returns:
            是否应该止盈
        """
        if self.engine.current_position == "long":
            return current_price >= entry_price * (1 + take_profit_percent)
        elif self.engine.current_position == "short":
            return current_price <= entry_price * (1 - take_profit_percent)
        return False

    def get_summary(self) -> dict:
        """获取大脑摘要"""
        return {
            "total_decisions": len(self.decision_history),
            "last_decision": self.last_decision.result.value if self.last_decision else None,
            "engine_status": self.engine.get_status()
        }
