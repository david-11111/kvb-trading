"""
投资哲学体系模块
核心功能：判断市场处于震荡区域还是单边区域
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from config import PHILOSOPHY_CONFIG, TRADE_CONFIG
from indicators import Indicators, MACDResult, CrossSignal


class MarketState(Enum):
    """市场状态"""
    OSCILLATION = "oscillation"  # 震荡市场
    TRENDING_UP = "trending_up"  # 单边上涨
    TRENDING_DOWN = "trending_down"  # 单边下跌
    UNCERTAIN = "uncertain"  # 不确定


class TradeAction(Enum):
    """交易动作"""
    BUY = "buy"          # 做多
    SELL = "sell"        # 做空
    CLOSE_LONG = "close_long"   # 平多
    CLOSE_SHORT = "close_short"  # 平空
    HOLD = "hold"        # 持有/观望


@dataclass
class MarketAnalysis:
    """市场分析结果"""
    state: MarketState           # 市场状态
    confidence: float            # 置信度 (0-1)
    atr_score: float            # ATR评分
    bollinger_score: float      # 布林带评分
    macd_score: float           # MACD评分
    trend_score: float          # 趋势评分
    reasoning: str              # 分析理由


@dataclass
class TradeAdvice:
    """交易建议"""
    action: TradeAction          # 建议动作
    market_state: MarketState    # 市场状态
    signal: CrossSignal          # MACD信号
    confidence: float            # 建议置信度
    take_profit: float          # 建议止盈比例
    position_size: float        # 建议仓位比例
    reasoning: str              # 建议理由


class InvestmentPhilosophy:
    """
    投资哲学体系

    核心理念：
    1. 市场有两种状态：震荡和单边
    2. 震荡市场：高抛低吸，小止盈
    3. 单边市场：顺势而为，大止盈
    4. 通过多维度指标综合判断市场状态
    """

    def __init__(self):
        self.config = PHILOSOPHY_CONFIG
        self.trade_config = TRADE_CONFIG
        self.indicators = Indicators()

    def analyze_market(self, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> MarketAnalysis:
        """
        分析市场状态

        综合考虑：
        1. ATR（波动率）
        2. 布林带位置
        3. MACD柱状态
        4. 趋势强度

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组

        Returns:
            MarketAnalysis对象
        """
        # 1. ATR分析 - 判断波动率水平
        atr_result = self.indicators.atr(high, low, close)
        atr_score = self._score_atr(atr_result.current_percentile)

        # 2. 布林带分析 - 判断价格位置
        bb_result = self.indicators.bollinger_bands(close)
        bollinger_score = self._score_bollinger(bb_result.percent_b, bb_result.bandwidth)

        # 3. MACD分析 - 判断动能状态
        macd_result = self.indicators.macd(close)
        macd_score = self._score_macd(macd_result)

        # 4. 趋势强度分析
        trend_strength = self.indicators.trend_strength(close)
        trend_score = trend_strength

        # 综合评分
        weights = self.config["weights"]
        total_score = (
            atr_score * weights["atr"] +
            bollinger_score * weights["bollinger"] +
            macd_score * weights["macd_histogram"] +
            trend_score * weights["trend_strength"]
        )

        # 判断市场状态
        state, confidence = self._determine_state(total_score, trend_strength, close)

        # 生成分析理由
        reasoning = self._generate_reasoning(
            atr_score, bollinger_score, macd_score, trend_score, state
        )

        return MarketAnalysis(
            state=state,
            confidence=confidence,
            atr_score=atr_score,
            bollinger_score=bollinger_score,
            macd_score=macd_score,
            trend_score=trend_score,
            reasoning=reasoning
        )

    def _score_atr(self, percentile: float) -> float:
        """
        ATR评分

        高ATR -> 高分 -> 倾向单边
        低ATR -> 低分 -> 倾向震荡
        """
        return percentile

    def _score_bollinger(self, percent_b: float, bandwidth: np.ndarray) -> float:
        """
        布林带评分

        价格在上下轨附近 -> 高分 -> 倾向单边
        价格在中轨附近 -> 低分 -> 倾向震荡
        带宽收窄 -> 低分 -> 震荡蓄势
        """
        # 位置评分：距离中轨越远分数越高
        position_score = abs(percent_b - 0.5) * 2

        # 带宽评分
        valid_bw = bandwidth[~np.isnan(bandwidth)]
        if len(valid_bw) > 0:
            current_bw = valid_bw[-1]
            bw_percentile = np.sum(valid_bw < current_bw) / len(valid_bw)
        else:
            bw_percentile = 0.5

        # 综合评分
        return (position_score + bw_percentile) / 2

    def _score_macd(self, macd_result: MACDResult) -> float:
        """
        MACD评分

        柱状图持续放大 -> 高分 -> 单边趋势
        柱状图在0轴附近 -> 低分 -> 震荡
        """
        valid_hist = macd_result.histogram[~np.isnan(macd_result.histogram)]
        if len(valid_hist) < 5:
            return 0.5

        # 当前柱大小
        current_hist = abs(valid_hist[-1])
        hist_std = np.std(valid_hist)

        if hist_std == 0:
            return 0.5

        # 柱大小评分
        size_score = min(current_hist / (2 * hist_std), 1.0)

        # 柱方向一致性（最近5根柱是否同向）
        recent_hist = valid_hist[-5:]
        same_direction = np.all(recent_hist > 0) or np.all(recent_hist < 0)
        direction_score = 1.0 if same_direction else 0.3

        # 柱是否在增大
        if len(valid_hist) >= 3:
            expanding = abs(valid_hist[-1]) > abs(valid_hist[-2]) > abs(valid_hist[-3])
            expand_score = 1.0 if expanding else 0.5
        else:
            expand_score = 0.5

        return (size_score + direction_score + expand_score) / 3

    def _determine_state(self, total_score: float, trend_strength: float,
                         close: np.ndarray) -> tuple:
        """
        确定市场状态

        Returns:
            (MarketState, confidence)
        """
        # 判断趋势方向
        if len(close) >= 20:
            ma20 = np.mean(close[-20:])
            current_price = close[-1]
            is_above_ma = current_price > ma20
        else:
            is_above_ma = None

        # 根据综合评分判断
        if total_score < 0.35:
            state = MarketState.OSCILLATION
            confidence = 1 - total_score  # 分数越低，震荡置信度越高
        elif total_score > 0.65:
            if is_above_ma:
                state = MarketState.TRENDING_UP
            else:
                state = MarketState.TRENDING_DOWN
            confidence = total_score
        else:
            state = MarketState.UNCERTAIN
            confidence = 1 - abs(total_score - 0.5) * 2

        return state, confidence

    def _generate_reasoning(self, atr_score: float, bollinger_score: float,
                           macd_score: float, trend_score: float,
                           state: MarketState) -> str:
        """生成分析理由"""
        reasons = []

        # ATR分析
        if atr_score < 0.3:
            reasons.append("波动率处于历史低位，市场较为平静")
        elif atr_score > 0.7:
            reasons.append("波动率处于历史高位，市场活跃")
        else:
            reasons.append("波动率处于正常水平")

        # 布林带分析
        if bollinger_score < 0.3:
            reasons.append("价格在布林带中轨附近，带宽收窄")
        elif bollinger_score > 0.7:
            reasons.append("价格接近布林带边缘，带宽扩张")

        # MACD分析
        if macd_score < 0.3:
            reasons.append("MACD柱在零轴附近波动，动能不足")
        elif macd_score > 0.7:
            reasons.append("MACD柱持续放大，动能充足")

        # 趋势分析
        if trend_score < 0.3:
            reasons.append("价格围绕均线波动，无明显趋势")
        elif trend_score > 0.7:
            reasons.append("价格与均线偏离较大，趋势明显")

        # 结论
        state_desc = {
            MarketState.OSCILLATION: "综合判断：当前处于震荡市场",
            MarketState.TRENDING_UP: "综合判断：当前处于单边上涨趋势",
            MarketState.TRENDING_DOWN: "综合判断：当前处于单边下跌趋势",
            MarketState.UNCERTAIN: "综合判断：市场状态不明确，建议观望"
        }
        reasons.append(state_desc[state])

        return "；".join(reasons)

    def generate_advice(self, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray,
                        current_position: Optional[str] = None) -> TradeAdvice:
        """
        生成交易建议

        投资哲学决策循环：
        1. 检测MACD信号（金叉/死叉）
        2. 分析市场状态
        3. 根据市场状态调整交易策略

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            current_position: 当前持仓 ("long", "short", None)

        Returns:
            TradeAdvice对象
        """
        # 计算MACD
        macd_result = self.indicators.macd(close)

        # 分析市场状态
        analysis = self.analyze_market(high, low, close)

        # 如果没有交叉信号，维持当前状态
        if macd_result.signal == CrossSignal.NONE:
            return TradeAdvice(
                action=TradeAction.HOLD,
                market_state=analysis.state,
                signal=CrossSignal.NONE,
                confidence=0.0,
                take_profit=0.0,
                position_size=0.0,
                reasoning="无MACD交叉信号，建议观望"
            )

        # 根据市场状态和信号生成建议
        if analysis.state == MarketState.OSCILLATION:
            return self._oscillation_strategy(macd_result, analysis, current_position)
        elif analysis.state in [MarketState.TRENDING_UP, MarketState.TRENDING_DOWN]:
            return self._trending_strategy(macd_result, analysis, current_position)
        else:
            return TradeAdvice(
                action=TradeAction.HOLD,
                market_state=analysis.state,
                signal=macd_result.signal,
                confidence=analysis.confidence,
                take_profit=0.0,
                position_size=0.0,
                reasoning="市场状态不明确，建议观望等待确认"
            )

    def _oscillation_strategy(self, macd_result: MACDResult,
                              analysis: MarketAnalysis,
                              current_position: Optional[str]) -> TradeAdvice:
        """
        震荡市场策略

        特点：
        - 高抛低吸
        - 小止盈（2%）
        - 轻仓位（30%）
        """
        config = self.trade_config["oscillation"]

        if macd_result.signal == CrossSignal.GOLDEN:
            # 金叉做多
            if current_position == "short":
                action = TradeAction.CLOSE_SHORT
                reasoning = "震荡市场出现金叉，平空头仓位"
            else:
                action = TradeAction.BUY
                reasoning = "震荡市场出现金叉，轻仓做多，设置小止盈"

        elif macd_result.signal == CrossSignal.DEATH:
            # 死叉做空
            if current_position == "long":
                action = TradeAction.CLOSE_LONG
                reasoning = "震荡市场出现死叉，平多头仓位"
            else:
                action = TradeAction.SELL
                reasoning = "震荡市场出现死叉，轻仓做空，设置小止盈"
        else:
            action = TradeAction.HOLD
            reasoning = "等待明确信号"

        return TradeAdvice(
            action=action,
            market_state=MarketState.OSCILLATION,
            signal=macd_result.signal,
            confidence=analysis.confidence * macd_result.signal_strength,
            take_profit=config["take_profit_percent"],
            position_size=config["position_size"],
            reasoning=f"{analysis.reasoning}；{reasoning}"
        )

    def _trending_strategy(self, macd_result: MACDResult,
                          analysis: MarketAnalysis,
                          current_position: Optional[str]) -> TradeAdvice:
        """
        单边市场策略

        特点：
        - 顺势而为
        - 大止盈（5%）
        - 重仓位（50%）
        """
        config = self.trade_config["trending"]
        is_uptrend = analysis.state == MarketState.TRENDING_UP

        if macd_result.signal == CrossSignal.GOLDEN:
            if is_uptrend:
                # 上涨趋势中的金叉，强烈做多
                if current_position == "short":
                    action = TradeAction.CLOSE_SHORT
                    reasoning = "上涨趋势中出现金叉，平空头仓位"
                else:
                    action = TradeAction.BUY
                    reasoning = "上涨趋势中出现金叉，重仓做多，让利润奔跑"
            else:
                # 下跌趋势中的金叉，可能是反弹
                if current_position == "short":
                    action = TradeAction.CLOSE_SHORT
                    reasoning = "下跌趋势中出现金叉反弹信号，减仓观望"
                else:
                    action = TradeAction.HOLD
                    reasoning = "下跌趋势中的金叉可能是反弹，观望为主"

        elif macd_result.signal == CrossSignal.DEATH:
            if not is_uptrend:
                # 下跌趋势中的死叉，强烈做空
                if current_position == "long":
                    action = TradeAction.CLOSE_LONG
                    reasoning = "下跌趋势中出现死叉，平多头仓位"
                else:
                    action = TradeAction.SELL
                    reasoning = "下跌趋势中出现死叉，重仓做空，让利润奔跑"
            else:
                # 上涨趋势中的死叉，可能是回调
                if current_position == "long":
                    action = TradeAction.CLOSE_LONG
                    reasoning = "上涨趋势中出现死叉回调信号，减仓观望"
                else:
                    action = TradeAction.HOLD
                    reasoning = "上涨趋势中的死叉可能是回调，观望为主"
        else:
            action = TradeAction.HOLD
            reasoning = "等待明确信号"

        return TradeAdvice(
            action=action,
            market_state=analysis.state,
            signal=macd_result.signal,
            confidence=analysis.confidence * macd_result.signal_strength,
            take_profit=config["take_profit_percent"],
            position_size=config["position_size"],
            reasoning=f"{analysis.reasoning}；{reasoning}"
        )

    def philosophical_inquiry(self, high: np.ndarray, low: np.ndarray,
                              close: np.ndarray) -> Dict:
        """
        哲学问询循环

        这是投资哲学体系的核心：通过一系列问题来决策

        Returns:
            完整的问询结果字典
        """
        result = {
            "questions": [],
            "answers": [],
            "conclusion": None
        }

        # 问题1：市场波动如何？
        atr_result = self.indicators.atr(high, low, close)
        q1 = "市场波动率如何？"
        if atr_result.current_percentile < 0.3:
            a1 = "波动率偏低，市场平静"
            volatility_state = "low"
        elif atr_result.current_percentile > 0.7:
            a1 = "波动率偏高，市场活跃"
            volatility_state = "high"
        else:
            a1 = "波动率正常"
            volatility_state = "normal"
        result["questions"].append(q1)
        result["answers"].append(a1)

        # 问题2：价格位置如何？
        bb_result = self.indicators.bollinger_bands(close)
        q2 = "价格相对于布林带的位置？"
        if bb_result.percent_b < 0.2:
            a2 = "价格接近下轨，处于超卖区域"
            price_position = "oversold"
        elif bb_result.percent_b > 0.8:
            a2 = "价格接近上轨，处于超买区域"
            price_position = "overbought"
        else:
            a2 = "价格在中间区域"
            price_position = "middle"
        result["questions"].append(q2)
        result["answers"].append(a2)

        # 问题3：动能状态如何？
        macd_result = self.indicators.macd(close)
        q3 = "MACD动能状态？"
        valid_hist = macd_result.histogram[~np.isnan(macd_result.histogram)]
        if len(valid_hist) > 0:
            if valid_hist[-1] > 0:
                a3 = "动能为正，多头占优"
                momentum = "bullish"
            else:
                a3 = "动能为负，空头占优"
                momentum = "bearish"
        else:
            a3 = "动能不明确"
            momentum = "neutral"
        result["questions"].append(q3)
        result["answers"].append(a3)

        # 问题4：趋势强度如何？
        trend = self.indicators.trend_strength(close)
        q4 = "趋势强度如何？"
        if trend < 0.3:
            a4 = "趋势较弱，可能是震荡"
            trend_state = "weak"
        elif trend > 0.7:
            a4 = "趋势较强，可能是单边"
            trend_state = "strong"
        else:
            a4 = "趋势中等"
            trend_state = "moderate"
        result["questions"].append(q4)
        result["answers"].append(a4)

        # 综合结论
        oscillation_score = 0
        if volatility_state == "low":
            oscillation_score += 1
        if price_position == "middle":
            oscillation_score += 1
        if trend_state == "weak":
            oscillation_score += 1

        if oscillation_score >= 2:
            result["conclusion"] = {
                "state": "OSCILLATION",
                "description": "市场处于震荡区域",
                "strategy": "建议高抛低吸，轻仓操作，设置较小止盈"
            }
        else:
            if momentum == "bullish" and price_position != "overbought":
                result["conclusion"] = {
                    "state": "TRENDING_UP",
                    "description": "市场处于上涨趋势",
                    "strategy": "建议顺势做多，可适当重仓，设置较大止盈"
                }
            elif momentum == "bearish" and price_position != "oversold":
                result["conclusion"] = {
                    "state": "TRENDING_DOWN",
                    "description": "市场处于下跌趋势",
                    "strategy": "建议顺势做空，可适当重仓，设置较大止盈"
                }
            else:
                result["conclusion"] = {
                    "state": "UNCERTAIN",
                    "description": "市场状态不明确",
                    "strategy": "建议观望，等待更明确的信号"
                }

        return result
