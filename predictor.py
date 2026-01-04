"""
价格预测模块

通过动量分析、趋势延续、波动模式等方法预测短期价格方向
用于弥补0.5秒行情延迟

注意：预测仅供参考，不保证准确率，请结合其他分析使用
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


@dataclass
class Prediction:
    """预测结果"""
    direction: Direction         # 预测方向
    confidence: float           # 置信度 0-1
    predicted_change: float     # 预测变动幅度
    reason: str                 # 预测依据
    methods_agree: int          # 多少种方法达成一致


class PricePredictor:
    """
    价格预测器

    使用多种方法综合预测短期价格方向：
    1. 动量分析 - 价格变化速度和加速度
    2. 趋势延续 - 趋势惯性
    3. 均值回归 - 超买超卖反转
    4. 波动突破 - 突破后延续
    """

    def __init__(self, lookback: int = 20):
        """
        初始化

        Args:
            lookback: 回溯周期数
        """
        self.lookback = lookback

    def predict(self, prices: List[float]) -> Prediction:
        """
        综合预测

        Args:
            prices: 价格序列（最新价在最后）

        Returns:
            Prediction 预测结果
        """
        if len(prices) < self.lookback:
            return Prediction(
                direction=Direction.NEUTRAL,
                confidence=0,
                predicted_change=0,
                reason="数据不足",
                methods_agree=0
            )

        prices = np.array(prices[-self.lookback:])

        # 各种预测方法
        momentum_pred = self._momentum_prediction(prices)
        trend_pred = self._trend_continuation(prices)
        reversion_pred = self._mean_reversion(prices)
        breakout_pred = self._volatility_breakout(prices)

        # 统计各方向票数
        predictions = [momentum_pred, trend_pred, reversion_pred, breakout_pred]
        up_votes = sum(1 for p in predictions if p[0] == Direction.UP)
        down_votes = sum(1 for p in predictions if p[0] == Direction.DOWN)

        # 确定最终方向
        if up_votes > down_votes and up_votes >= 2:
            direction = Direction.UP
            methods_agree = up_votes
        elif down_votes > up_votes and down_votes >= 2:
            direction = Direction.DOWN
            methods_agree = down_votes
        else:
            direction = Direction.NEUTRAL
            methods_agree = 0

        # 计算综合置信度
        confidences = [p[1] for p in predictions if p[0] == direction]
        avg_confidence = np.mean(confidences) if confidences else 0

        # 预测变动幅度
        changes = [p[2] for p in predictions if p[0] == direction]
        avg_change = np.mean(changes) if changes else 0

        # 生成原因说明
        reasons = []
        if momentum_pred[0] == direction:
            reasons.append("动量")
        if trend_pred[0] == direction:
            reasons.append("趋势")
        if reversion_pred[0] == direction:
            reasons.append("回归")
        if breakout_pred[0] == direction:
            reasons.append("突破")

        return Prediction(
            direction=direction,
            confidence=float(avg_confidence),
            predicted_change=float(avg_change),
            reason=" + ".join(reasons) if reasons else "无明确信号",
            methods_agree=methods_agree
        )

    def _momentum_prediction(self, prices: np.ndarray) -> Tuple[Direction, float, float]:
        """
        动量预测

        分析价格变化的速度和加速度
        速度 = 一阶差分
        加速度 = 二阶差分
        """
        # 计算收益率
        returns = np.diff(prices) / prices[:-1]

        # 短期动量 (最近5个周期)
        short_momentum = np.mean(returns[-5:]) if len(returns) >= 5 else np.mean(returns)

        # 动量加速度
        if len(returns) >= 3:
            acceleration = returns[-1] - returns[-2]
        else:
            acceleration = 0

        # 判断方向
        if short_momentum > 0.0001 and acceleration >= 0:
            direction = Direction.UP
            confidence = min(abs(short_momentum) * 100, 0.8)
        elif short_momentum < -0.0001 and acceleration <= 0:
            direction = Direction.DOWN
            confidence = min(abs(short_momentum) * 100, 0.8)
        else:
            direction = Direction.NEUTRAL
            confidence = 0.3

        predicted_change = short_momentum * prices[-1]

        return (direction, confidence, predicted_change)

    def _trend_continuation(self, prices: np.ndarray) -> Tuple[Direction, float, float]:
        """
        趋势延续预测

        趋势有惯性，短期内大概率延续
        使用线性回归斜率判断趋势
        """
        n = len(prices)
        x = np.arange(n)

        # 线性回归
        slope = np.polyfit(x, prices, 1)[0]

        # 趋势强度 (R²)
        y_pred = slope * x + np.polyfit(x, prices, 1)[1]
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 判断方向
        if slope > 0 and r_squared > 0.5:
            direction = Direction.UP
            confidence = min(r_squared, 0.85)
        elif slope < 0 and r_squared > 0.5:
            direction = Direction.DOWN
            confidence = min(r_squared, 0.85)
        else:
            direction = Direction.NEUTRAL
            confidence = 0.3

        predicted_change = slope  # 预测每周期变动

        return (direction, confidence, predicted_change)

    def _mean_reversion(self, prices: np.ndarray) -> Tuple[Direction, float, float]:
        """
        均值回归预测

        价格偏离均值过远时，有回归倾向
        使用布林带判断
        """
        mean = np.mean(prices)
        std = np.std(prices)

        if std == 0:
            return (Direction.NEUTRAL, 0.3, 0)

        current = prices[-1]
        z_score = (current - mean) / std

        # 超过2个标准差，预测回归
        if z_score > 2:
            direction = Direction.DOWN  # 超买，预测回落
            confidence = min(abs(z_score) / 4, 0.7)
            predicted_change = -(current - mean) * 0.3  # 预测回归30%
        elif z_score < -2:
            direction = Direction.UP  # 超卖，预测反弹
            confidence = min(abs(z_score) / 4, 0.7)
            predicted_change = (mean - current) * 0.3
        else:
            direction = Direction.NEUTRAL
            confidence = 0.3
            predicted_change = 0

        return (direction, confidence, predicted_change)

    def _volatility_breakout(self, prices: np.ndarray) -> Tuple[Direction, float, float]:
        """
        波动突破预测

        价格突破近期高低点后，往往会延续
        """
        if len(prices) < 10:
            return (Direction.NEUTRAL, 0.3, 0)

        current = prices[-1]
        recent_high = np.max(prices[-10:-1])  # 不含当前价
        recent_low = np.min(prices[-10:-1])
        recent_range = recent_high - recent_low

        if recent_range == 0:
            return (Direction.NEUTRAL, 0.3, 0)

        # 判断突破
        if current > recent_high:
            direction = Direction.UP
            breakout_strength = (current - recent_high) / recent_range
            confidence = min(0.5 + breakout_strength, 0.8)
            predicted_change = recent_range * 0.5  # 预测延续50%
        elif current < recent_low:
            direction = Direction.DOWN
            breakout_strength = (recent_low - current) / recent_range
            confidence = min(0.5 + breakout_strength, 0.8)
            predicted_change = -recent_range * 0.5
        else:
            direction = Direction.NEUTRAL
            confidence = 0.3
            predicted_change = 0

        return (direction, confidence, predicted_change)

    def get_quick_prediction(self, prices: List[float]) -> str:
        """
        快速预测（用于实时显示）

        Returns:
            "↑↑" / "↑" / "→" / "↓" / "↓↓"
        """
        pred = self.predict(prices)

        if pred.direction == Direction.UP:
            if pred.confidence > 0.6:
                return "↑↑"
            else:
                return "↑"
        elif pred.direction == Direction.DOWN:
            if pred.confidence > 0.6:
                return "↓↓"
            else:
                return "↓"
        else:
            return "→"


class CrossMarketPredictor:
    """
    跨市场预测器

    利用品种间的相关性进行预测
    例如：美元涨 -> 黄金跌，原油涨 -> 股市涨
    """

    # 品种相关性矩阵 (经验值)
    CORRELATIONS = {
        ("XAUUSD", "USOIL"): 0.3,      # 黄金与原油正相关
        ("XAUUSD", "EURUSD"): 0.5,     # 黄金与欧元正相关
        ("XAUUSD", "DXY"): -0.8,       # 黄金与美元负相关
        ("USOIL", "CADJPY"): 0.4,      # 原油与加元正相关
        ("ETHUSD", "BTCUSD"): 0.9,     # 以太坊与比特币高度正相关
    }

    def __init__(self):
        self.price_history = {}

    def update_price(self, symbol: str, price: float):
        """更新价格"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol].pop(0)

    def predict_from_related(self, target_symbol: str) -> Optional[Tuple[Direction, float, str]]:
        """
        根据相关品种预测目标品种

        Args:
            target_symbol: 目标品种

        Returns:
            (方向, 置信度, 原因) 或 None
        """
        predictions = []

        for (sym1, sym2), corr in self.CORRELATIONS.items():
            related_symbol = None
            if sym1 == target_symbol:
                related_symbol = sym2
            elif sym2 == target_symbol:
                related_symbol = sym1
                corr = corr  # 相关性对称

            if related_symbol and related_symbol in self.price_history:
                prices = self.price_history[related_symbol]
                if len(prices) >= 5:
                    # 计算相关品种的短期方向
                    change = (prices[-1] - prices[-5]) / prices[-5]

                    if change > 0.001:
                        related_direction = Direction.UP
                    elif change < -0.001:
                        related_direction = Direction.DOWN
                    else:
                        continue

                    # 根据相关性推断目标方向
                    if corr > 0:
                        target_direction = related_direction
                    else:
                        target_direction = Direction.DOWN if related_direction == Direction.UP else Direction.UP

                    confidence = abs(corr) * abs(change) * 10
                    predictions.append((target_direction, confidence, f"{related_symbol}联动"))

        if not predictions:
            return None

        # 取置信度最高的预测
        best = max(predictions, key=lambda x: x[1])
        return best


# ==================== 测试 ====================

if __name__ == "__main__":
    import random

    # 生成模拟价格数据
    prices = [100.0]
    for _ in range(50):
        change = random.gauss(0.001, 0.005)  # 随机波动
        prices.append(prices[-1] * (1 + change))

    # 添加趋势
    for _ in range(20):
        prices.append(prices[-1] * 1.002)  # 上涨趋势

    predictor = PricePredictor()
    result = predictor.predict(prices)

    print(f"预测方向: {result.direction.value}")
    print(f"置信度: {result.confidence:.1%}")
    print(f"预测变动: {result.predicted_change:.4f}")
    print(f"依据: {result.reason}")
    print(f"方法一致: {result.methods_agree}/4")
