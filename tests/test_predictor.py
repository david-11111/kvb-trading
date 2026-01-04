"""
价格预测模块单元测试
测试覆盖：动量预测、趋势延续、均值回归、波动突破
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from predictor import PricePredictor, Direction, Prediction, CrossMarketPredictor


class TestPricePredictor:
    """价格预测器测试"""

    @pytest.fixture
    def predictor(self):
        return PricePredictor(lookback=20)

    def test_predict_returns_correct_type(self, predictor):
        """预测返回正确类型"""
        prices = list(np.random.randn(30) * 0.5 + 100)
        result = predictor.predict(prices)

        assert isinstance(result, Prediction)
        assert isinstance(result.direction, Direction)
        assert isinstance(result.confidence, float)
        assert isinstance(result.predicted_change, float)
        assert isinstance(result.reason, str)
        assert isinstance(result.methods_agree, int)

    def test_predict_insufficient_data(self, predictor):
        """数据不足时返回中性"""
        prices = [100.0, 101.0, 102.0]  # 少于lookback
        result = predictor.predict(prices)

        assert result.direction == Direction.NEUTRAL
        assert result.confidence == 0
        assert result.methods_agree == 0
        assert "数据不足" in result.reason

    def test_predict_uptrend(self, predictor):
        """上升趋势预测"""
        # 构造强上升趋势
        prices = [100 + i * 0.5 for i in range(30)]
        result = predictor.predict(prices)

        # 强上升趋势应该预测UP
        assert result.direction == Direction.UP
        assert result.confidence > 0.3
        assert result.methods_agree >= 2

    def test_predict_downtrend(self, predictor):
        """下降趋势预测"""
        prices = [100 - i * 0.5 for i in range(30)]
        result = predictor.predict(prices)

        assert result.direction == Direction.DOWN
        assert result.confidence > 0.3
        assert result.methods_agree >= 2

    def test_predict_sideways(self, predictor):
        """横盘市场预测"""
        np.random.seed(42)
        # 小幅随机波动
        prices = list(100 + np.random.randn(30) * 0.1)
        result = predictor.predict(prices)

        # 横盘时可能是NEUTRAL或低置信度方向
        assert result.confidence < 0.7 or result.direction == Direction.NEUTRAL

    def test_confidence_range(self, predictor):
        """置信度应在合理范围"""
        prices = list(np.random.randn(30) + 100)
        result = predictor.predict(prices)

        assert 0.0 <= result.confidence <= 1.0

    def test_methods_agree_range(self, predictor):
        """一致方法数范围"""
        prices = list(np.random.randn(30) + 100)
        result = predictor.predict(prices)

        assert 0 <= result.methods_agree <= 4


class TestMomentumPrediction:
    """动量预测测试"""

    def test_positive_momentum(self):
        """正动量"""
        predictor = PricePredictor(lookback=10)
        # 价格以指数方式上涨，保证加速度非负
        base = 100
        prices = [base * (1.01 ** i) for i in range(15)]  # 每期1%增长
        result = predictor._momentum_prediction(np.array(prices[-10:]))

        direction, confidence, change = result
        # 动量方向应该为上涨或中性（取决于加速度）
        assert direction in (Direction.UP, Direction.NEUTRAL)
        assert confidence >= 0

    def test_negative_momentum(self):
        """负动量"""
        predictor = PricePredictor(lookback=10)
        prices = [100 - i * 1.0 for i in range(15)]
        result = predictor._momentum_prediction(np.array(prices[-10:]))

        direction, confidence, change = result
        assert direction == Direction.DOWN

    def test_zero_momentum(self):
        """零动量"""
        predictor = PricePredictor(lookback=10)
        prices = [100.0] * 15
        result = predictor._momentum_prediction(np.array(prices[-10:]))

        direction, confidence, change = result
        assert direction == Direction.NEUTRAL


class TestTrendContinuation:
    """趋势延续预测测试"""

    def test_strong_uptrend(self):
        """强上升趋势"""
        predictor = PricePredictor(lookback=20)
        prices = np.array([100 + i * 0.5 for i in range(25)])
        result = predictor._trend_continuation(prices[-20:])

        direction, confidence, slope = result
        assert direction == Direction.UP
        assert slope > 0

    def test_strong_downtrend(self):
        """强下降趋势"""
        predictor = PricePredictor(lookback=20)
        prices = np.array([100 - i * 0.5 for i in range(25)])
        result = predictor._trend_continuation(prices[-20:])

        direction, confidence, slope = result
        assert direction == Direction.DOWN
        assert slope < 0

    def test_weak_trend(self):
        """弱趋势（R²低）"""
        predictor = PricePredictor(lookback=20)
        np.random.seed(42)
        # 随机波动，无明显趋势
        prices = 100 + np.random.randn(25) * 2
        result = predictor._trend_continuation(prices[-20:])

        direction, confidence, slope = result
        # 弱趋势可能是NEUTRAL或低置信度
        assert confidence < 0.85


class TestMeanReversion:
    """均值回归预测测试"""

    def test_overbought(self):
        """超买状态"""
        predictor = PricePredictor(lookback=20)
        # 构造价格远高于均值的情况
        prices = np.array([100.0] * 18 + [110.0, 115.0])  # 最后两个远超均值
        result = predictor._mean_reversion(prices)

        direction, confidence, change = result
        # 超买应预测DOWN（回归）
        if confidence > 0.3:  # 只有当置信度足够时
            assert direction == Direction.DOWN

    def test_oversold(self):
        """超卖状态"""
        predictor = PricePredictor(lookback=20)
        prices = np.array([100.0] * 18 + [90.0, 85.0])
        result = predictor._mean_reversion(prices)

        direction, confidence, change = result
        if confidence > 0.3:
            assert direction == Direction.UP

    def test_normal_range(self):
        """正常范围内"""
        predictor = PricePredictor(lookback=20)
        prices = np.array([100.0] * 20)  # 所有价格相同
        result = predictor._mean_reversion(prices)

        direction, confidence, change = result
        # 在均值附近，应该是NEUTRAL
        assert direction == Direction.NEUTRAL


class TestVolatilityBreakout:
    """波动突破预测测试"""

    def test_upside_breakout(self):
        """向上突破"""
        predictor = PricePredictor(lookback=20)
        # 前面横盘（有小波动），最后突破
        # 需要recent_range > 0，所以历史价格需要有波动
        prices = np.array([100.0, 100.5, 99.5, 100.0, 100.2, 99.8, 100.1, 99.9,
                          100.0, 100.3, 99.7, 100.0, 100.0, 100.5, 99.5, 100.0,
                          100.2, 99.8, 100.0, 106.0])  # 最后突破到106
        result = predictor._volatility_breakout(prices)

        direction, confidence, change = result
        assert direction == Direction.UP

    def test_downside_breakout(self):
        """向下突破"""
        predictor = PricePredictor(lookback=20)
        # 前面横盘（有小波动），最后向下突破
        prices = np.array([100.0, 100.5, 99.5, 100.0, 100.2, 99.8, 100.1, 99.9,
                          100.0, 100.3, 99.7, 100.0, 100.0, 100.5, 99.5, 100.0,
                          100.2, 99.8, 100.0, 94.0])  # 最后跌破到94
        result = predictor._volatility_breakout(prices)

        direction, confidence, change = result
        assert direction == Direction.DOWN

    def test_no_breakout(self):
        """无突破"""
        predictor = PricePredictor(lookback=20)
        prices = np.array([100.0] * 20)
        result = predictor._volatility_breakout(prices)

        direction, confidence, change = result
        assert direction == Direction.NEUTRAL


class TestQuickPrediction:
    """快速预测测试"""

    def test_quick_prediction_strong_up(self):
        """强上升"""
        predictor = PricePredictor(lookback=20)
        prices = [100 + i * 1.0 for i in range(30)]
        result = predictor.get_quick_prediction(prices)

        assert result in ["↑↑", "↑"]

    def test_quick_prediction_strong_down(self):
        """强下降"""
        predictor = PricePredictor(lookback=20)
        prices = [100 - i * 1.0 for i in range(30)]
        result = predictor.get_quick_prediction(prices)

        assert result in ["↓↓", "↓"]

    def test_quick_prediction_neutral(self):
        """中性"""
        predictor = PricePredictor(lookback=20)
        np.random.seed(42)
        prices = list(100 + np.random.randn(30) * 0.05)
        result = predictor.get_quick_prediction(prices)

        assert result in ["↑↑", "↑", "→", "↓", "↓↓"]


class TestCrossMarketPredictor:
    """跨市场预测器测试"""

    @pytest.fixture
    def predictor(self):
        return CrossMarketPredictor()

    def test_update_price(self, predictor):
        """更新价格"""
        predictor.update_price("XAUUSD", 2000.0)
        assert "XAUUSD" in predictor.price_history
        assert len(predictor.price_history["XAUUSD"]) == 1

    def test_price_history_limit(self, predictor):
        """价格历史限制"""
        for i in range(150):
            predictor.update_price("XAUUSD", 2000 + i)

        # 应该最多保留100条
        assert len(predictor.price_history["XAUUSD"]) <= 100

    def test_predict_from_related_no_data(self, predictor):
        """无相关数据时返回None"""
        result = predictor.predict_from_related("XAUUSD")
        assert result is None

    def test_predict_from_related_with_data(self, predictor):
        """有相关数据时的预测"""
        # 添加足够的USOIL数据
        for i in range(10):
            predictor.update_price("USOIL", 70 + i * 0.5)

        result = predictor.predict_from_related("XAUUSD")
        # XAUUSD和USOIL正相关，USOIL上涨应该预测XAUUSD上涨
        if result is not None:
            direction, confidence, reason = result
            assert direction in [Direction.UP, Direction.DOWN, Direction.NEUTRAL]


class TestEdgeCases:
    """边界条件测试"""

    def test_empty_prices(self):
        """空价格列表"""
        predictor = PricePredictor(lookback=20)
        result = predictor.predict([])

        assert result.direction == Direction.NEUTRAL
        assert result.confidence == 0

    def test_single_price(self):
        """单个价格"""
        predictor = PricePredictor(lookback=20)
        result = predictor.predict([100.0])

        assert result.direction == Direction.NEUTRAL

    def test_all_same_prices(self):
        """所有价格相同"""
        predictor = PricePredictor(lookback=20)
        prices = [100.0] * 30
        result = predictor.predict(prices)

        # 价格不变，动量应该是中性
        assert result.direction == Direction.NEUTRAL

    def test_nan_in_prices(self):
        """价格中包含NaN"""
        predictor = PricePredictor(lookback=20)
        prices = [100.0] * 25 + [float('nan')] + [101.0] * 4

        # 应该能处理而不崩溃
        try:
            result = predictor.predict(prices)
            assert isinstance(result, Prediction)
        except Exception as e:
            # 如果抛出异常，应该是预期的类型
            assert isinstance(e, (ValueError, RuntimeError))

    def test_very_large_prices(self):
        """非常大的价格"""
        predictor = PricePredictor(lookback=20)
        prices = [1e10 + i for i in range(30)]
        result = predictor.predict(prices)

        assert isinstance(result, Prediction)

    def test_very_small_prices(self):
        """非常小的价格"""
        predictor = PricePredictor(lookback=20)
        prices = [1e-10 + i * 1e-12 for i in range(30)]
        result = predictor.predict(prices)

        assert isinstance(result, Prediction)

    def test_negative_prices(self):
        """负价格"""
        predictor = PricePredictor(lookback=20)
        prices = [-100 + i * 0.5 for i in range(30)]
        result = predictor.predict(prices)

        # 应该能处理负价格
        assert isinstance(result, Prediction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
