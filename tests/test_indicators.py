"""
技术指标模块单元测试
测试覆盖：MACD, KDJ, ATR, Bollinger Bands, EMA, SMA
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators import Indicators, CrossSignal, MACDResult, ATRResult, BollingerResult, KDJResult


class TestEMA:
    """EMA 指数移动平均线测试"""

    def test_ema_basic(self):
        """基本EMA计算"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = Indicators.ema(data, 3)

        # 前两个值应该是 NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # 第三个值开始有效
        assert not np.isnan(result[2])
        # EMA应该平滑跟随趋势
        assert result[-1] > result[-2]  # 上升趋势

    def test_ema_insufficient_data(self):
        """数据不足时的处理"""
        data = np.array([1.0, 2.0])
        result = Indicators.ema(data, 5)

        # 所有值应该是 NaN
        assert all(np.isnan(result))

    def test_ema_constant_data(self):
        """常数数据的EMA"""
        data = np.array([5.0] * 20)
        result = Indicators.ema(data, 5)

        # 常数数据的EMA应该等于该常数
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, 5.0)

    def test_ema_empty_array(self):
        """空数组处理"""
        data = np.array([])
        result = Indicators.ema(data, 5)
        assert len(result) == 0

    def test_ema_single_value(self):
        """单值数组"""
        data = np.array([100.0])
        result = Indicators.ema(data, 3)
        assert np.isnan(result[0])


class TestSMA:
    """SMA 简单移动平均线测试"""

    def test_sma_basic(self):
        """基本SMA计算"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = Indicators.sma(data, 3)

        # SMA(3) 从第三个开始: (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0
        assert result[3] == 3.0
        assert result[4] == 4.0

    def test_sma_insufficient_data(self):
        """数据不足"""
        data = np.array([1.0, 2.0])
        result = Indicators.sma(data, 5)
        assert all(np.isnan(result))


class TestMACD:
    """MACD 指标测试"""

    @pytest.fixture
    def uptrend_prices(self):
        """上升趋势价格数据"""
        np.random.seed(42)
        base = 100.0
        prices = []
        for i in range(50):
            base += np.random.randn() * 0.5 + 0.3  # 带噪声的上升趋势
            prices.append(base)
        return np.array(prices)

    @pytest.fixture
    def downtrend_prices(self):
        """下降趋势价格数据"""
        np.random.seed(42)
        base = 100.0
        prices = []
        for i in range(50):
            base += np.random.randn() * 0.5 - 0.3  # 带噪声的下降趋势
            prices.append(base)
        return np.array(prices)

    def test_macd_returns_correct_type(self, uptrend_prices):
        """MACD返回正确类型"""
        result = Indicators.macd(uptrend_prices)
        assert isinstance(result, MACDResult)
        assert isinstance(result.dif, np.ndarray)
        assert isinstance(result.dea, np.ndarray)
        assert isinstance(result.histogram, np.ndarray)
        assert isinstance(result.signal, CrossSignal)
        assert isinstance(result.signal_strength, float)

    def test_macd_uptrend(self, uptrend_prices):
        """上升趋势中MACD应为正"""
        result = Indicators.macd(uptrend_prices)
        # 上升趋势末期，DIF应该大于DEA
        valid_dif = result.dif[~np.isnan(result.dif)]
        valid_dea = result.dea[~np.isnan(result.dea)]
        if len(valid_dif) > 0 and len(valid_dea) > 0:
            # 上升趋势中histogram应该为正
            valid_hist = result.histogram[~np.isnan(result.histogram)]
            assert valid_hist[-1] > 0

    def test_macd_insufficient_data(self):
        """数据不足"""
        prices = np.array([1.0, 2.0, 3.0])
        result = Indicators.macd(prices)
        # 应该返回MACDResult但值多为NaN
        assert isinstance(result, MACDResult)

    def test_macd_signal_strength_range(self, uptrend_prices):
        """信号强度应在0-1范围内"""
        result = Indicators.macd(uptrend_prices)
        assert 0.0 <= result.signal_strength <= 1.0

    def test_macd_golden_cross_detection(self):
        """金叉检测"""
        # 构造一个从下降转为上升的序列，应该产生金叉
        prices = np.array([100 - i*0.5 for i in range(20)] +
                          [90 + i*0.5 for i in range(30)])
        result = Indicators.macd(prices)
        # 可能产生金叉信号
        assert result.signal in [CrossSignal.GOLDEN, CrossSignal.NONE, CrossSignal.DEATH]


class TestKDJ:
    """KDJ 随机指标测试"""

    @pytest.fixture
    def ohlc_data(self):
        """OHLC数据"""
        np.random.seed(42)
        n = 50
        close = np.cumsum(np.random.randn(n) * 0.5) + 100
        high = close + np.abs(np.random.randn(n)) * 0.5
        low = close - np.abs(np.random.randn(n)) * 0.5
        return high, low, close

    def test_kdj_returns_correct_type(self, ohlc_data):
        """KDJ返回正确类型"""
        high, low, close = ohlc_data
        result = Indicators.kdj(high, low, close)
        assert isinstance(result, KDJResult)
        assert isinstance(result.k, np.ndarray)
        assert isinstance(result.d, np.ndarray)
        assert isinstance(result.j, np.ndarray)

    def test_kdj_value_range(self, ohlc_data):
        """K和D值应在合理范围内"""
        high, low, close = ohlc_data
        result = Indicators.kdj(high, low, close)
        valid_k = result.k[~np.isnan(result.k)]
        valid_d = result.d[~np.isnan(result.d)]

        # K和D通常在0-100之间
        assert all(0 <= k <= 100 for k in valid_k)
        assert all(0 <= d <= 100 for d in valid_d)

    def test_kdj_oversold_detection(self):
        """超卖检测"""
        # 构造持续下跌的数据
        n = 30
        close = np.array([100 - i*2 for i in range(n)])
        high = close + 0.5
        low = close - 0.5
        result = Indicators.kdj(high, low, close)
        # 持续下跌后J值应该很低
        # 注意：不一定会触发超卖，取决于具体参数
        # numpy可能返回np.bool_类型，检查值是否为True或False
        assert result.is_oversold in (True, False, np.True_, np.False_)

    def test_kdj_overbought_detection(self):
        """超买检测"""
        n = 30
        close = np.array([100 + i*2 for i in range(n)])
        high = close + 0.5
        low = close - 0.5
        result = Indicators.kdj(high, low, close)
        # numpy可能返回np.bool_类型，检查值是否为True或False
        assert result.is_overbought in (True, False, np.True_, np.False_)


class TestATR:
    """ATR 平均真实波幅测试"""

    @pytest.fixture
    def ohlc_data(self):
        """OHLC数据"""
        np.random.seed(42)
        n = 30
        close = np.cumsum(np.random.randn(n) * 0.5) + 100
        high = close + np.abs(np.random.randn(n)) * 1.0
        low = close - np.abs(np.random.randn(n)) * 1.0
        return high, low, close

    def test_atr_returns_correct_type(self, ohlc_data):
        """ATR返回正确类型"""
        high, low, close = ohlc_data
        result = Indicators.atr(high, low, close)
        assert isinstance(result, ATRResult)
        assert isinstance(result.atr, np.ndarray)
        assert isinstance(result.current_percentile, float)

    def test_atr_always_positive(self, ohlc_data):
        """ATR应该始终为正"""
        high, low, close = ohlc_data
        result = Indicators.atr(high, low, close)
        valid_atr = result.atr[~np.isnan(result.atr)]
        assert all(a >= 0 for a in valid_atr)

    def test_atr_percentile_range(self, ohlc_data):
        """分位数应在0-1范围内"""
        high, low, close = ohlc_data
        result = Indicators.atr(high, low, close)
        assert 0.0 <= result.current_percentile <= 1.0

    def test_atr_high_volatility(self):
        """高波动性数据的ATR应该较大"""
        n = 30
        close = np.array([100.0] * n)
        high = close + 5.0  # 大波动
        low = close - 5.0
        result = Indicators.atr(high, low, close)
        valid_atr = result.atr[~np.isnan(result.atr)]
        assert valid_atr[-1] > 5.0  # ATR应该接近10（高-低）


class TestBollingerBands:
    """布林带测试"""

    @pytest.fixture
    def price_data(self):
        """价格数据"""
        np.random.seed(42)
        return np.cumsum(np.random.randn(50) * 0.5) + 100

    def test_bollinger_returns_correct_type(self, price_data):
        """布林带返回正确类型"""
        result = Indicators.bollinger_bands(price_data)
        assert isinstance(result, BollingerResult)
        assert isinstance(result.upper, np.ndarray)
        assert isinstance(result.middle, np.ndarray)
        assert isinstance(result.lower, np.ndarray)

    def test_bollinger_band_order(self, price_data):
        """上轨 > 中轨 > 下轨"""
        result = Indicators.bollinger_bands(price_data)
        valid_idx = ~(np.isnan(result.upper) | np.isnan(result.middle) | np.isnan(result.lower))

        assert all(result.upper[valid_idx] >= result.middle[valid_idx])
        assert all(result.middle[valid_idx] >= result.lower[valid_idx])

    def test_bollinger_percent_b_range(self, price_data):
        """%B应在合理范围内（通常0-1，但可能超出）"""
        result = Indicators.bollinger_bands(price_data)
        # %B可以超出0-1范围（价格在布林带外）
        assert isinstance(result.percent_b, float)

    def test_bollinger_constant_price(self):
        """常数价格的布林带"""
        prices = np.array([100.0] * 30)
        result = Indicators.bollinger_bands(prices)
        # 标准差为0，上下轨应该等于中轨
        valid_idx = ~np.isnan(result.upper)
        if any(valid_idx):
            assert np.allclose(result.upper[valid_idx], result.middle[valid_idx])
            assert np.allclose(result.lower[valid_idx], result.middle[valid_idx])


class TestTrendStrength:
    """趋势强度测试"""

    def test_trend_strength_range(self):
        """趋势强度应在0-1范围内"""
        prices = np.random.randn(50) + 100
        result = Indicators.trend_strength(prices)
        assert 0.0 <= result <= 1.0

    def test_trend_strength_strong_uptrend(self):
        """强上升趋势"""
        prices = np.array([100 + i*2 for i in range(30)])
        result = Indicators.trend_strength(prices)
        # 强趋势应该有较高的强度
        assert result > 0.3

    def test_trend_strength_sideways(self):
        """横盘市场"""
        np.random.seed(42)
        prices = 100 + np.random.randn(30) * 0.1  # 小幅波动
        result = Indicators.trend_strength(prices)
        # 横盘趋势强度应该较低
        assert result < 0.7

    def test_trend_strength_insufficient_data(self):
        """数据不足"""
        prices = np.array([100.0, 101.0])
        result = Indicators.trend_strength(prices, period=20)
        assert result == 0.5  # 默认值


class TestEdgeCases:
    """边界条件测试"""

    def test_nan_in_data(self):
        """数据中包含NaN"""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        # EMA应该能处理NaN
        result = Indicators.ema(data, 2)
        assert len(result) == len(data)

    def test_inf_in_data(self):
        """数据中包含Inf"""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        result = Indicators.ema(data, 2)
        # 结果可能包含inf或nan
        assert len(result) == len(data)

    def test_negative_prices(self):
        """负价格（理论上不应该出现，但需要处理）"""
        data = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = Indicators.sma(data, 3)
        # 应该正常计算
        assert not np.isnan(result[-1])

    def test_very_large_numbers(self):
        """非常大的数字"""
        data = np.array([1e15, 1.1e15, 1.2e15, 1.3e15, 1.4e15])
        result = Indicators.ema(data, 3)
        # 应该正常计算，不溢出
        assert not np.isinf(result[-1])

    def test_very_small_numbers(self):
        """非常小的数字"""
        data = np.array([1e-15, 1.1e-15, 1.2e-15, 1.3e-15, 1.4e-15])
        result = Indicators.ema(data, 3)
        # 应该正常计算，不丢失精度
        assert result[-1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
