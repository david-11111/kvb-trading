"""
技术指标计算模块
包含MACD、ATR、布林带等指标的计算
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

from config import MACD_CONFIG, PHILOSOPHY_CONFIG


class CrossSignal(Enum):
    """交叉信号类型"""
    NONE = "none"           # 无信号
    GOLDEN = "golden"       # 金叉（DIF上穿DEA）
    DEATH = "death"         # 死叉（DIF下穿DEA）


@dataclass
class MACDResult:
    """MACD计算结果"""
    dif: np.ndarray          # DIF线（快线-慢线）
    dea: np.ndarray          # DEA线（DIF的EMA）
    histogram: np.ndarray    # MACD柱（DIF-DEA)*2
    signal: CrossSignal      # 当前交叉信号
    signal_strength: float   # 信号强度 (0-1)


@dataclass
class ATRResult:
    """ATR计算结果"""
    atr: np.ndarray              # ATR值
    current_percentile: float    # 当前ATR在历史中的分位数


@dataclass
class BollingerResult:
    """布林带计算结果"""
    upper: np.ndarray        # 上轨
    middle: np.ndarray       # 中轨
    lower: np.ndarray        # 下轨
    bandwidth: np.ndarray    # 带宽
    percent_b: float         # 当前价格在布林带中的位置


@dataclass
class KDJResult:
    """KDJ计算结果"""
    k: np.ndarray           # K线（快线）
    d: np.ndarray           # D线（慢线）
    j: np.ndarray           # J线（超买超卖）
    signal: CrossSignal     # 当前交叉信号
    is_oversold: bool       # 是否超卖（J<20）
    is_overbought: bool     # 是否超买（J>80）


class Indicators:
    """技术指标计算类"""

    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """
        计算指数移动平均线(EMA)

        Args:
            data: 价格数据
            period: EMA周期

        Returns:
            EMA数组
        """
        if len(data) < period:
            return np.full(len(data), np.nan)

        alpha = 2 / (period + 1)
        ema_values = np.zeros(len(data))

        # 第一个EMA值使用SMA
        ema_values[period - 1] = np.mean(data[:period])

        # 计算后续EMA
        for i in range(period, len(data)):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i - 1]

        # 前面的值设为NaN
        ema_values[:period - 1] = np.nan

        return ema_values

    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """
        计算简单移动平均线(SMA)

        Args:
            data: 价格数据
            period: SMA周期

        Returns:
            SMA数组
        """
        if len(data) < period:
            return np.full(len(data), np.nan)

        sma_values = np.zeros(len(data))
        sma_values[:period - 1] = np.nan

        for i in range(period - 1, len(data)):
            sma_values[i] = np.mean(data[i - period + 1:i + 1])

        return sma_values

    @classmethod
    def macd(cls, close_prices: np.ndarray,
             fast_period: int = None,
             slow_period: int = None,
             signal_period: int = None) -> MACDResult:
        """
        计算MACD指标

        Args:
            close_prices: 收盘价数组
            fast_period: 快速EMA周期，默认12
            slow_period: 慢速EMA周期，默认26
            signal_period: 信号线周期，默认9

        Returns:
            MACDResult对象，包含DIF、DEA、MACD柱、交叉信号
        """
        fast_period = fast_period or MACD_CONFIG["fast_period"]
        slow_period = slow_period or MACD_CONFIG["slow_period"]
        signal_period = signal_period or MACD_CONFIG["signal_period"]

        # 计算快慢EMA
        ema_fast = cls.ema(close_prices, fast_period)
        ema_slow = cls.ema(close_prices, slow_period)

        # DIF = 快速EMA - 慢速EMA
        dif = ema_fast - ema_slow

        # DEA = DIF的EMA（信号线）
        dea = cls.ema(dif[~np.isnan(dif)], signal_period)

        # 对齐DEA数组长度
        full_dea = np.full(len(close_prices), np.nan)
        start_idx = len(close_prices) - len(dea)
        full_dea[start_idx:] = dea
        dea = full_dea

        # MACD柱 = (DIF - DEA) * 2
        histogram = (dif - dea) * 2

        # 检测交叉信号
        signal = cls._detect_cross_signal(dif, dea)

        # 计算信号强度
        signal_strength = cls._calculate_signal_strength(dif, dea, histogram)

        return MACDResult(
            dif=dif,
            dea=dea,
            histogram=histogram,
            signal=signal,
            signal_strength=signal_strength
        )

    @staticmethod
    def _detect_cross_signal(dif: np.ndarray, dea: np.ndarray) -> CrossSignal:
        """
        检测MACD金叉/死叉信号

        金叉：DIF从下方上穿DEA
        死叉：DIF从上方下穿DEA
        """
        # 至少需要2个有效数据点
        valid_mask = ~(np.isnan(dif) | np.isnan(dea))
        if np.sum(valid_mask) < 2:
            return CrossSignal.NONE

        # 获取最后两个有效点
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return CrossSignal.NONE

        prev_idx, curr_idx = valid_indices[-2], valid_indices[-1]

        prev_diff = dif[prev_idx] - dea[prev_idx]
        curr_diff = dif[curr_idx] - dea[curr_idx]

        # 金叉：前一个DIF<DEA，当前DIF>DEA
        if prev_diff < 0 and curr_diff > 0:
            return CrossSignal.GOLDEN

        # 死叉：前一个DIF>DEA，当前DIF<DEA
        if prev_diff > 0 and curr_diff < 0:
            return CrossSignal.DEATH

        return CrossSignal.NONE

    @staticmethod
    def _calculate_signal_strength(dif: np.ndarray, dea: np.ndarray,
                                   histogram: np.ndarray) -> float:
        """
        计算信号强度 (0-1)

        考虑因素：
        1. DIF与DEA的差距
        2. MACD柱的大小
        3. 动量变化
        """
        valid_hist = histogram[~np.isnan(histogram)]
        if len(valid_hist) < 2:
            return 0.0

        # 当前柱与历史对比
        current_hist = abs(valid_hist[-1])
        hist_std = np.std(valid_hist)

        if hist_std == 0:
            return 0.5

        # 归一化到0-1
        strength = min(current_hist / (3 * hist_std), 1.0)

        return strength

    @classmethod
    def kdj(cls, high: np.ndarray, low: np.ndarray, close: np.ndarray,
            n: int = 9, m1: int = 3, m2: int = 3) -> KDJResult:
        """
        计算KDJ指标（随机指标）
        
        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            n: RSV周期，默认9
            m1: K线平滑周期，默认3
            m2: D线平滑周期，默认3
            
        Returns:
            KDJResult对象
        """
        length = len(close)
        
        # 计算RSV (Raw Stochastic Value)
        rsv = np.zeros(length)
        for i in range(n - 1, length):
            highest = np.max(high[i - n + 1:i + 1])
            lowest = np.min(low[i - n + 1:i + 1])
            if highest != lowest:
                rsv[i] = (close[i] - lowest) / (highest - lowest) * 100
            else:
                rsv[i] = 50
        
        # 计算K值 (RSV的EMA)
        k = np.zeros(length)
        k[n - 1] = 50  # 初始值
        for i in range(n, length):
            k[i] = (m1 - 1) / m1 * k[i - 1] + 1 / m1 * rsv[i]
        
        # 计算D值 (K的EMA)
        d = np.zeros(length)
        d[n - 1] = 50  # 初始值
        for i in range(n, length):
            d[i] = (m2 - 1) / m2 * d[i - 1] + 1 / m2 * k[i]
        
        # 计算J值
        j = 3 * k - 2 * d
        
        # 前面的无效值设为NaN
        k[:n - 1] = np.nan
        d[:n - 1] = np.nan
        j[:n - 1] = np.nan
        
        # 检测交叉信号
        signal = cls._detect_kdj_cross(k, d)
        
        # 判断超买超卖
        current_j = j[-1] if not np.isnan(j[-1]) else 50
        is_oversold = current_j < 20
        is_overbought = current_j > 80
        
        return KDJResult(
            k=k,
            d=d,
            j=j,
            signal=signal,
            is_oversold=is_oversold,
            is_overbought=is_overbought
        )
    
    @staticmethod
    def _detect_kdj_cross(k: np.ndarray, d: np.ndarray) -> CrossSignal:
        """
        检测KDJ金叉/死叉信号
        
        金叉：K从下方上穿D
        死叉：K从上方下穿D
        """
        valid_mask = ~(np.isnan(k) | np.isnan(d))
        if np.sum(valid_mask) < 2:
            return CrossSignal.NONE
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return CrossSignal.NONE
        
        prev_idx, curr_idx = valid_indices[-2], valid_indices[-1]
        
        prev_diff = k[prev_idx] - d[prev_idx]
        curr_diff = k[curr_idx] - d[curr_idx]
        
        # 金叉：K上穿D
        if prev_diff < 0 and curr_diff > 0:
            return CrossSignal.GOLDEN
        
        # 死叉：K下穿D
        if prev_diff > 0 and curr_diff < 0:
            return CrossSignal.DEATH
        
        return CrossSignal.NONE

    @classmethod
    def atr(cls, high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = None) -> ATRResult:
        """
        计算平均真实波幅(ATR)

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            period: ATR周期，默认14

        Returns:
            ATRResult对象
        """
        period = period or PHILOSOPHY_CONFIG["atr_period"]

        # 计算True Range
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]

        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # ATR = TR的EMA
        atr_values = cls.ema(tr, period)

        # 计算当前ATR的历史分位数
        valid_atr = atr_values[~np.isnan(atr_values)]
        if len(valid_atr) > 0:
            current_atr = valid_atr[-1]
            percentile = np.sum(valid_atr < current_atr) / len(valid_atr)
        else:
            percentile = 0.5

        return ATRResult(atr=atr_values, current_percentile=percentile)

    @classmethod
    def bollinger_bands(cls, close: np.ndarray, period: int = None,
                        num_std: float = None) -> BollingerResult:
        """
        计算布林带

        Args:
            close: 收盘价数组
            period: 周期，默认20
            num_std: 标准差倍数，默认2

        Returns:
            BollingerResult对象
        """
        period = period or PHILOSOPHY_CONFIG["bb_period"]
        num_std = num_std or PHILOSOPHY_CONFIG["bb_std"]

        # 中轨 = SMA
        middle = cls.sma(close, period)

        # 计算标准差
        std = np.zeros(len(close))
        std[:period - 1] = np.nan
        for i in range(period - 1, len(close)):
            std[i] = np.std(close[i - period + 1:i + 1])

        # 上轨和下轨
        upper = middle + num_std * std
        lower = middle - num_std * std

        # 带宽 = (上轨 - 下轨) / 中轨
        bandwidth = (upper - lower) / middle

        # %B = (当前价 - 下轨) / (上轨 - 下轨)
        valid_idx = ~(np.isnan(upper) | np.isnan(lower))
        if np.any(valid_idx):
            last_valid = np.where(valid_idx)[0][-1]
            current_price = close[last_valid]
            percent_b = (current_price - lower[last_valid]) / (upper[last_valid] - lower[last_valid])
        else:
            percent_b = 0.5

        return BollingerResult(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b
        )

    @classmethod
    def trend_strength(cls, close: np.ndarray, period: int = 20) -> float:
        """
        计算趋势强度 (0-1)

        使用价格与移动平均线的偏离程度来衡量

        Args:
            close: 收盘价数组
            period: 周期

        Returns:
            趋势强度，0表示无趋势，1表示强趋势
        """
        if len(close) < period:
            return 0.5

        ma = cls.sma(close, period)
        valid_ma = ma[~np.isnan(ma)]

        if len(valid_ma) == 0:
            return 0.5

        # 计算价格偏离MA的程度
        last_idx = len(close) - 1
        deviation = abs(close[last_idx] - ma[last_idx]) / ma[last_idx]

        # 计算方向一致性（价格持续在MA上方或下方）
        recent_prices = close[-period:]
        recent_ma = ma[-period:]
        valid_mask = ~np.isnan(recent_ma)

        if np.sum(valid_mask) > 0:
            above_ma = np.sum(recent_prices[valid_mask] > recent_ma[valid_mask])
            consistency = abs(above_ma / np.sum(valid_mask) - 0.5) * 2
        else:
            consistency = 0

        # 综合评分
        strength = min((deviation * 10 + consistency) / 2, 1.0)

        return strength

    @classmethod
    def kdj(cls, high: np.ndarray, low: np.ndarray, close: np.ndarray,
            n: int = 9, m1: int = 3, m2: int = 3) -> KDJResult:
        """
        计算KDJ指标（随机指标）
        
        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            n: RSV周期，默认9
            m1: K线平滑周期，默认3
            m2: D线平滑周期，默认3
            
        Returns:
            KDJResult对象
        """
        length = len(close)
        
        # 计算RSV (Raw Stochastic Value)
        rsv = np.zeros(length)
        for i in range(n - 1, length):
            highest = np.max(high[i - n + 1:i + 1])
            lowest = np.min(low[i - n + 1:i + 1])
            if highest != lowest:
                rsv[i] = (close[i] - lowest) / (highest - lowest) * 100
            else:
                rsv[i] = 50
        
        # 计算K值 (RSV的EMA)
        k = np.zeros(length)
        k[n - 1] = 50  # 初始值
        for i in range(n, length):
            k[i] = (m1 - 1) / m1 * k[i - 1] + 1 / m1 * rsv[i]
        
        # 计算D值 (K的EMA)
        d = np.zeros(length)
        d[n - 1] = 50  # 初始值
        for i in range(n, length):
            d[i] = (m2 - 1) / m2 * d[i - 1] + 1 / m2 * k[i]
        
        # 计算J值
        j = 3 * k - 2 * d
        
        # 前面的无效值设为NaN
        k[:n - 1] = np.nan
        d[:n - 1] = np.nan
        j[:n - 1] = np.nan
        
        # 检测交叉信号
        signal = cls._detect_kdj_cross(k, d)
        
        # 判断超买超卖
        current_j = j[-1] if not np.isnan(j[-1]) else 50
        is_oversold = current_j < 20
        is_overbought = current_j > 80
        
        return KDJResult(
            k=k,
            d=d,
            j=j,
            signal=signal,
            is_oversold=is_oversold,
            is_overbought=is_overbought
        )
    
    @staticmethod
    def _detect_kdj_cross(k: np.ndarray, d: np.ndarray) -> CrossSignal:
        """
        检测KDJ金叉/死叉信号
        
        金叉：K从下方上穿D
        死叉：K从上方下穿D
        """
        valid_mask = ~(np.isnan(k) | np.isnan(d))
        if np.sum(valid_mask) < 2:
            return CrossSignal.NONE
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return CrossSignal.NONE
        
        prev_idx, curr_idx = valid_indices[-2], valid_indices[-1]
        
        prev_diff = k[prev_idx] - d[prev_idx]
        curr_diff = k[curr_idx] - d[curr_idx]
        
        # 金叉：K上穿D
        if prev_diff < 0 and curr_diff > 0:
            return CrossSignal.GOLDEN
        
        # 死叉：K下穿D
        if prev_diff > 0 and curr_diff < 0:
            return CrossSignal.DEATH
        
        return CrossSignal.NONE
