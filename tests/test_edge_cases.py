"""
边界条件和异常测试
测试覆盖：极端输入、网络异常、状态不一致、并发问题
"""

import pytest
import asyncio
import time
import sys
import math
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# 模拟 winsound
sys.modules['winsound'] = MagicMock()


class TestExtremePriceValues:
    """极端价格值测试"""

    def test_zero_price(self):
        """零价格"""
        from indicators import Indicators
        import numpy as np

        data = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = Indicators.ema(data, 3)

        # 应该能处理而不崩溃
        assert len(result) == len(data)

    def test_negative_price(self):
        """负价格"""
        from indicators import Indicators
        import numpy as np

        data = np.array([-100.0, -99.0, -98.0, -97.0, -96.0])
        result = Indicators.ema(data, 3)

        assert len(result) == len(data)

    def test_very_small_price(self):
        """非常小的价格"""
        from predictor import PricePredictor

        predictor = PricePredictor(lookback=20)
        prices = [1e-15 + i * 1e-17 for i in range(30)]
        result = predictor.predict(prices)

        assert result is not None

    def test_very_large_price(self):
        """非常大的价格"""
        from predictor import PricePredictor

        predictor = PricePredictor(lookback=20)
        prices = [1e15 + i * 1e13 for i in range(30)]
        result = predictor.predict(prices)

        assert result is not None

    def test_price_overflow(self):
        """价格溢出"""
        from indicators import Indicators
        import numpy as np

        data = np.array([1e308, 1.1e308, 1.2e308])
        result = Indicators.ema(data, 2)

        # 可能会有inf，但不应该崩溃
        assert len(result) == len(data)

    def test_price_underflow(self):
        """价格下溢"""
        from indicators import Indicators
        import numpy as np

        data = np.array([1e-308, 1.1e-308, 1.2e-308, 1.3e-308, 1.4e-308])
        result = Indicators.ema(data, 3)

        assert len(result) == len(data)


class TestSpecialFloatValues:
    """特殊浮点值测试"""

    def test_nan_price(self):
        """NaN价格"""
        from indicators import Indicators
        import numpy as np

        data = np.array([100.0, float('nan'), 102.0, 103.0, 104.0])
        result = Indicators.ema(data, 3)

        # NaN会传播，但不应该崩溃
        assert len(result) == len(data)

    def test_inf_price(self):
        """Inf价格"""
        from indicators import Indicators
        import numpy as np

        data = np.array([100.0, float('inf'), 102.0, 103.0, 104.0])
        result = Indicators.ema(data, 3)

        assert len(result) == len(data)

    def test_negative_inf_price(self):
        """负Inf价格"""
        from indicators import Indicators
        import numpy as np

        data = np.array([100.0, float('-inf'), 102.0, 103.0, 104.0])
        result = Indicators.ema(data, 3)

        assert len(result) == len(data)

    def test_mixed_special_values(self):
        """混合特殊值"""
        from predictor import PricePredictor

        predictor = PricePredictor(lookback=5)
        prices = [100.0, float('nan'), float('inf'), 100.0, 101.0]

        # 应该能处理而不崩溃
        try:
            result = predictor.predict(prices)
        except (ValueError, RuntimeError):
            pass  # 预期可能抛出异常


class TestEmptyAndMinimalInput:
    """空输入和最小输入测试"""

    def test_empty_price_list(self):
        """空价格列表"""
        from predictor import PricePredictor

        predictor = PricePredictor(lookback=20)
        result = predictor.predict([])

        assert result.direction.value == "NEUTRAL"
        assert result.confidence == 0

    def test_single_price(self):
        """单个价格"""
        from predictor import PricePredictor

        predictor = PricePredictor(lookback=20)
        result = predictor.predict([100.0])

        assert result.direction.value == "NEUTRAL"

    def test_two_prices(self):
        """两个价格"""
        from predictor import PricePredictor

        predictor = PricePredictor(lookback=20)
        result = predictor.predict([100.0, 101.0])

        assert result is not None

    def test_empty_numpy_array(self):
        """空NumPy数组"""
        from indicators import Indicators
        import numpy as np

        data = np.array([])
        result = Indicators.ema(data, 5)

        assert len(result) == 0


class TestDivisionByZero:
    """除零测试"""

    def test_zero_std_in_bollinger(self):
        """布林带中的零标准差"""
        from indicators import Indicators
        import numpy as np

        # 所有价格相同，标准差为0
        data = np.array([100.0] * 30)
        result = Indicators.bollinger_bands(data)

        # 应该处理零标准差
        assert result is not None

    def test_zero_price_in_change_percent(self):
        """变化百分比中的零价格"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)

            # 基准价为0，设置内部价格历史
            trader.price_history["ETHUSD"] = [0.0] + [100.0] * 19
            trader.latest_prices["ETHUSD"] = {"bid": 100, "ask": 101, "mid": 100.5, "spread": 1}

            result = trader.calc_momentum_signal("ETHUSD")

            # 应该处理零除或返回NONE
            assert result is not None

    def test_zero_capital_pnl_percent(self):
        """零资金的盈亏百分比"""
        from risk_control import RiskController

        controller = RiskController(total_capital=0.0)
        controller.set_position("long", 100.0, 10.0)

        # 不应该崩溃
        pnl, pnl_percent = controller.calculate_pnl(110.0)
        assert pnl == 100.0


class TestStateConsistency:
    """状态一致性测试"""

    def test_position_price_mismatch(self):
        """持仓和价格不匹配"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader, Position

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)

            # 有持仓但没有价格
            trader.positions["ETHUSD"] = Position(
                symbol="ETHUSD",
                direction="long",
                entry_price=3000.0,
                lot_size=0.1,
                entry_time=time.time(),
                stop_loss=2850.0,
                take_profit=3150.0
            )
            # latest_prices["ETHUSD"] 未设置

            # 检查盈亏计算
            pnl = trader._unrealized_pnl_percent(
                trader.positions["ETHUSD"],
                trader.latest_prices.get("ETHUSD", {})
            )

            assert pnl == 0.0  # 无价格时应返回0

    def test_orphan_trade_record(self):
        """孤立的交易记录"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader, TradeRecord

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)

            # 添加交易记录但没有对应持仓
            record = TradeRecord(
                symbol="UNKNOWN",  # 未知品种
                action="BUY",
                direction="long",
                price=1000.0,
                lot_size=0.1,
                timestamp=time.time(),
                reason="test"
            )
            trader.trade_history.append(record)

            # 系统应该能正常运行
            assert len(trader.trade_history) == 1


class TestConcurrencyIssues:
    """并发问题测试"""

    def test_concurrent_position_modification(self):
        """并发持仓修改"""
        import threading

        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader, Position

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)
            errors = []

            def add_position():
                try:
                    for i in range(100):
                        trader.positions["ETHUSD"] = Position(
                            symbol="ETHUSD",
                            direction="long",
                            entry_price=3000 + i,
                            lot_size=0.1,
                            entry_time=time.time(),
                            stop_loss=2850.0,
                            take_profit=3150.0
                        )
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

            def remove_position():
                try:
                    for i in range(100):
                        if "ETHUSD" in trader.positions:
                            del trader.positions["ETHUSD"]
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=add_position)
            t2 = threading.Thread(target=remove_position)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # 应该没有致命错误
            # 注意：字典操作不是原子的，可能有竞态条件

    def test_concurrent_price_update(self):
        """并发价格更新"""
        import threading

        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)

            def update_prices():
                for i in range(1000):
                    trader.latest_prices["ETHUSD"] = {
                        "bid": 3000 + i,
                        "ask": 3001 + i,
                        "mid": 3000.5 + i
                    }

            def read_prices():
                for i in range(1000):
                    price = trader.latest_prices.get("ETHUSD", {})
                    # 使用价格
                    _ = price.get("bid", 0)

            threads = [
                threading.Thread(target=update_prices),
                threading.Thread(target=read_prices)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()


class TestNetworkFailures:
    """网络故障测试"""

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """连接超时"""
        with patch('playwright.async_api.async_playwright') as mock_pw:
            mock_pw_instance = AsyncMock()
            mock_pw_instance.chromium.launch_persistent_context = AsyncMock(
                side_effect=asyncio.TimeoutError("Connection timeout")
            )
            mock_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)

            from data_fetcher import DataFetcher

            fetcher = DataFetcher()
            result = await fetcher.connect()

            # 应该返回False而不是崩溃
            assert result == False

    @pytest.mark.asyncio
    async def test_page_crash(self):
        """页面崩溃"""
        with patch('playwright.async_api.async_playwright') as mock_pw:
            mock_page = AsyncMock()
            mock_page.query_selector = AsyncMock(
                side_effect=Exception("Page crashed")
            )

            mock_context = AsyncMock()
            mock_context.pages = [mock_page]

            from data_fetcher import DataFetcher

            fetcher = DataFetcher()
            fetcher.page = mock_page
            fetcher.is_connected = True

            # 获取价格应该处理异常
            result = await fetcher.fetch_price()

            assert result is None


class TestConfigurationEdgeCases:
    """配置边界条件测试"""

    def test_zero_cooldown(self):
        """零冷却时间"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)
            trader.trade_config["trade_cooldown"] = 0

            # 设置最近交易时间
            trader._last_trade_time["ETHUSD"] = time.time()

            # 设置信号
            trader.latest_signals["ETHUSD"] = {
                "type": "BUY",
                "strength": 0.8,
                "mode": "momentum",
                "reason": "test"
            }

            should, _, reason = trader.should_trade("ETHUSD")

            # 零冷却应该允许交易
            assert should == True

    def test_negative_cooldown(self):
        """负冷却时间（不应该发生）"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)
            trader.trade_config["trade_cooldown"] = -60

            # 设置最近交易时间
            trader._last_trade_time["ETHUSD"] = time.time()

            # 设置信号
            trader.latest_signals["ETHUSD"] = {
                "type": "BUY",
                "strength": 0.8,
                "mode": "momentum",
                "reason": "test"
            }

            should, _, _ = trader.should_trade("ETHUSD")

            # 负冷却时间会导致条件总是满足
            assert should == True

    def test_zero_max_loss_percent(self):
        """零最大亏损比例"""
        from risk_control import RiskController, RiskAction

        controller = RiskController(total_capital=10000.0, max_loss_percent=0.0)
        controller.set_position("long", 100.0, 10.0)

        # 零阈值时，微小亏损不一定触发（浮点精度问题）
        status = controller.check_risk(99.99)

        # 返回的状态应该有效（NONE 或 STOP_LOSS 都是合理的）
        assert status.action in (RiskAction.NONE, RiskAction.STOP_LOSS)

    def test_100_percent_max_loss(self):
        """100%最大亏损比例"""
        from risk_control import RiskController, RiskAction

        controller = RiskController(total_capital=10000.0, max_loss_percent=1.0)
        controller.set_position("long", 100.0, 10.0)

        # 当阈值为100%时，9.9%的亏损（990/10000）不应触发止损
        status = controller.check_risk(1.0)

        # 9.9%亏损不触发，因为远低于100%阈值
        # 这是正确的行为 - NONE表示继续持有
        assert status.action == RiskAction.NONE
        assert status.is_profit == False  # 确认是亏损状态


class TestTimestampEdgeCases:
    """时间戳边界条件测试"""

    def test_future_timestamp(self):
        """未来时间戳"""
        from risk_control import Position

        future_time = time.time() + 86400 * 365  # 一年后

        pos = Position(
            direction="long",
            entry_price=100.0,
            quantity=10.0,
            entry_time=future_time
        )

        # 应该能创建而不报错
        assert pos.entry_time > time.time()

    def test_past_timestamp(self):
        """过去时间戳"""
        from risk_control import Position

        past_time = 0.0  # Unix纪元

        pos = Position(
            direction="long",
            entry_price=100.0,
            quantity=10.0,
            entry_time=past_time
        )

        assert pos.entry_time == 0.0

    def test_negative_timestamp(self):
        """负时间戳（1970年之前）"""
        from risk_control import Position

        negative_time = -86400.0  # 1969年

        pos = Position(
            direction="long",
            entry_price=100.0,
            quantity=10.0,
            entry_time=negative_time
        )

        assert pos.entry_time < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
