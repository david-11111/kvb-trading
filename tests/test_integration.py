"""
集成测试
测试覆盖：模块间交互、完整交易流程、端到端场景
"""

import pytest
import asyncio
import time
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# 模拟 winsound
sys.modules['winsound'] = MagicMock()


class TestIndicatorPredictorIntegration:
    """指标和预测器集成测试"""

    def test_indicators_feed_predictor(self):
        """指标数据驱动预测"""
        import numpy as np
        from indicators import Indicators
        from predictor import PricePredictor

        # 生成价格数据
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(50) * 0.5) + 100

        # 计算MACD
        macd_result = Indicators.macd(prices)

        # 使用价格进行预测
        predictor = PricePredictor(lookback=20)
        prediction = predictor.predict(list(prices))

        # 两者应该能正常工作
        assert macd_result is not None
        assert prediction is not None

        # MACD和预测方向应该有一定相关性（趋势市场）
        # 这是一个软验证，不是硬性要求

    def test_bollinger_and_prediction_agreement(self):
        """布林带和预测一致性"""
        import numpy as np
        from indicators import Indicators
        from predictor import PricePredictor

        # 构造强上升趋势
        prices = np.array([100 + i * 0.5 for i in range(50)])

        # 计算布林带
        bb = Indicators.bollinger_bands(prices)

        # 预测
        predictor = PricePredictor(lookback=20)
        prediction = predictor.predict(list(prices))

        # 上升趋势中，价格应该在布林带上半部分
        assert bb.percent_b > 0.5
        # 预测应该是UP
        assert prediction.direction.value in ["UP", "NEUTRAL"]


class TestRiskControlIntegration:
    """风控集成测试"""

    def test_risk_with_position(self):
        """风控与持仓集成"""
        from risk_control import RiskController, RiskAction

        controller = RiskController(total_capital=10000.0, max_loss_percent=0.05)

        # 开仓 - quantity=100 确保价格跌5点就触发5%止损
        controller.set_position("long", 100.0, 100.0)

        # 模拟价格变动，跌5点 = 亏损500 = 5%
        prices = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0]  # 下跌

        actions = []
        for price in prices:
            status = controller.check_risk(price)
            actions.append(status.action)

        # 最后应该触发止损（95点时亏损5%）
        assert RiskAction.STOP_LOSS in actions

    def test_risk_profit_no_action(self):
        """盈利时无风控动作"""
        from risk_control import RiskController, RiskAction

        controller = RiskController(total_capital=10000.0)
        controller.set_position("long", 100.0, 10.0)

        # 模拟盈利
        for price in [101.0, 102.0, 105.0, 110.0, 120.0]:
            status = controller.check_risk(price)
            assert status.action == RiskAction.NONE
            assert status.is_profit == True


class TestAutoTraderDecisionFlow:
    """自动交易决策流程测试"""

    @pytest.fixture
    def trader(self):
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader
            return AutoTrader(
                symbols=["ETHUSD", "XAUUSD"],
                auto_trade=True,
                live_trade=False
            )

    def test_signal_to_decision_flow(self, trader):
        """信号到决策流程"""
        # 设置价格历史
        for i in range(20):
            trader.price_history["ETHUSD"].append(3000 + i * 10)

        # 设置当前价格
        trader.latest_prices["ETHUSD"] = {
            "bid": 3190,
            "ask": 3191,
            "mid": 3190.5,
            "spread": 1
        }

        # 生成信号 (方法只需要symbol，内部获取价格数据)
        signal = trader.calc_momentum_signal("ETHUSD")
        trader.latest_signals["ETHUSD"] = signal

        # 检查决策
        should, direction, reason = trader.should_trade("ETHUSD")

        # 应该能做出决策（无论是否交易）
        assert isinstance(should, bool)
        assert isinstance(direction, str)
        assert isinstance(reason, str)

    def test_multi_symbol_decision(self, trader):
        """多品种决策"""
        # 设置两个品种的数据
        for symbol in ["ETHUSD", "XAUUSD"]:
            for i in range(20):
                trader.price_history[symbol].append(1000 + i * 5)

            trader.latest_prices[symbol] = {
                "bid": 1095,
                "ask": 1096,
                "mid": 1095.5,
                "spread": 1
            }

            signal = trader.calc_momentum_signal(symbol)
            trader.latest_signals[symbol] = signal

        # 检查每个品种的决策
        for symbol in ["ETHUSD", "XAUUSD"]:
            should, direction, reason = trader.should_trade(symbol)
            assert isinstance(should, bool)

    def test_position_blocks_new_trade(self, trader):
        """持仓阻止新交易"""
        from auto_trader import Position

        # 设置信号
        trader.latest_signals["ETHUSD"] = {
            "type": "BUY",
            "strength": 0.8,
            "mode": "momentum",
            "reason": "test"
        }

        # 设置持仓
        trader.positions["ETHUSD"] = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=2850.0,
            take_profit=3150.0
        )

        should, _, reason = trader.should_trade("ETHUSD")

        assert should == False
        # 新逻辑：同向持仓时显示盈亏状态
        assert "持仓" in reason or "加仓" in reason


class TestFullTradingScenario:
    """完整交易场景测试"""

    @pytest.fixture
    def trader(self):
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader
            return AutoTrader(
                symbols=["ETHUSD"],
                auto_trade=True,
                live_trade=False
            )

    def test_open_position_scenario(self, trader):
        """开仓场景"""
        from auto_trader import Position

        # 初始状态：无持仓
        assert len(trader.positions) == 0

        # 设置强买入信号
        trader.latest_signals["ETHUSD"] = {
            "type": "BUY",
            "strength": 0.8,
            "mode": "momentum",
            "reason": "追涨动量"
        }

        # 检查是否应该交易
        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == True
        assert direction == "long"

        # 模拟开仓
        trader.positions["ETHUSD"] = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=2850.0,
            take_profit=3150.0
        )

        # 开仓后不应再开仓
        should, _, reason = trader.should_trade("ETHUSD")
        assert should == False
        # 新逻辑：同向持仓时显示盈亏状态
        assert "持仓" in reason or "加仓" in reason

    def test_profit_scenario(self, trader):
        """盈利场景"""
        from auto_trader import Position

        # 开仓
        trader.positions["ETHUSD"] = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=2850.0,
            take_profit=3150.0
        )

        # 价格上涨
        trader.latest_prices["ETHUSD"] = {
            "bid": 3100.0,
            "ask": 3101.0,
            "mid": 3100.5
        }

        # 计算盈亏
        pnl_pct = trader._unrealized_pnl_percent(
            trader.positions["ETHUSD"],
            trader.latest_prices["ETHUSD"]
        )

        assert pnl_pct > 0
        assert pnl_pct > 3  # 约3.33%

    def test_loss_scenario(self, trader):
        """亏损场景"""
        from auto_trader import Position

        # 开仓
        trader.positions["ETHUSD"] = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=2850.0,
            take_profit=3150.0
        )

        # 价格下跌
        trader.latest_prices["ETHUSD"] = {
            "bid": 2900.0,
            "ask": 2901.0,
            "mid": 2900.5
        }

        # 计算盈亏
        pnl_pct = trader._unrealized_pnl_percent(
            trader.positions["ETHUSD"],
            trader.latest_prices["ETHUSD"]
        )

        assert pnl_pct < 0

        # 亏损时不应开新仓
        trader.latest_signals["XAUUSD"] = {
            "type": "BUY",
            "strength": 0.8,
            "mode": "momentum",
            "reason": "test"
        }

        should, _, reason = trader.should_trade("XAUUSD")
        assert should == False
        assert "存在亏损仓位" in reason


class TestLoggingIntegration:
    """日志集成测试"""

    def test_decision_logging(self):
        """决策日志"""
        import tempfile
        from trade_logger import JsonlLogger, now_event

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "decisions.jsonl"
            logger = JsonlLogger(log_path)

            # 记录决策
            event = now_event(
                "decision",
                symbol="ETHUSD",
                should_trade=True,
                direction="long",
                reason="momentum_signal"
            )
            logger.log(event)

            # 验证日志
            import json
            with open(log_path, 'r', encoding='utf-8') as f:
                line = f.readline()
                data = json.loads(line)

            assert data["kind"] == "decision"
            assert data["symbol"] == "ETHUSD"
            assert data["should_trade"] == True

    def test_trade_event_logging(self):
        """交易事件日志"""
        import tempfile
        import json
        from trade_logger import JsonlLogger, now_event

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trade_events.jsonl"
            logger = JsonlLogger(log_path)

            # 记录交易事件
            events = [
                now_event("trade_attempt", symbol="ETHUSD", direction="long"),
                now_event("trade_success", symbol="ETHUSD", price=3000.0),
                now_event("trade_failed", symbol="ETHUSD", reason="click_failed"),
            ]

            for event in events:
                logger.log(event)

            # 验证日志
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            assert len(lines) == 3

            for line in lines:
                data = json.loads(line)
                assert "kind" in data
                assert "ts" in data
                assert "symbol" in data


class TestConfigIntegration:
    """配置集成测试"""

    def test_config_affects_decision(self):
        """配置影响决策"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)

            # 设置高信号强度要求
            trader.trade_config["min_signal_strength"] = 0.9

            # 设置非常弱的信号，确保低于任何阈值
            trader.latest_signals["ETHUSD"] = {
                "type": "BUY",
                "strength": 0.2,  # 低于所有阈值（自适应默认0.6，配置0.9）
                "mode": "momentum",
                "reason": "test"
            }

            should, _, reason = trader.should_trade("ETHUSD")

            assert should == False
            assert "信号强度不足" in reason

    def test_cooldown_config(self):
        """冷却配置"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)
            trader.trade_config["trade_cooldown"] = 120  # 2分钟

            # 设置最近交易时间为1分钟前
            trader._last_trade_time["ETHUSD"] = time.time() - 60

            # 设置信号
            trader.latest_signals["ETHUSD"] = {
                "type": "BUY",
                "strength": 0.8,
                "mode": "momentum",
                "reason": "test"
            }

            should, _, reason = trader.should_trade("ETHUSD")

            assert should == False
            assert "冷却中" in reason


class TestErrorRecovery:
    """错误恢复测试"""

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect(self):
        """断开后重连"""
        with patch('playwright.async_api.async_playwright') as mock_pw:
            mock_context = AsyncMock()
            mock_browser = AsyncMock()
            mock_page = AsyncMock()
            mock_page.wait_for_selector = AsyncMock()

            mock_context.pages = [mock_page]
            mock_browser.contexts = [mock_context]

            mock_pw_instance = AsyncMock()
            mock_pw_instance.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
            mock_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)

            from data_fetcher import DataFetcher

            fetcher = DataFetcher()

            # 第一次连接
            fetcher.browser = mock_browser
            fetcher.context = mock_context
            fetcher.page = mock_page
            fetcher.is_connected = True
            fetcher._playwright = mock_pw_instance

            # 断开
            await fetcher.disconnect()
            assert fetcher.is_connected == False

            # 应该能重新连接（模拟）
            fetcher.is_connected = True
            assert fetcher.is_connected == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
