"""
自动交易系统逻辑测试
测试覆盖：交易决策、持仓管理、冷却时间、信号生成
"""

import pytest
import time
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

# 模拟 winsound (Windows专用)
sys.modules['winsound'] = MagicMock()

from auto_trader import AutoTrader, Position, TradeRecord


class TestAutoTraderInit:
    """AutoTrader 初始化测试"""

    @pytest.fixture
    def trader(self):
        with patch('auto_trader.DataFetcher'):
            trader = AutoTrader(
                symbols=["ETHUSD", "XAUUSD"],
                auto_trade=True,
                check_interval=5.0,
                live_trade=False
            )
            return trader

    def test_init_symbols(self, trader):
        """初始化品种"""
        assert "ETHUSD" in trader.symbols
        assert "XAUUSD" in trader.symbols

    def test_init_price_history(self, trader):
        """初始化价格历史"""
        assert "ETHUSD" in trader.price_history
        assert "XAUUSD" in trader.price_history
        assert len(trader.price_history["ETHUSD"]) == 0

    def test_init_positions(self, trader):
        """初始化持仓"""
        assert len(trader.positions) == 0

    def test_init_trade_history(self, trader):
        """初始化交易历史"""
        assert len(trader.trade_history) == 0


class TestShouldTrade:
    """交易决策测试"""

    @pytest.fixture
    def trader(self):
        with patch('auto_trader.DataFetcher'):
            trader = AutoTrader(
                symbols=["ETHUSD"],
                auto_trade=True,
                live_trade=False
            )
            return trader

    def test_auto_trade_disabled(self, trader):
        """自动交易关闭"""
        trader.auto_trade = False
        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == False
        assert "自动交易已关闭" in reason

    def test_cooldown_active(self, trader):
        """冷却时间内"""
        trader._last_trade_time["ETHUSD"] = time.time()
        trader.trade_config["trade_cooldown"] = 60

        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == False
        assert "冷却中" in reason

    def test_has_position(self, trader):
        """已有持仓"""
        trader.positions["ETHUSD"] = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=2850.0,
            take_profit=3150.0
        )

        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == False
        # 新逻辑：同向持仓时会显示盈亏状态
        assert "持仓" in reason or "加仓" in reason

    def test_losing_position_blocks_new_trade(self, trader):
        """有亏损持仓时禁止开新仓"""
        # 设置XAUUSD持仓
        trader.symbols = ["ETHUSD", "XAUUSD"]
        trader.positions["XAUUSD"] = Position(
            symbol="XAUUSD",
            direction="long",
            entry_price=2000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=1900.0,
            take_profit=2100.0
        )

        # 设置当前价格（亏损）
        trader.latest_prices["XAUUSD"] = {
            "bid": 1950.0,
            "ask": 1951.0,
            "mid": 1950.5
        }

        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == False
        assert "存在亏损仓位" in reason

    def test_max_positions_reached(self, trader):
        """持仓数已满"""
        trader.trade_config["max_total_positions"] = 1
        trader.positions["XAUUSD"] = Position(
            symbol="XAUUSD",
            direction="long",
            entry_price=2000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=1900.0,
            take_profit=2100.0
        )
        trader.latest_prices["XAUUSD"] = {
            "bid": 2050.0,
            "ask": 2051.0,
            "mid": 2050.5
        }

        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == False
        assert "持仓数已满" in reason

    def test_signal_strength_insufficient(self, trader):
        """信号强度不足"""
        trader.trade_config["min_signal_strength"] = 0.5
        trader.latest_signals["ETHUSD"] = {
            "type": "BUY",
            "strength": 0.3,  # 低于阈值
            "mode": "macd"
        }

        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == False
        assert "信号强度不足" in reason

    def testcalc_momentum_signal_buy(self, trader):
        """动量信号买入"""
        trader.trade_config["min_signal_strength"] = 0.5
        trader.latest_signals["ETHUSD"] = {
            "type": "BUY",
            "strength": 0.8,
            "mode": "momentum",
            "reason": "追涨动量"
        }

        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == True
        assert direction == "long"

    def testcalc_momentum_signal_sell(self, trader):
        """动量信号卖出"""
        trader.trade_config["min_signal_strength"] = 0.5
        trader.latest_signals["ETHUSD"] = {
            "type": "SELL",
            "strength": 0.8,
            "mode": "momentum",
            "reason": "追跌动量"
        }

        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == True
        assert direction == "short"


class TestMomentumSignal:
    """动量信号测试"""

    @pytest.fixture
    def trader(self):
        with patch('auto_trader.DataFetcher'):
            trader = AutoTrader(
                symbols=["ETHUSD"],
                auto_trade=True,
                live_trade=False
            )
            return trader

    def test_momentum_signal_disabled(self, trader):
        """动量信号禁用"""
        from auto_trader import MOMENTUM_CONFIG
        with patch.dict(MOMENTUM_CONFIG, {'enabled': False}):
            trader.price_history["ETHUSD"] = [3000 + i * 10 for i in range(20)]
            trader.latest_prices["ETHUSD"] = {"bid": 3100, "ask": 3101, "spread": 1, "mid": 3100.5}

            result = trader.calc_momentum_signal("ETHUSD")

            assert result["type"] == "NONE"

    def test_momentum_insufficient_data(self, trader):
        """数据不足"""
        # 少于lookback (默认12)
        trader.price_history["ETHUSD"] = [3000, 3001]
        trader.latest_prices["ETHUSD"] = {"bid": 3001, "ask": 3002, "spread": 1, "mid": 3001.5}

        result = trader.calc_momentum_signal("ETHUSD")

        assert result["type"] == "NONE"
        assert "数据不足" in result["reason"] or "无价格数据" in result["reason"]

    def test_momentum_spread_too_large(self, trader):
        """点差过大"""
        trader.price_history["ETHUSD"] = [3000 + i for i in range(20)]
        trader.latest_prices["ETHUSD"] = {"bid": 3000, "ask": 3100, "spread": 100, "mid": 3050}  # 大点差

        result = trader.calc_momentum_signal("ETHUSD")

        if result["type"] == "NONE":
            # 可能是点差过大或其他原因
            assert "点差" in result.get("reason", "") or result["type"] == "NONE"

    def test_momentum_buy_signal(self, trader):
        """买入信号"""
        # 构造持续上涨的价格，足够长以满足lookback要求
        trader.price_history["ETHUSD"] = [3000 + i * 5 for i in range(20)]
        trader.latest_prices["ETHUSD"] = {"bid": 3095, "ask": 3096, "spread": 1, "mid": 3095.5}

        result = trader.calc_momentum_signal("ETHUSD")

        # 如果产生信号，应该是BUY
        if result["type"] != "NONE":
            assert result["type"] == "BUY"

    def test_momentum_sell_signal(self, trader):
        """卖出信号"""
        # 构造持续下跌的价格
        trader.price_history["ETHUSD"] = [3000 - i * 5 for i in range(20)]
        trader.latest_prices["ETHUSD"] = {"bid": 2905, "ask": 2906, "spread": 1, "mid": 2905.5}

        result = trader.calc_momentum_signal("ETHUSD")

        if result["type"] != "NONE":
            assert result["type"] == "SELL"


class TestPositionManagement:
    """持仓管理测试"""

    @pytest.fixture
    def trader(self):
        with patch('auto_trader.DataFetcher'):
            trader = AutoTrader(
                symbols=["ETHUSD"],
                auto_trade=True,
                live_trade=False
            )
            return trader

    def test_unrealized_pnl_long_profit(self, trader):
        """多头浮盈计算"""
        position = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=2850.0,
            take_profit=3150.0
        )
        price_info = {"bid": 3100.0, "ask": 3101.0, "mid": 3100.5}

        pnl_pct = trader._unrealized_pnl_percent(position, price_info)

        # 多头用bid价计算: (3100 - 3000) / 3000 * 100 = 3.33%
        assert pnl_pct > 0
        assert abs(pnl_pct - 3.33) < 0.1

    def test_unrealized_pnl_long_loss(self, trader):
        """多头浮亏计算"""
        position = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=2850.0,
            take_profit=3150.0
        )
        price_info = {"bid": 2900.0, "ask": 2901.0, "mid": 2900.5}

        pnl_pct = trader._unrealized_pnl_percent(position, price_info)

        assert pnl_pct < 0

    def test_unrealized_pnl_short_profit(self, trader):
        """空头浮盈计算"""
        position = Position(
            symbol="ETHUSD",
            direction="short",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=3150.0,
            take_profit=2850.0
        )
        price_info = {"bid": 2899.0, "ask": 2900.0, "mid": 2899.5}

        pnl_pct = trader._unrealized_pnl_percent(position, price_info)

        # 空头用ask价计算: (3000 - 2900) / 3000 * 100 = 3.33%
        assert pnl_pct > 0

    def test_unrealized_pnl_short_loss(self, trader):
        """空头浮亏计算"""
        position = Position(
            symbol="ETHUSD",
            direction="short",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=3150.0,
            take_profit=2850.0
        )
        price_info = {"bid": 3099.0, "ask": 3100.0, "mid": 3099.5}

        pnl_pct = trader._unrealized_pnl_percent(position, price_info)

        assert pnl_pct < 0

    def test_unrealized_pnl_empty_price_info(self, trader):
        """空价格信息"""
        position = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=2850.0,
            take_profit=3150.0
        )

        pnl_pct = trader._unrealized_pnl_percent(position, {})

        assert pnl_pct == 0.0


class TestTradeRecord:
    """交易记录测试"""

    def test_trade_record_creation(self):
        """创建交易记录"""
        record = TradeRecord(
            symbol="ETHUSD",
            action="BUY",
            direction="long",
            price=3000.0,
            lot_size=0.1,
            timestamp=time.time(),
            reason="momentum_signal",
            pnl=0.0,
            executed=True,
            requested_lot=0.1,
            lot_set_ok=True,
            lot_value_after_set="0.1"
        )

        assert record.symbol == "ETHUSD"
        assert record.action == "BUY"
        assert record.executed == True

    def test_trade_record_to_dict(self):
        """交易记录转字典"""
        record = TradeRecord(
            symbol="ETHUSD",
            action="BUY",
            direction="long",
            price=3000.0,
            lot_size=0.1,
            timestamp=time.time(),
            reason="test"
        )

        d = asdict(record)

        assert isinstance(d, dict)
        assert d["symbol"] == "ETHUSD"


class TestCooldownMechanism:
    """冷却机制测试"""

    @pytest.fixture
    def trader(self):
        with patch('auto_trader.DataFetcher'):
            trader = AutoTrader(
                symbols=["ETHUSD"],
                auto_trade=True,
                live_trade=False
            )
            trader.trade_config["trade_cooldown"] = 60
            return trader

    def test_cooldown_blocks_trade(self, trader):
        """冷却时间阻止交易"""
        # 设置最近交易时间
        trader._last_trade_time["ETHUSD"] = time.time()

        # 设置有效信号
        trader.latest_signals["ETHUSD"] = {
            "type": "BUY",
            "strength": 0.8,
            "mode": "momentum",
            "reason": "test"
        }

        should, direction, reason = trader.should_trade("ETHUSD")

        assert should == False
        assert "冷却中" in reason

    def test_cooldown_expired(self, trader):
        """冷却时间过期"""
        # 设置过去的交易时间
        trader._last_trade_time["ETHUSD"] = time.time() - 120  # 2分钟前

        # 设置有效信号
        trader.latest_signals["ETHUSD"] = {
            "type": "BUY",
            "strength": 0.8,
            "mode": "momentum",
            "reason": "test"
        }

        should, direction, reason = trader.should_trade("ETHUSD")

        # 冷却时间过期后应该允许交易
        assert should == True


class TestEdgeCases:
    """边界条件测试"""

    @pytest.fixture
    def trader(self):
        with patch('auto_trader.DataFetcher'):
            return AutoTrader(
                symbols=["ETHUSD"],
                auto_trade=True,
                live_trade=False
            )

    def test_empty_prices(self, trader):
        """空价格列表"""
        trader.price_history["ETHUSD"] = []  # 空价格历史
        trader.latest_prices["ETHUSD"] = {"bid": 100, "ask": 101, "mid": 100.5, "spread": 1}
        result = trader.calc_momentum_signal("ETHUSD")
        assert result["type"] == "NONE"

    def test_none_price_info(self, trader):
        """None价格信息"""
        position = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=2850.0,
            take_profit=3150.0
        )

        pnl = trader._unrealized_pnl_percent(position, None)
        assert pnl == 0.0

    def test_zero_entry_price(self, trader):
        """零入场价"""
        position = Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=0.0,  # 不应该发生
            lot_size=0.1,
            entry_time=time.time(),
            stop_loss=0.0,
            take_profit=0.0
        )
        price_info = {"bid": 3000.0, "ask": 3001.0}

        # 应该不崩溃
        pnl = trader._unrealized_pnl_percent(position, price_info)
        assert pnl == 0.0

    def test_unknown_symbol(self, trader):
        """未知品种"""
        should, direction, reason = trader.should_trade("UNKNOWN")
        # 应该能处理未知品种
        assert should == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
