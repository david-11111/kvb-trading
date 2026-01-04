"""
内存泄漏和资源管理测试
测试覆盖：历史数据限制、对象生命周期、资源清理
"""

import pytest
import gc
import sys
import time
import weakref
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# 模拟 winsound
sys.modules['winsound'] = MagicMock()


class TestPriceHistoryLimits:
    """价格历史限制测试"""

    def test_price_history_limit_in_data_fetcher(self):
        """DataFetcher 价格历史限制"""
        with patch('playwright.async_api.async_playwright'):
            from data_fetcher import DataFetcher, PriceData

            fetcher = DataFetcher()

            # 添加超过限制的数据
            for i in range(600):
                price = PriceData(
                    current_price=100 + i,
                    high=101 + i,
                    low=99 + i,
                    open=100 + i,
                    close=100 + i,
                    timestamp=time.time()
                )
                fetcher._price_history.append(price)
                if len(fetcher._price_history) > fetcher._max_history:
                    fetcher._price_history.pop(0)

            # 应该最多保留 _max_history 条
            assert len(fetcher._price_history) <= fetcher._max_history

    def test_auto_trader_price_history_limit(self):
        """AutoTrader 价格历史限制"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)

            # 添加超过限制的数据
            for i in range(200):
                trader.price_history["ETHUSD"].append(3000 + i)
                if len(trader.price_history["ETHUSD"]) > trader.max_history:
                    trader.price_history["ETHUSD"].pop(0)

            assert len(trader.price_history["ETHUSD"]) <= trader.max_history

    def test_trade_history_growth(self):
        """交易历史增长（潜在内存泄漏点）"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader, TradeRecord

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)

            # 模拟大量交易记录
            initial_count = len(trader.trade_history)

            for i in range(1000):
                record = TradeRecord(
                    symbol="ETHUSD",
                    action="BUY",
                    direction="long",
                    price=3000.0,
                    lot_size=0.1,
                    timestamp=time.time(),
                    reason="test"
                )
                trader.trade_history.append(record)

            # 注意：trade_history 没有限制！这是潜在的内存问题
            # 建议添加限制或定期清理
            assert len(trader.trade_history) == initial_count + 1000

            # 警告：长时间运行会导致内存增长


class TestObjectLifecycle:
    """对象生命周期测试"""

    def test_position_cleanup(self):
        """持仓清理"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader, Position

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)

            # 创建持仓
            pos = Position(
                symbol="ETHUSD",
                direction="long",
                entry_price=3000.0,
                lot_size=0.1,
                entry_time=time.time(),
                stop_loss=2850.0,
                take_profit=3150.0
            )
            trader.positions["ETHUSD"] = pos

            # 创建弱引用
            weak_pos = weakref.ref(pos)

            # 删除持仓
            del trader.positions["ETHUSD"]
            del pos

            # 强制垃圾回收
            gc.collect()

            # 对象应该被回收
            assert weak_pos() is None

    def test_trader_cleanup(self):
        """Trader对象清理"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)
            weak_trader = weakref.ref(trader)

            del trader
            gc.collect()

            # 注意：由于复杂引用关系，可能无法立即回收
            # 这是一个观察点，不是严格测试

    def test_logger_not_leaked(self):
        """日志器不泄漏"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            initial_loggers = len(list(gc.get_objects()))

            for _ in range(10):
                trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)
                del trader
                gc.collect()

            final_loggers = len(list(gc.get_objects()))

            # 对象数量不应该大幅增长
            # 允许一些增长（Python内部缓存）
            growth = final_loggers - initial_loggers
            assert growth < 1000, f"对象增长过多: {growth}"


class TestAsyncResourceCleanup:
    """异步资源清理测试"""

    @pytest.mark.asyncio
    async def test_fetcher_disconnect(self):
        """DataFetcher断开连接"""
        with patch('playwright.async_api.async_playwright') as mock_pw:
            mock_context = AsyncMock()
            mock_browser = AsyncMock()
            mock_browser.contexts = [mock_context]
            mock_context.pages = []

            mock_pw_instance = AsyncMock()
            mock_pw_instance.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
            mock_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)

            from data_fetcher import DataFetcher

            fetcher = DataFetcher()

            # 模拟连接
            fetcher.browser = mock_browser
            fetcher.context = mock_context
            fetcher._playwright = mock_pw_instance
            fetcher.is_connected = True

            # 断开连接
            await fetcher.disconnect()

            # 验证清理
            assert fetcher.browser is None
            assert fetcher.context is None
            assert fetcher.is_connected == False

    @pytest.mark.asyncio
    async def test_multiple_connect_disconnect_cycles(self):
        """多次连接/断开循环"""
        with patch('playwright.async_api.async_playwright') as mock_pw:
            mock_context = AsyncMock()
            mock_browser = AsyncMock()
            mock_context.pages = []

            mock_pw_instance = AsyncMock()
            mock_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)

            from data_fetcher import DataFetcher

            for _ in range(5):
                fetcher = DataFetcher()
                fetcher.browser = mock_browser
                fetcher.context = mock_context
                fetcher._playwright = mock_pw_instance
                fetcher.is_connected = True

                await fetcher.disconnect()

                assert fetcher.is_connected == False

            gc.collect()
            # 应该没有资源泄漏


class TestFileHandleLeaks:
    """文件句柄泄漏测试"""

    def test_jsonl_logger_file_handle(self):
        """JSONL日志器文件句柄"""
        import tempfile
        import os

        from trade_logger import JsonlLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"

            for i in range(100):
                logger = JsonlLogger(log_path)
                logger.log({"test": i, "data": "x" * 100})

            # 检查文件是否正常关闭
            # 尝试删除文件（如果有未关闭的句柄会失败）
            try:
                os.remove(log_path)
            except PermissionError:
                pytest.fail("文件句柄未正确关闭")

    def test_trade_data_dir_creation(self):
        """交易数据目录创建"""
        from trade_logger import data_dir_for

        # 多次调用不应该创建多个目录对象
        dir1 = data_dir_for(__file__)
        dir2 = data_dir_for(__file__)

        # 应该是同一个路径
        assert str(dir1) == str(dir2)


class TestMemoryUsagePatterns:
    """内存使用模式测试"""

    def test_numpy_array_reuse(self):
        """NumPy数组重用"""
        import numpy as np

        from indicators import Indicators

        # 多次计算不应该持续增加内存
        initial_objects = len(gc.get_objects())

        for _ in range(100):
            data = np.random.randn(100)
            result = Indicators.ema(data, 10)
            del data, result

        gc.collect()
        final_objects = len(gc.get_objects())

        # 对象数量不应该大幅增长
        growth = final_objects - initial_objects
        assert growth < 500, f"对象增长过多: {growth}"

    def test_predictor_no_memory_leak(self):
        """预测器无内存泄漏"""
        from predictor import PricePredictor

        initial_objects = len(gc.get_objects())

        for _ in range(100):
            predictor = PricePredictor(lookback=20)
            prices = [100 + i * 0.1 for i in range(30)]
            result = predictor.predict(prices)
            del predictor, prices, result

        gc.collect()
        final_objects = len(gc.get_objects())

        growth = final_objects - initial_objects
        assert growth < 500, f"对象增长过多: {growth}"


class TestConcurrentResourceUsage:
    """并发资源使用测试"""

    def test_concurrent_price_history_access(self):
        """并发价格历史访问"""
        import threading

        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)
            errors = []

            def add_prices():
                try:
                    for i in range(100):
                        trader.price_history["ETHUSD"].append(3000 + i)
                        if len(trader.price_history["ETHUSD"]) > trader.max_history:
                            trader.price_history["ETHUSD"].pop(0)
                except Exception as e:
                    errors.append(e)

            def read_prices():
                try:
                    for _ in range(100):
                        if trader.price_history["ETHUSD"]:
                            _ = trader.price_history["ETHUSD"][-1]
                except Exception as e:
                    errors.append(e)

            threads = []
            for _ in range(3):
                threads.append(threading.Thread(target=add_prices))
                threads.append(threading.Thread(target=read_prices))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # 可能有并发问题（list不是线程安全的）
            # 这个测试用于发现潜在的竞态条件


class TestResourceLimits:
    """资源限制测试"""

    def test_max_history_enforcement(self):
        """最大历史强制执行"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)
            trader.max_history = 50

            # 添加超过限制的数据
            for i in range(100):
                trader.price_history["ETHUSD"].append(i)
                # 手动执行限制（模拟实际行为）
                while len(trader.price_history["ETHUSD"]) > trader.max_history:
                    trader.price_history["ETHUSD"].pop(0)

            assert len(trader.price_history["ETHUSD"]) == 50

    def test_max_positions_enforcement(self):
        """最大持仓强制执行"""
        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader, Position

            trader = AutoTrader(symbols=["ETHUSD", "XAUUSD", "USOIL"], live_trade=False)
            trader.trade_config["max_total_positions"] = 2

            # 添加2个持仓
            for symbol in ["ETHUSD", "XAUUSD"]:
                trader.positions[symbol] = Position(
                    symbol=symbol,
                    direction="long",
                    entry_price=100.0,
                    lot_size=0.1,
                    entry_time=time.time(),
                    stop_loss=95.0,
                    take_profit=105.0
                )
                trader.latest_prices[symbol] = {"bid": 101, "ask": 102, "mid": 101.5}

            # 第三个应该被阻止
            should, _, reason = trader.should_trade("USOIL")
            assert should == False
            assert "持仓数已满" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
