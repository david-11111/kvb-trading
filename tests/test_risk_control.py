"""
风控模块单元测试
测试覆盖：止损逻辑、盈亏计算、风控监控
"""

import pytest
import time
import threading
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from risk_control import RiskController, RiskManager, RiskAction, RiskStatus, Position


class TestRiskController:
    """风控控制器测试"""

    @pytest.fixture
    def controller(self):
        return RiskController(total_capital=10000.0, max_loss_percent=0.05)

    def test_init(self, controller):
        """初始化测试"""
        assert controller.total_capital == 10000.0
        assert controller.max_loss_percent == 0.05
        assert controller.max_loss_amount == 500.0  # 5% of 10000
        assert controller.current_position is None

    def test_set_position(self, controller):
        """设置持仓"""
        controller.set_position("long", 100.0, 10.0)

        assert controller.current_position is not None
        assert controller.current_position.direction == "long"
        assert controller.current_position.entry_price == 100.0
        assert controller.current_position.quantity == 10.0

    def test_clear_position(self, controller):
        """清除持仓"""
        controller.set_position("long", 100.0, 10.0)
        controller.clear_position()

        assert controller.current_position is None

    def test_update_capital(self, controller):
        """更新资金"""
        controller.update_capital(20000.0)

        assert controller.total_capital == 20000.0
        assert controller.max_loss_amount == 1000.0  # 5% of 20000


class TestCalculatePnL:
    """盈亏计算测试"""

    @pytest.fixture
    def controller(self):
        return RiskController(total_capital=10000.0)

    def test_no_position(self, controller):
        """无持仓"""
        pnl, pnl_percent = controller.calculate_pnl(100.0)
        assert pnl == 0.0
        assert pnl_percent == 0.0

    def test_long_profit(self, controller):
        """多头盈利"""
        controller.set_position("long", 100.0, 10.0)
        pnl, pnl_percent = controller.calculate_pnl(110.0)  # 价格上涨10

        assert pnl == 100.0  # (110-100) * 10
        assert pnl_percent == 0.01  # 100/10000

    def test_long_loss(self, controller):
        """多头亏损"""
        controller.set_position("long", 100.0, 10.0)
        pnl, pnl_percent = controller.calculate_pnl(90.0)  # 价格下跌10

        assert pnl == -100.0  # (90-100) * 10
        assert pnl_percent == -0.01

    def test_short_profit(self, controller):
        """空头盈利"""
        controller.set_position("short", 100.0, 10.0)
        pnl, pnl_percent = controller.calculate_pnl(90.0)  # 价格下跌10

        assert pnl == 100.0  # -(90-100) * 10
        assert pnl_percent == 0.01

    def test_short_loss(self, controller):
        """空头亏损"""
        controller.set_position("short", 100.0, 10.0)
        pnl, pnl_percent = controller.calculate_pnl(110.0)  # 价格上涨10

        assert pnl == -100.0  # -(110-100) * 10
        assert pnl_percent == -0.01


class TestCheckRisk:
    """风险检查测试"""

    @pytest.fixture
    def controller(self):
        return RiskController(total_capital=10000.0, max_loss_percent=0.05)

    def test_no_position(self, controller):
        """无持仓"""
        status = controller.check_risk(100.0)

        assert status.action == RiskAction.NONE
        assert status.current_pnl == 0.0
        assert "无持仓" in status.message

    def test_in_profit(self, controller):
        """盈利状态"""
        controller.set_position("long", 100.0, 10.0)
        status = controller.check_risk(110.0)

        assert status.action == RiskAction.NONE
        assert status.is_profit == True
        assert status.current_pnl == 100.0
        assert "盈利中" in status.message

    def test_small_loss(self, controller):
        """小幅亏损（未触发止损）"""
        # quantity=100, 价格跌2点 = 亏损200 (2%)
        controller.set_position("long", 100.0, 100.0)
        status = controller.check_risk(98.0)  # 亏损200，2%

        assert status.action == RiskAction.NONE
        assert status.is_profit == False
        assert "继续监控" in status.message

    def test_warning_threshold(self, controller):
        """接近止损线（80%警戒）"""
        # quantity=100, 价格跌4.1点 = 亏损410 (4.1%)，触发警告 (80% of 5% = 4%)
        controller.set_position("long", 100.0, 100.0)
        status = controller.check_risk(95.9)  # 亏损410，4.1% > 4%警戒线

        assert status.action == RiskAction.WARNING
        assert "风险警告" in status.message

    def test_stop_loss_triggered(self, controller):
        """触发止损"""
        # quantity=100, 价格跌5点 = 亏损500 (5%)
        controller.set_position("long", 100.0, 100.0)
        status = controller.check_risk(95.0)  # 亏损500，5%

        assert status.action == RiskAction.STOP_LOSS
        assert "触发止损" in status.message

    def test_exact_stop_loss_threshold(self, controller):
        """刚好在止损线"""
        # quantity=100, 价格跌5点 = 刚好5%亏损
        controller.set_position("long", 100.0, 100.0)
        status = controller.check_risk(95.0)  # 亏损500，正好5%

        assert status.action == RiskAction.STOP_LOSS

    def test_beyond_stop_loss(self, controller):
        """超过止损线"""
        # quantity=100, 价格跌10点 = 亏损1000 (10%)
        controller.set_position("long", 100.0, 100.0)
        status = controller.check_risk(90.0)  # 亏损1000，10%

        assert status.action == RiskAction.STOP_LOSS


class TestRiskMonitoring:
    """风控监控测试"""

    @pytest.fixture
    def controller(self):
        return RiskController(total_capital=10000.0)

    def test_start_stop_monitoring(self, controller):
        """启动和停止监控"""
        price = [100.0]
        controller.set_position("long", 100.0, 10.0)

        controller.start_monitoring(lambda: price[0], check_interval=0.1)
        assert controller.is_monitoring == True

        time.sleep(0.2)

        controller.stop_monitoring()
        assert controller.is_monitoring == False

    def test_double_start_monitoring(self, controller):
        """重复启动监控"""
        price = [100.0]
        controller.set_position("long", 100.0, 10.0)

        controller.start_monitoring(lambda: price[0], check_interval=0.1)
        controller.start_monitoring(lambda: price[0], check_interval=0.1)  # 第二次

        # 应该只有一个监控线程
        assert controller.is_monitoring == True

        controller.stop_monitoring()

    def test_stop_loss_callback(self, controller):
        """止损回调测试"""
        callback_called = [False]
        callback_status = [None]

        def on_stop_loss(status):
            callback_called[0] = True
            callback_status[0] = status

        controller.on_stop_loss = on_stop_loss
        # quantity=100，价格跌5点以上触发5%止损
        controller.set_position("long", 100.0, 100.0)

        # 模拟价格下跌导致止损
        price = [100.0]

        controller.start_monitoring(lambda: price[0], check_interval=0.05)
        time.sleep(0.1)

        # 触发止损：价格跌10点 = 亏损1000 (10%)
        price[0] = 90.0
        time.sleep(0.2)

        # 检查回调是否被调用
        assert callback_called[0] == True
        assert callback_status[0] is not None
        assert callback_status[0].action == RiskAction.STOP_LOSS

    def test_get_risk_summary(self, controller):
        """获取风控摘要"""
        controller.set_position("long", 100.0, 10.0)

        summary = controller.get_risk_summary()

        assert summary["total_capital"] == 10000.0
        assert summary["has_position"] == True
        assert summary["position_direction"] == "long"
        assert summary["position_entry_price"] == 100.0


class TestRiskManager:
    """风控管理器测试"""

    @pytest.fixture
    def manager(self):
        return RiskManager(total_capital=10000.0)

    def test_open_close_position(self, manager):
        """开仓和平仓"""
        manager.open_position("long", 100.0, 10.0)
        assert manager.controller.current_position is not None

        manager.close_position()
        assert manager.controller.current_position is None

    def test_check(self, manager):
        """检查风险"""
        manager.open_position("long", 100.0, 10.0)
        status = manager.check(110.0)

        assert isinstance(status, RiskStatus)
        assert status.is_profit == True

    def test_should_stop_loss(self, manager):
        """是否应该止损"""
        # quantity=100，便于计算：价格跌5点 = 5%亏损
        manager.open_position("long", 100.0, 100.0)

        # 盈利时不应止损
        assert manager.should_stop_loss(110.0) == False

        # 大幅亏损时应止损 (价格跌10点 = 10%亏损)
        assert manager.should_stop_loss(90.0) == True

    def test_get_pnl(self, manager):
        """获取盈亏"""
        manager.open_position("long", 100.0, 10.0)
        pnl, pnl_percent = manager.get_pnl(110.0)

        assert pnl == 100.0
        assert pnl_percent == 0.01

    def test_auto_monitor(self, manager):
        """自动监控"""
        manager.open_position("long", 100.0, 10.0)

        callback_called = [False]

        def on_stop_loss(status):
            callback_called[0] = True

        price = [100.0]
        manager.start_auto_monitor(lambda: price[0], on_stop_loss)

        time.sleep(0.1)
        manager.stop_auto_monitor()

        # 监控应该已停止
        assert manager.controller.is_monitoring == False


class TestEdgeCases:
    """边界条件测试"""

    def test_zero_capital(self):
        """零资金"""
        controller = RiskController(total_capital=0.0)
        controller.set_position("long", 100.0, 10.0)

        # 计算PnL时不应崩溃
        pnl, pnl_percent = controller.calculate_pnl(110.0)
        assert pnl == 100.0
        # 零资金时百分比计算会有问题，但不应崩溃

    def test_negative_capital(self):
        """负资金（不应该发生，但需要处理）"""
        controller = RiskController(total_capital=-1000.0)
        # 应该能创建，但行为可能不正确

    def test_zero_quantity(self):
        """零数量持仓"""
        controller = RiskController(total_capital=10000.0)
        controller.set_position("long", 100.0, 0.0)

        pnl, pnl_percent = controller.calculate_pnl(110.0)
        assert pnl == 0.0

    def test_zero_entry_price(self):
        """零入场价"""
        controller = RiskController(total_capital=10000.0)
        controller.set_position("long", 0.0, 10.0)

        # 应该能计算而不崩溃
        pnl, pnl_percent = controller.calculate_pnl(100.0)

    def test_very_large_loss(self):
        """巨大亏损"""
        controller = RiskController(total_capital=10000.0, max_loss_percent=0.05)
        controller.set_position("long", 100.0, 1000.0)  # 大仓位

        status = controller.check_risk(1.0)  # 价格暴跌到1

        assert status.action == RiskAction.STOP_LOSS

    def test_very_large_profit(self):
        """巨大盈利"""
        controller = RiskController(total_capital=10000.0)
        controller.set_position("long", 100.0, 10.0)

        status = controller.check_risk(1000000.0)  # 价格暴涨

        assert status.action == RiskAction.NONE
        assert status.is_profit == True


class TestConcurrency:
    """并发测试"""

    def test_concurrent_check_risk(self):
        """并发检查风险"""
        controller = RiskController(total_capital=10000.0)
        controller.set_position("long", 100.0, 10.0)

        results = []

        def check_risk():
            for _ in range(100):
                status = controller.check_risk(105.0)
                results.append(status)

        threads = [threading.Thread(target=check_risk) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 500
        assert all(isinstance(r, RiskStatus) for r in results)

    def test_concurrent_set_clear_position(self):
        """并发设置和清除持仓"""
        controller = RiskController(total_capital=10000.0)

        def set_position():
            for i in range(50):
                controller.set_position("long", 100.0 + i, 10.0)
                time.sleep(0.001)

        def clear_position():
            for i in range(50):
                controller.clear_position()
                time.sleep(0.001)

        t1 = threading.Thread(target=set_position)
        t2 = threading.Thread(target=clear_position)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 应该能正常完成而不崩溃


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
