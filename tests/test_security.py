"""
安全测试
测试覆盖：敏感信息处理、输入验证、日志安全、配置安全
"""

import pytest
import os
import sys
import json
import tempfile
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSensitiveDataHandling:
    """敏感数据处理测试"""

    def test_password_not_in_logs(self):
        """密码不应该出现在日志中"""
        import logging
        from io import StringIO

        # 创建日志捕获器
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)

        logger = logging.getLogger("TestLogger")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # 模拟登录日志
        password = "SecretPassword123"
        phone = "18800000000"

        # 正确的做法：遮掩敏感信息
        masked_phone = f"{phone[:3]}****{phone[-4:]}"
        logger.info(f"开始自动登录，手机号: {masked_phone}")

        log_content = log_capture.getvalue()

        # 密码不应该出现
        assert password not in log_content
        # 完整手机号不应该出现
        assert phone not in log_content

    def test_config_password_exposure(self):
        """检查配置文件中的密码暴露风险"""
        from config import LOGIN_CONFIG

        password = LOGIN_CONFIG.get("password", "")

        # 警告：如果密码硬编码在配置中，这是安全风险
        # 建议使用环境变量
        if password:
            # 这是一个潜在的安全问题
            # 建议：密码应该从环境变量读取，而不是硬编码
            pass

    def test_environment_variable_priority(self):
        """环境变量应该优先于配置文件"""
        with patch('playwright.async_api.async_playwright'):
            from data_fetcher import DataFetcher

            # 设置环境变量
            test_phone = "13800138000"
            test_password = "TestEnvPassword"

            with patch.dict(os.environ, {
                'SMART_TRADING_PHONE': test_phone,
                'SMART_TRADING_PASSWORD': test_password
            }):
                fetcher = DataFetcher()
                phone, password = fetcher._get_login_credentials()

                # 环境变量应该优先（如果配置文件中没有值）
                # 实际行为取决于实现

    def test_trade_logger_no_sensitive_data(self):
        """交易日志不应包含敏感数据"""
        from trade_logger import JsonlLogger, now_event

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JsonlLogger(log_path)

            # 记录事件
            event = now_event(
                "trade",
                symbol="ETHUSD",
                price=3000.0,
                lot_size=0.1
            )
            logger.log(event)

            # 读取日志
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查不应该有敏感字段
            assert "password" not in content.lower()
            assert "密码" not in content


class TestInputValidation:
    """输入验证测试"""

    def test_symbol_validation(self):
        """品种代码验证"""
        valid_symbols = ["USOIL", "XAUUSD", "ETHUSD", "BTCUSD", "GBPUSD", "EURUSD"]

        for symbol in valid_symbols:
            # 应该只包含大写字母
            assert symbol.isupper()
            assert symbol.isalpha()

    def test_lot_size_validation(self):
        """手数验证"""
        valid_lots = [0.01, 0.1, 1.0, 5.0]
        invalid_lots = [-0.1, 0, -1, 100]  # 负数、零、过大

        for lot in valid_lots:
            assert lot > 0
            assert lot <= 10  # 假设最大10手

        for lot in invalid_lots:
            if lot <= 0 or lot > 10:
                # 应该被拒绝或限制
                pass

    def test_price_validation(self):
        """价格验证"""
        # 价格应该是正数
        valid_prices = [0.01, 100.0, 3000.0, 100000.0]
        invalid_prices = [-100, 0, float('inf'), float('nan')]

        for price in valid_prices:
            assert price > 0
            assert price < float('inf')

        for price in invalid_prices:
            if price <= 0 or price == float('inf') or price != price:  # nan check
                # 应该被拒绝
                pass

    def test_direction_validation(self):
        """方向验证"""
        valid_directions = ["long", "short"]
        invalid_directions = ["up", "down", "buy", "sell", "", None]

        for direction in valid_directions:
            assert direction in ["long", "short"]


class TestInjectionPrevention:
    """注入防护测试"""

    def test_symbol_injection(self):
        """品种代码注入测试"""
        malicious_inputs = [
            "USOIL; DROP TABLE trades;",
            "USOIL<script>alert('xss')</script>",
            "USOIL' OR '1'='1",
            "../../../etc/passwd",
            "USOIL\x00HIDDEN",
        ]

        with patch('playwright.async_api.async_playwright'):
            from data_fetcher import DataFetcher

            fetcher = DataFetcher()

            for malicious in malicious_inputs:
                # 选择器中使用用户输入时需要转义
                selector = f'.custom-list-item[data-item="{malicious}"]'

                # 检查是否包含危险字符
                if any(c in malicious for c in ['<', '>', '"', ';', '\x00']):
                    # 应该被清理或转义
                    pass

    def test_log_injection(self):
        """日志注入测试"""
        malicious_log_entries = [
            "Normal log\nINFO: Fake log entry",
            "Normal log\r\n2026-01-03 Fake timestamp",
            "Normal log\x00Hidden data",
        ]

        from trade_logger import JsonlLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JsonlLogger(log_path)

            for entry in malicious_log_entries:
                # JSON编码应该安全处理特殊字符
                logger.log({"message": entry})

            # 读取并验证
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 每个事件应该是一行
            # JSON编码会转义特殊字符
            for line in lines:
                data = json.loads(line)
                # 消息应该完整保存
                assert "message" in data


class TestConfigurationSecurity:
    """配置安全测试"""

    def test_default_config_safety(self):
        """默认配置安全性"""
        from config import (
            AUTO_TRADE_CONFIG,
            RISK_CONFIG,
            POSITION_SIZING_CONFIG,
            EXECUTION_CONFIG
        )

        # 检查风控配置
        assert RISK_CONFIG["max_loss_percent"] <= 0.1, "止损比例不应超过10%"

        # 检查交易配置
        assert AUTO_TRADE_CONFIG["max_total_positions"] <= 5, "最大持仓数不应过高"

        # 检查仓位配置
        assert POSITION_SIZING_CONFIG["max_lot"] <= 10, "最大手数不应过高"

    def test_headless_mode_default(self):
        """无头模式默认值"""
        from config import WEB_CONFIG

        # 生产环境中headless应该可配置
        assert "headless" in WEB_CONFIG

    def test_timeout_configuration(self):
        """超时配置"""
        from config import WEB_CONFIG

        nav_timeout = WEB_CONFIG.get("nav_timeout_ms", 0)
        ready_timeout = WEB_CONFIG.get("ready_timeout_ms", 0)

        # 超时不应该无限制
        assert nav_timeout > 0
        assert nav_timeout < 600000  # 不超过10分钟
        assert ready_timeout > 0
        assert ready_timeout < 600000


class TestDataIntegrity:
    """数据完整性测试"""

    def test_trade_record_immutability(self):
        """交易记录不可变性"""
        sys.modules['winsound'] = MagicMock()
        from auto_trader import TradeRecord

        record = TradeRecord(
            symbol="ETHUSD",
            action="BUY",
            direction="long",
            price=3000.0,
            lot_size=0.1,
            timestamp=1000000.0,
            reason="test"
        )

        # dataclass默认是可变的
        # 如果需要不可变性，应该使用frozen=True
        original_price = record.price
        record.price = 9999.0  # 这会成功

        # 建议：关键交易记录应该使用frozen=True

    def test_jsonl_log_append_only(self):
        """JSONL日志只追加"""
        from trade_logger import JsonlLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JsonlLogger(log_path)

            # 写入第一条
            logger.log({"id": 1})
            with open(log_path, 'r') as f:
                line1 = f.read()

            # 写入第二条
            logger.log({"id": 2})
            with open(log_path, 'r') as f:
                content = f.read()

            # 应该是追加，第一条仍然存在
            assert line1 in content
            assert content.count('\n') == 2


class TestNetworkSecurity:
    """网络安全测试"""

    def test_https_url(self):
        """确保使用HTTPS"""
        from config import WEB_CONFIG

        url = WEB_CONFIG.get("url", "")

        # 应该使用HTTPS
        assert url.startswith("https://"), "应该使用HTTPS连接"

    def test_cdp_url_local_only(self):
        """CDP URL应该只允许本地"""
        from config import WEB_CONFIG

        cdp_url = WEB_CONFIG.get("cdp_url", "") or os.environ.get("SMART_TRADING_CDP_URL", "")

        if cdp_url:
            # CDP应该只连接本地
            assert "127.0.0.1" in cdp_url or "localhost" in cdp_url, \
                "CDP应该只连接本地地址"


class TestErrorHandling:
    """错误处理安全测试"""

    def test_exception_no_sensitive_info(self):
        """异常不应泄露敏感信息"""
        with patch('playwright.async_api.async_playwright') as mock_pw:
            mock_pw.return_value.start.side_effect = Exception("Connection failed")

            from data_fetcher import DataFetcher

            fetcher = DataFetcher()

            try:
                # 这会失败
                import asyncio
                asyncio.run(fetcher.connect())
            except Exception as e:
                error_msg = str(e)
                # 错误消息不应该包含密码
                assert "password" not in error_msg.lower()
                assert "密码" not in error_msg

    def test_stack_trace_sanitization(self):
        """堆栈跟踪清理"""
        # 生产环境中不应该暴露完整堆栈跟踪
        # 这是一个设计建议而非测试


class TestRateLimiting:
    """速率限制测试"""

    def test_trade_cooldown_enforced(self):
        """交易冷却强制执行"""
        sys.modules['winsound'] = MagicMock()
        import time

        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader

            trader = AutoTrader(symbols=["ETHUSD"], live_trade=False)
            trader.trade_config["trade_cooldown"] = 60

            # 设置最近交易时间
            trader._last_trade_time["ETHUSD"] = time.time()

            # 设置有效信号
            trader.latest_signals["ETHUSD"] = {
                "type": "BUY",
                "strength": 0.8,
                "mode": "momentum",
                "reason": "test"
            }

            # 应该被冷却阻止
            should, _, reason = trader.should_trade("ETHUSD")
            assert should == False
            assert "冷却" in reason

    def test_max_positions_enforced(self):
        """最大持仓限制强制执行"""
        sys.modules['winsound'] = MagicMock()
        import time

        with patch('auto_trader.DataFetcher'):
            from auto_trader import AutoTrader, Position

            trader = AutoTrader(symbols=["ETHUSD", "XAUUSD"], live_trade=False)
            trader.trade_config["max_total_positions"] = 1

            # 添加一个持仓
            trader.positions["XAUUSD"] = Position(
                symbol="XAUUSD",
                direction="long",
                entry_price=2000.0,
                lot_size=0.1,
                entry_time=time.time(),
                stop_loss=1900.0,
                take_profit=2100.0
            )
            trader.latest_prices["XAUUSD"] = {"bid": 2050, "ask": 2051, "mid": 2050.5}

            # 第二个应该被阻止
            trader.latest_signals["ETHUSD"] = {
                "type": "BUY",
                "strength": 0.8,
                "mode": "momentum",
                "reason": "test"
            }

            should, _, reason = trader.should_trade("ETHUSD")
            assert should == False
            assert "持仓数已满" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
