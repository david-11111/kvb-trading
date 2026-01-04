"""
Pytest配置文件
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 模拟 Windows 专用模块
sys.modules['winsound'] = MagicMock()


def pytest_configure(config):
    """配置pytest"""
    config.addinivalue_line(
        "markers", "slow: 标记慢速测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记集成测试"
    )
    config.addinivalue_line(
        "markers", "security: 标记安全测试"
    )


@pytest.fixture
def sample_prices():
    """示例价格数据"""
    import numpy as np
    np.random.seed(42)
    return list(np.cumsum(np.random.randn(50) * 0.5) + 100)


@pytest.fixture
def sample_ohlc():
    """示例OHLC数据"""
    import numpy as np
    np.random.seed(42)

    n = 50
    close = np.cumsum(np.random.randn(n) * 0.5) + 100
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5

    return high, low, close


@pytest.fixture
def mock_price_info():
    """模拟价格信息"""
    return {
        "bid": 3000.0,
        "ask": 3001.0,
        "mid": 3000.5,
        "spread": 1.0
    }
