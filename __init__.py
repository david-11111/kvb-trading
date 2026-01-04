"""
智能交易平台

模块结构：
- config.py: 配置文件
- indicators.py: 技术指标计算（MACD、ATR、布林带等）
- philosophy.py: 投资哲学体系（震荡/单边判断）
- risk_control.py: 风控模块（5%止损）
- data_fetcher.py: 数据获取（Playwright自动化）
- trader.py: 交易执行（buy/sell/平仓）
- decision_engine.py: 信号决策引擎
- main.py: 主程序入口
"""

__version__ = "1.0.0"
__author__ = "Smart Trading"
