# Smart Trading 项目状态

## 最后更新: 2025-12-30

## 项目概述
基于 KVB 交易平台 (mykvb.com/trade) 的自动化交易系统，使用 Playwright 进行网页自动化操作。

## 目标品种
- USOIL (美原油)

## 账户信息
- 账号: 20002619 (Classic)
- 当前余额: ~522.68 USD

---

## 已完成功能

### 1. 数据获取 (data_fetcher.py)
- [x] 连接 KVB 交易页面
- [x] 获取 OHLC 价格 (.top-price-num)
- [x] 获取买入/卖出价格 (button.price-btn-bid / .price-btn-ask)
- [x] 获取点差 (.price-btn-spread)
- [x] 获取账户信息 (li.profit__item)
- [x] 执行买入/卖出点击
- [x] 持久化登录状态 (browser_data/)

### 2. 技术指标 (indicators.py)
- [x] EMA / SMA
- [x] MACD (金叉/死叉检测)
- [x] KDJ (金叉/死叉检测) - 新增
- [x] ATR
- [x] 布林带
- [x] 趋势强度

### 3. 交易策略 (strategy.py) - 新建
- [x] 开仓策略: MACD + KDJ 共振
  - MACD金叉 + KDJ金叉 = 做多
  - MACD死叉 + KDJ死叉 = 做空
- [x] 加仓策略: 盈利5% + 布林带下轨反弹
- [x] 止损策略: 亏损达5%强制平仓
- [x] 持仓状态管理

### 4. 交易监控 (monitor.py) - 新建
- [x] 整合数据获取、策略分析、交易执行
- [x] 实时价格监控
- [x] 信号生成和回调
- [x] 手动/自动交易模式

### 5. 风控系统 (risk_control.py)
- [x] 5% 止损线
- [x] 持仓盈亏计算
- [x] 自动监控线程

---

## 文件结构

```
smart_trading/
├── __init__.py
├── config.py           # 配置参数
├── data_fetcher.py     # 数据获取 (Playwright)
├── indicators.py       # 技术指标计算
├── strategy.py         # 交易策略 [新建]
├── monitor.py          # 交易监控器 [新建]
├── risk_control.py     # 风控模块
├── decision_engine.py  # 决策引擎
├── trader.py           # 交易执行
├── philosophy.py       # 交易哲学
├── main.py             # 主程序入口
├── requirements.txt    # 依赖
├── browser_data/       # 浏览器持久化数据(保存登录状态)
└── PROJECT_STATUS.md   # 本文件
```

---

## 关键选择器 (KVB页面)

| 元素 | 选择器 |
|------|--------|
| OHLC价格 | `.top-price-num` |
| 买入按钮 | `button.price-btn-bid` |
| 卖出按钮 | `button.price-btn-ask` |
| 点差 | `.price-btn-spread` |
| 当前品种 | `.kline-toolbar-symbol` |
| 账户信息容器 | `li.profit__item` |
| 账户标签 | `span.label` |
| 账户数值 | `span.value` |
| 确认下单按钮 | `button:has-text("市价买入")` / `button:has-text("市价卖出")` |

---

## 使用示例

### 启动监控器
```python
from monitor import TradingMonitor, MonitorConfig

config = MonitorConfig(
    total_capital=500.0,
    auto_trade=False,  # 手动确认模式
    check_interval=5.0
)
monitor = TradingMonitor(config)
await monitor.start()
await monitor.run()
```

### 手动交易
```python
from data_fetcher import DataFetcher

fetcher = DataFetcher(headless=False)
await fetcher.connect()

# 获取价格
price = await fetcher.fetch_price()
print(f"当前价格: {price.close}")

# 买入
await fetcher.click_buy()
# 然后点击确认按钮完成下单
```

---

## 待完成功能

- [x] 自动平仓功能基础版（`monitor.py` 已实现：切换持仓Tab/选中行/右键/双击/确认按钮多策略尝试；如页面更新需微调选择器）
- [ ] 多品种支持
- [ ] 历史交易记录
- [ ] 回测系统
- [ ] GUI界面

---

## 注意事项

1. 首次运行需要手动登录 KVB 账户，登录状态会保存在 `browser_data/` 目录
2. 默认交易手数: 0.01 手
3. 风控参数: 止损5%, 加仓确认5%
4. 最大加仓次数: 3次
