"""
智能交易平台配置文件
"""

# 网站配置
WEB_CONFIG = {
    "url": "https://mykvb.com/trade",
    "browser": "chromium",  # 可选: chromium, firefox, webkit
    "headless": False,  # 是否无头模式
    # 网络慢/挑战页时需要更长超时（香港/海外线路常见）
    "nav_timeout_ms": 300000,    # 页面导航超时
    "ready_timeout_ms": 240000,  # 关键交易按钮就绪超时
    # 连接重试（网络差/挑战页/偶发关闭时更稳）
    "connect_retries": 3,
    "connect_retry_sleep_sec": 2,
    # 可选：连接到已打开的Chrome（观察者模式）
    "cdp_url": "",
    # 可选：模拟手机端页面（通常更轻量、更快；但页面布局可能变化）
    # 也可用环境变量 SMART_TRADING_MOBILE=1 临时开启
    "emulate_mobile": False,
    # 加速：阻止部分大资源加载（不影响交易按钮/价格，但能显著提速）
    # 可选资源类型：image, media, font, stylesheet
    "block_resource_types": ["image", "media", "font"],
    # 可选：指定浏览器可执行文件（例如夸克/Chrome）。也可用环境变量 SMART_TRADING_BROWSER_EXE 覆盖。
    # "browser_executable_path": r"C:\\Path\\To\\Quark.exe",
}

# MACD参数
MACD_CONFIG = {
    "fast_period": 12,   # 快速EMA周期
    "slow_period": 26,   # 慢速EMA周期
    "signal_period": 9,  # 信号线周期
}

# 风控配置
RISK_CONFIG = {
    "max_loss_percent": 0.05,  # 最大亏损比例 5%
    "check_interval": 1,       # 风控检查间隔（秒）
}

# 投资哲学配置
PHILOSOPHY_CONFIG = {
    # ATR相关
    "atr_period": 14,
    "atr_low_threshold": 0.3,   # ATR低于历史30%分位数视为震荡
    "atr_high_threshold": 0.7,  # ATR高于历史70%分位数视为单边

    # 布林带相关
    "bb_period": 20,
    "bb_std": 2,
    "bb_squeeze_threshold": 0.1,  # 价格距中轨10%以内视为震荡

    # MACD柱相关
    "macd_zero_threshold": 0.0001,  # MACD柱接近0的阈值

    # 综合判断权重
    "weights": {
        "atr": 0.3,
        "bollinger": 0.3,
        "macd_histogram": 0.2,
        "trend_strength": 0.2,
    }
}

# 交易配置
TRADE_CONFIG = {
    # 震荡市场策略
    "oscillation": {
        "take_profit_percent": 0.02,  # 止盈2%
        "position_size": 0.3,          # 仓位30%
    },
    # 单边市场策略
    "trending": {
        "take_profit_percent": 0.05,  # 止盈5%
        "position_size": 0.5,          # 仓位50%
    }
}

# 多品种监控配置
MULTI_SYMBOL_CONFIG = {
    # 监控的品种列表
    "symbols": ["USOIL", "XAUUSD", "ETHUSD"],

    # 检查间隔（秒）
    "check_interval": 5.0,

    # 保存多少条历史数据用于计算波动率
    "history_length": 60,

    # 波动率阈值(%)，超过此值认为有交易机会
    "volatility_threshold": 0.5,

    # 品种特性配置
    "symbol_settings": {
        "USOIL": {
            "name": "美原油",
            "pip_value": 0.01,      # 最小变动单位
            "typical_spread": 30,   # 典型点差
            "lot_size": 0.10,       # 下单手数
            # Required for POSITION_SIZING_CONFIG.mode="risk_percent"
            # Estimated USD PnL for 1.0 price move per 1.0 lot. Fill with your broker's contract spec.
            "value_per_1_price_move_per_1_lot": 10.0,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 5.0,
        },
        "XAUUSD": {
            "name": "黄金",
            "pip_value": 0.01,
            "typical_spread": 20,
            "lot_size": 0.05,       # 下单手数
            "value_per_1_price_move_per_1_lot": 100.0,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 5.0,
        },
        "ETHUSD": {
            "name": "以太坊",
            "pip_value": 0.01,
            "typical_spread": 100,
            "lot_size": 0.10,       # 下单手数
            "value_per_1_price_move_per_1_lot": 10.0,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 5.0,
        },
    }
}

# 自动交易配置
AUTO_TRADE_CONFIG = {
    # 是否启用自动交易
    "enabled": True,

    # 下单条件
    "min_signal_strength": 0.5,     # 最小信号强度
    "min_prediction_confidence": 0.5,  # 最小预测置信度
    "require_signal_prediction_agree": False,  # 允许仅凭动量信号交易（不要求预测一致）

    # 风控
    "max_positions_per_symbol": 1,  # 每个品种最大持仓数
    "max_total_positions": 3,       # 总最大持仓数
    "stop_loss_percent": 0.05,      # 止损比例 5%
    "take_profit_percent": 0.03,    # 止盈比例 3% (若设置了reward_risk_ratio则此值被覆盖)
    "reward_risk_ratio": 4.0,       # 盈亏比 1:4 (止盈=止损*4)

    # 冷却时间
    "trade_cooldown": 60,           # 下单后冷却时间（秒）
}

# Position sizing:
# - mode="fixed": use symbol_settings.lot_size
# - mode="risk_percent": compute lot by risk% at stop-loss (requires per-symbol value_per_1_price_move_per_1_lot)
POSITION_SIZING_CONFIG = {
    "mode": "fixed",          # "fixed" | "risk_percent" - 使用固定手数
    "risk_percent": 0.01,     # per-trade risk fraction
    "use_equity": True,       # True=equity, False=free_margin
    "min_lot": 0.01,
    "max_lot": 5.0,
    "lot_step": 0.01,
    "account_refresh_sec": 30
}

# 追涨/追跌动量配置（更贴近"看到走强就跟"的操盘直觉）
MOMENTUM_CONFIG = {
    "enabled": True,
    # 用最近 N 个 tick 判断动量（auto_trader 默认 5s 一次；12≈1分钟）
    "lookback_points": 12,  # 改为12个点（约1分钟），避免短期噪声
    # 最近连续上涨/下跌 tick 数要求（减少噪声触发）；0 表示"只要涨/跌就追"（不要求连续）
    "min_consecutive_ticks": 3,  # 改为至少连续3个tick同向，确认趋势
    # 单边涨跌幅阈值(%)：达到即认为"有机会"
    # 说明：ETH 价格波动尺度更大，但这里用百分比统一口径
    "min_change_percent": {
        "USOIL": 0.05,  # 提高到0.05%，减少假信号
        "XAUUSD": 0.05,
        "ETHUSD": 0.05,
    },
    # 点差过滤（避免点差太大追进去吃亏）
    "max_spread_percent": {
        "USOIL": 0.05,
        "XAUUSD": 0.02,
        "ETHUSD": 0.15,
    },
}

# 盈利回撤到接近 0 时强平（“赚到手里才算”）
BREAKEVEN_CLOSE_CONFIG = {
    "enabled": True,
    # 只有当浮盈曾经达到这个安全垫(%)后，才启用“回撤强平”
    "arm_profit_percent": 0.03,
    # 当前浮盈回撤到该阈值(%)以下则强平；0 表示回到不赚就跑
    "close_if_pnl_below_percent": 0.0,
}

# Reverse (close then open opposite) safety:
# Only allow reversing when the existing position has a “profit cushion”.
# This prevents frequent flip-flops in choppy markets and avoids reversing from tiny profits.
REVERSE_CONFIG = {
    "enabled": True,
    # Reverse behavior:
    # - "only_when_losing": only reverse when the existing (opposite) position is losing; if it's winning, keep it.
    # - "profit_cushion": only reverse when the existing position has enough profit cushion.
    "mode": "only_when_losing",
    # Used when mode="profit_cushion". Note: AutoTrader uses percent values (e.g. 1.0 means +1.0%).
    "min_profit_cushion_percent": 1.0,
}

# Add-on-profit (scale in) behavior:
# When a position is already in the same direction and is profitable, allow opening an additional order
# to increase exposure (while keeping the existing position).
ADD_ON_SIGNAL_CONFIG = {
    "enabled": True,
    # Require current unrealized PnL (%) >= this threshold to allow adding.
    # Note: AutoTrader uses percent values (e.g. 0.2 means +0.2%).
    "min_profit_percent": 0.2,
    # Add size = initial lot * ratio.
    "add_lot_ratio": 1.0,
    # Minimum seconds between add-on-signal opens for the same symbol.
    "cooldown_sec": 60,
}

# 盈利后加仓（追涨/顺势加仓）
PYRAMID_CONFIG = {
    "enabled": True,
    # Pure momentum pyramiding using R multiple.
    # R = initial risk distance = abs(entry_price - stop_loss_price).
    "mode": "r_multiple",          # "r_multiple" | "percent"
    "first_add_r": 0.5,            # first add at +0.5R
    "add_every_r": 0.5,            # then add every +0.5R
    "min_signal_strength_for_add": 0.5,
    "require_momentum_same_direction": True,
    "add_cooldown_sec": 60,
    "max_adds_per_symbol": {
        "USOIL": 3,
        "XAUUSD": 3,
        "ETHUSD": 3,
    },
    # Add size as a ratio of initial entry lot (reward multiplier applies).
    "add_lot_ratio": 0.5,
}

# 奖励：盈利单越多，允许更积极的加仓（上限保护）
REWARD_CONFIG = {
    "enabled": True,
    # 每笔盈利平仓奖励点数（按盈亏为正判断）
    "win_points": 1,
    # 奖励对“加仓手数”的增益：每点奖励增加比例
    "add_lot_bonus_per_point": 0.1,
    # 奖励倍数上限（例如 2.0 表示最多加仓手数翻倍）
    "max_add_lot_multiplier": 2.0,
}

# 日志配置
LOG_CONFIG = {
    "log_file": "trading.log",
    "log_level": "INFO",
}

# 自动登录配置
# 敏感信息建议通过环境变量设置：
#   SMART_TRADING_PHONE=18800000000
#   SMART_TRADING_PASSWORD=your_password
# 或者直接在下面填写（注意不要提交到版本控制）
LOGIN_CONFIG = {
    "enabled": True,  # 是否启用自动登录
    # 建议只用环境变量配置，避免把账号密码写进代码：
    #   SMART_TRADING_PHONE=...
    #   SMART_TRADING_PASSWORD=...
    "phone": "",      # 手机号（优先用环境变量 SMART_TRADING_PHONE）
    "password": "",   # 密码（优先用环境变量 SMART_TRADING_PASSWORD）
    "login_timeout_ms": 30000,   # 登录超时时间
    "wait_after_login_sec": 5,   # 登录后等待页面加载的时间
}

# Execution behavior
# Note: KVB网页的买/卖按钮通常只对“当前选中品种”生效，所以：
# - allow_switch_for_open=False：不自动切换品种开仓（避免频繁切换导致卡顿/误操作）
# - allow_switch_for_close=True：允许为风控/平仓而切换（否则可能无法平仓）
EXECUTION_CONFIG = {
    "allow_switch_for_open": False,
    "allow_switch_for_close": True,
    # When allow_switch_for_open is False, try "quick trade" from watchlist row if possible.
    "allow_list_trade": True,
    # Hard safety toggles for live clicks.
    # Use env overrides:
    #   SMART_TRADING_ALLOW_LIVE_OPEN=0/1
    #   SMART_TRADING_ALLOW_LIVE_CLOSE=0/1
    "allow_live_open": True,
    "allow_live_close": True,
    # If lot-size input cannot be set (UI changed / slow load), do we block the trade?
    # - False: continue with the platform's default lot size (still logs lot_set_ok=false)
    # - True: block and log trade_blocked reason="set_lot_size_failed"
    "require_lot_set": False,
    # 是否需要点击确认按钮（如果KVB平台设置了不需要确认，设为False）
    # - False: 点击买入/卖出按钮后直接认为交易成功（用户已在平台设置不需要确认）
    # - True: 点击买入/卖出按钮后还需要找到并点击确认按钮
    # mykvb/trade 默认下单/平仓通常需要二次确认（弹出“新订单”窗口后点底部按钮）
    "require_confirm": True,
    # Screenshots are helpful for debugging but can be slow on poor networks.
    "screenshot_on_trade_failure": False,
    "screenshot_on_trade_success": False,
    "screenshot_full_page": False,
    # After clicking buy/sell, verify the position appears on the platform.
    # This prevents “clicked but not actually placed” when confirmation is required.
    "verify_open_after_trade": True,
    # After clicking close, verify the position disappears on the platform.
    "verify_close_after_trade": True,
    "verify_timeout_sec": 10,
}
