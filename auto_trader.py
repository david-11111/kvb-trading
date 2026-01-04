"""
自动交易系统

整合：
- 多品种监控
- 价格预测
- 信号生成
- 自动下单
- 风控管理
"""

import asyncio
import logging
import time
import json
import os
import winsound
import traceback
import hashlib
import sys
import platform
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from data_fetcher import DataFetcher
from indicators import Indicators
from predictor import PricePredictor, Direction
from evolution_engine import EvolutionEngine, create_evolution_engine
from config import (
    WEB_CONFIG,
    MULTI_SYMBOL_CONFIG,
    AUTO_TRADE_CONFIG,
    MACD_CONFIG,
    MOMENTUM_CONFIG,
    BREAKEVEN_CLOSE_CONFIG,
    REVERSE_CONFIG,
    ADD_ON_SIGNAL_CONFIG,
    PYRAMID_CONFIG,
    REWARD_CONFIG,
    POSITION_SIZING_CONFIG,
    EXECUTION_CONFIG,
)
from trade_logger import JsonlLogger, data_dir_for, dated_jsonl_path, now_event


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    direction: str          # "long" / "short"
    entry_price: float
    lot_size: float
    entry_time: float
    stop_loss: float
    take_profit: float
    add_count: int = 0
    max_unrealized_pnl_percent: float = 0.0
    last_add_time: float = 0.0
    initial_entry_price: float = 0.0
    initial_lot_size: float = 0.0
    r_distance: float = 0.0


@dataclass
class TradeRecord:
    """交易记录"""
    symbol: str
    action: str             # "BUY" / "SELL" / "CLOSE"
    direction: str
    price: float
    lot_size: float
    timestamp: float
    reason: str
    pnl: float = 0.0
    executed: bool = False  # True=真实点击下单，False=纸面/模拟
    requested_lot: float = 0.0
    lot_set_ok: bool = False
    lot_value_after_set: str = ""


class AutoTrader:
    """
    自动交易系统
    """

    def __init__(
        self,
        symbols: List[str] = None,
        auto_trade: bool = True,
        check_interval: float = 5.0,
        live_trade: bool = False,
        headless: bool | None = None,
    ):
        # 品种配置
        self.symbols = symbols or MULTI_SYMBOL_CONFIG.get("symbols", ["USOIL", "XAUUSD", "ETHUSD"])
        self.symbol_settings = MULTI_SYMBOL_CONFIG.get("symbol_settings", {})
        self.auto_trade = auto_trade
        self.check_interval = check_interval
        self.live_trade = live_trade

        # 交易配置
        self.trade_config = AUTO_TRADE_CONFIG
        self.exec_config = dict(EXECUTION_CONFIG)
        # Env overrides for safety toggles.
        def _env_bool(name: str) -> Optional[bool]:
            v = (os.environ.get(name) or "").strip().lower()
            if not v:
                return None
            if v in ("1", "true", "yes", "y", "on"):
                return True
            if v in ("0", "false", "no", "n", "off"):
                return False
            return None

        allow_open = _env_bool("SMART_TRADING_ALLOW_LIVE_OPEN")
        if allow_open is not None:
            self.exec_config["allow_live_open"] = bool(allow_open)
        allow_close = _env_bool("SMART_TRADING_ALLOW_LIVE_CLOSE")
        if allow_close is not None:
            self.exec_config["allow_live_close"] = bool(allow_close)
        require_confirm = _env_bool("SMART_TRADING_REQUIRE_CONFIRM")
        if require_confirm is not None:
            self.exec_config["require_confirm"] = bool(require_confirm)

        # 数据获取器
        if headless is None:
            headless = bool(WEB_CONFIG.get("headless", False))
        self.fetcher = DataFetcher(headless=bool(headless))

        # 预测器
        self.predictor = PricePredictor(lookback=20)

        # 价格历史
        self.price_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.high_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.low_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.max_history = 100

        # 最新数据
        self.latest_prices: Dict[str, dict] = {}
        self.latest_predictions: Dict[str, dict] = {}
        self.latest_signals: Dict[str, dict] = {}

        # 持仓管理
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.reward_points: int = 0
        self.latest_account = None
        self._last_account_refresh_ts: float = 0.0

        # 冷却时间
        self._last_trade_time: Dict[str, float] = {}

        # 当前品种
        self.current_symbol: str = ""

        # 状态
        self._is_running = False

        # 数据目录（固定在 smart_trading/trade_data，避免因 CWD 不同导致写到别处）
        self.data_dir = data_dir_for(__file__)
        self.data_dir.mkdir(exist_ok=True)
        self._decision_logger = JsonlLogger(dated_jsonl_path(self.data_dir, "decisions"))
        self._trade_event_logger = JsonlLogger(dated_jsonl_path(self.data_dir, "trade_events"))
        self._manual_stop_file = self.data_dir / "STOP_TRADING"
        self._manual_trade_stop = False
        self._last_decision_reason: Dict[str, str] = {}
        self._last_heartbeat_ts: float = 0.0
        self._last_error_ts: float = 0.0
        self._recent_error_count: int = 0
        self._trade_paused_until: float = 0.0
        self.exit_reason: str = ""
        self._pending_reverse: dict = {}  # 反向开仓标记：需要先平旧仓再开新仓
        self._pending_close_only: dict = {}  # 只平仓标记：亏损时只平仓不反向开仓
        self._pending_add: dict = {}  # 同向盈利加仓标记：允许在方向正确时增大仓位

        self.logger = logging.getLogger("AutoTrader")

        # 进化引擎（自我反思与学习）- 必须在data_dir设置后初始化
        evolution_data_dir = self.data_dir.parent / "evolution_data"
        self.evolution_engine = create_evolution_engine(
            auto_trader=self,
            data_dir=str(evolution_data_dir)
        )

    def _log_decision(self, symbol: str, should: bool, direction: str, reason: str):
        prev_reason = self._last_decision_reason.get(symbol)
        if prev_reason == reason and not should:
            return
        self._last_decision_reason[symbol] = reason

        self._decision_logger.log(now_event(
            "decision",
            symbol=symbol,
            should_trade=should,
            direction=direction,
            reason=reason,
            signal=self.latest_signals.get(symbol, {}),
            prediction=self.latest_predictions.get(symbol, {}),
            price=self.latest_prices.get(symbol, {}),
            auto_trade=self.auto_trade,
            live_trade=self.live_trade,
        ))

    async def _click_confirm(self, preferred_text: str) -> bool:
        page = getattr(self.fetcher, "page", None)
        if not page:
            self.logger.warning("_click_confirm: 无page对象")
            return False

        self.logger.info(f"尝试点击确认按钮: {preferred_text}")

        # Notes:
        # - 确认按钮通常出现在订单弹窗/对话框中。直接在全页面找 `.order-button` 等过于宽泛，容易误点买/卖按钮导致重复开仓。
        # - Playwright 的 locator + retry + 关闭弹窗检测，比单次 query_selector 更稳。

        import asyncio

        preferred_text = (preferred_text or "").strip()
        texts = [t for t in [
            preferred_text,
            "市价买入",
            "市价卖出",
            "市价",
            "确认",
            "确定",
            "OK",
            "Confirm",
        ] if t]

        cancel_words = ("取消", "关闭", "返回", "Cancel", "Close", "No")

        async def dialog_visible() -> bool:
            return await self._trade_dialog_visible()

        async def click_and_confirm(loc, desc: str) -> bool:
            try:
                if (await loc.count()) <= 0:
                    return False
                el = loc.first
                if not await el.is_visible():
                    return False
                try:
                    await el.scroll_into_view_if_needed(timeout=800)
                except Exception:
                    pass
                try:
                    await el.click(timeout=1200)
                except Exception:
                    try:
                        await el.click(force=True, timeout=1200)
                    except Exception as e:
                        self.logger.debug(f"确认点击失败({desc}): {e}")
                        return False

                # 点击后等待弹窗消失（如果一键交易没有弹窗，外层会绕过 _click_confirm）
                for _ in range(10):
                    await asyncio.sleep(0.15)
                    if not await dialog_visible():
                        self.logger.info(f"确认按钮点击成功: {desc}")
                        return True
                # 弹窗可能仍在，但也可能已提交、UI卡住；留给后续 verify_open_after_trade 来兜底。
                self.logger.info(f"已点击确认({desc})，但弹窗未在预期时间内消失")
                return True
            except Exception as e:
                self.logger.debug(f"确认点击异常({desc}): {e}")
                return False

        # 1) 优先在对话框容器内点击（减少误点）
        dialog_selectors = [
            "[role='dialog']",
            ".order-dialog, .order-modal, .trade-modal, .order-ticket, .trade-ticket, .order-panel, .trade-panel",
            ".ant-modal, .modal, .dialog, .MuiDialog-root",
        ]
        for scope_sel in dialog_selectors:
            try:
                scope = page.locator(scope_sel).first
                if (await scope.count()) <= 0 or (not await scope.is_visible()):
                    continue

                for t in texts:
                    # 先精确匹配按钮文本
                    if await click_and_confirm(scope.locator(f'button:has-text("{t}")'), f"{scope_sel}/button:{t}"):
                        return True
                    if await click_and_confirm(scope.locator(f'[role=\"button\"]:has-text(\"{t}\")'), f"{scope_sel}/rolebtn:{t}"):
                        return True
                    # 某些版本用 div/span 承载点击
                    if await click_and_confirm(scope.locator(f'div:has-text("{t}")'), f"{scope_sel}/div:{t}"):
                        return True

                # 兜底：在对话框内找 “primary/confirm” 类按钮（仍保持在 scope 内，避免误点全局买卖按钮）
                primary = scope.locator("button.btn-primary, button.primary, button:has([class*='primary']), [role='button'][class*='primary'], [class*='confirm']")
                if await click_and_confirm(primary, f"{scope_sel}/primary"):
                    return True

                # 最后兜底：取对话框内最后一个可见按钮（跳过取消类）
                buttons = scope.locator("button, [role='button']")
                try:
                    cnt = await buttons.count()
                except Exception:
                    cnt = 0
                for i in range(min(cnt, 12) - 1, -1, -1):
                    el = buttons.nth(i)
                    try:
                        if not await el.is_visible():
                            continue
                        label = (await el.inner_text()).strip()
                        if any(w in label for w in cancel_words):
                            continue
                        if await click_and_confirm(el, f"{scope_sel}/fallback_last:{label}"):
                            return True
                    except Exception:
                        continue
            except Exception:
                continue

        # 2) 如果没找到对话框容器：回退到全局按文本点击（仍只点“确认/市价”类按钮）
        for t in texts:
            for sel in [f'button:has-text("{t}")', f'[role="button"]:has-text("{t}")']:
                try:
                    if await click_and_confirm(page.locator(sel), f"global:{sel}"):
                        return True
                except Exception:
                    continue

        # 3) 最后：键盘回车（很多弹窗默认确认按钮为主按钮）
        try:
            await page.keyboard.press("Enter")
            for _ in range(10):
                await asyncio.sleep(0.15)
                if not await dialog_visible():
                    self.logger.info("确认按钮回退策略：Enter 生效，弹窗已关闭")
                    return True
        except Exception as e:
            self.logger.debug(f"确认按钮回退策略 Enter 失败: {e}")

        self.logger.warning("未找到确认按钮/确认未生效")
        return False

    async def _trade_dialog_visible(self) -> bool:
        """
        Best-effort detection for the “新订单/确认” dialog after clicking buy/sell/close.
        If the platform is configured as one-click trading, the dialog may not exist.
        """
        page = getattr(self.fetcher, "page", None)
        if not page:
            return False
        # Prefer “dialog-like” containers first; they reduce false positives on the main page.
        for scope_sel in [
            "[role='dialog']",
            ".order-dialog, .order-modal, .trade-modal, .order-ticket, .trade-ticket, .order-panel, .trade-panel",
            ".ant-modal, .modal, .dialog, .MuiDialog-root",
        ]:
            try:
                scope = page.locator(scope_sel).first
                if (await scope.count()) > 0 and await scope.is_visible():
                    return True
            except Exception:
                continue

        # Fallback: key texts/buttons.
        for sel in [
            "text=新订单",
            "text=市价执行",
            "text=Market Execution",
            "button:has-text('市价买入')",
            "button:has-text('市价卖出')",
            "button:has-text('市价')",
            "button:has-text('确认')",
            "button:has-text('确定')",
            "[role='button']:has-text('确认')",
            "[role='button']:has-text('确定')",
        ]:
            try:
                el = page.locator(sel).first
                if (await el.count()) > 0 and await el.is_visible():
                    return True
            except Exception:
                continue
        return False

    async def _wait_platform_position(self, symbol: str, should_exist: bool) -> bool:
        timeout = float(self.exec_config.get("verify_timeout_sec", 10) or 10)
        deadline = time.time() + max(1.0, timeout)
        sym = (symbol or "").upper()
        while time.time() < deadline:
            try:
                positions = await self.fetcher.get_open_positions()
                exists = any((p.get("symbol", "") or "").upper() == sym for p in (positions or []))
                if exists == should_exist:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
        return False

    async def _count_platform_positions(self, symbol: str) -> int:
        sym = (symbol or "").upper()
        try:
            positions = await self.fetcher.get_open_positions()
        except Exception:
            return 0
        return sum(1 for p in (positions or []) if (p.get("symbol", "") or "").upper() == sym)

    async def _wait_platform_position_count_decreased(self, symbol: str, before: int) -> bool:
        timeout = float(self.exec_config.get("verify_timeout_sec", 10) or 10)
        deadline = time.time() + max(1.0, timeout)
        while time.time() < deadline:
            after = await self._count_platform_positions(symbol)
            if after < int(before):
                return True
            await asyncio.sleep(1.0)
        return False

    async def _wait_platform_position_count_increased(self, symbol: str, before: int) -> bool:
        timeout = float(self.exec_config.get("verify_timeout_sec", 10) or 10)
        deadline = time.time() + max(1.0, timeout)
        while time.time() < deadline:
            after = await self._count_platform_positions(symbol)
            if after > int(before):
                return True
            await asyncio.sleep(1.0)
        return False

    async def _screenshot(self, name: str):
        page = getattr(self.fetcher, "page", None)
        if not page:
            return
        if name.startswith("trade_failed") and not bool(self.exec_config.get("screenshot_on_trade_failure", False)):
            return
        if name.startswith("trade_after") and not bool(self.exec_config.get("screenshot_on_trade_success", False)):
            return
        try:
            path = self.data_dir / f"{name}_{int(time.time())}.png"
            await page.screenshot(path=str(path), full_page=bool(self.exec_config.get("screenshot_full_page", False)))
        except Exception:
            return

    async def start(self) -> bool:
        """启动"""
        self.logger.info("正在启动自动交易系统...")

        # 加载之前的持仓（重启恢复）
        self._load_positions()

        ok = await self.fetcher.connect()
        if not ok:
            self.logger.error("无法连接到交易页面")
            return False

        await asyncio.sleep(3)
        self.current_symbol = await self.fetcher.get_symbol() or ""

        self._is_running = True
        self._emit_version_fingerprint()
        self.logger.info("自动交易系统已启动!")
        return True

    def _file_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _emit_version_fingerprint(self):
        """
        Write a reproducible fingerprint of the running code into trade_events.
        This makes it verifiable that the current process is using the latest files.
        """
        try:
            root = Path(__file__).resolve().parent
            files = {
                "auto_trader.py": root / "auto_trader.py",
                "config.py": root / "config.py",
                "data_fetcher.py": root / "data_fetcher.py",
            }
            fp = {}
            for name, p in files.items():
                if p.exists():
                    fp[name] = {
                        "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                        "sha256": self._file_sha256(p),
                    }
            info = {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "cwd": str(Path.cwd()),
            }
            self._trade_event_logger.log(now_event("version", files=fp, runtime=info))
            # Also show a short fingerprint in console/logs.
            short = fp.get("auto_trader.py", {}).get("sha256", "")[:12]
            if short:
                self.logger.info(f"Version fingerprint: auto_trader.py sha256={short}…")
        except Exception:
            return

    async def refresh_account(self, force: bool = False):
        interval = float(POSITION_SIZING_CONFIG.get("account_refresh_sec", 30) or 30)
        if not force and (time.time() - self._last_account_refresh_ts) < interval:
            return
        self._last_account_refresh_ts = time.time()
        try:
            acct = await self.fetcher.fetch_account()
            if acct:
                self.latest_account = acct
        except Exception:
            return

    async def stop(self):
        """停止"""
        self._is_running = False
        self._save_trade_history()

        # 关闭进化引擎，保存学习数据
        try:
            self.evolution_engine.shutdown()
            # 生成每日报告
            report = self.evolution_engine.generate_daily_report()
            self.logger.info(f"\n{report}")
            # 生成策略进化报告（贝叶斯学习结果）
            evolution_report = self.evolution_engine.generate_evolution_report()
            self.logger.info(f"\n{evolution_report}")
        except Exception as e:
            self.logger.warning(f"进化引擎关闭失败: {e}")

        await self.fetcher.disconnect()
        self.logger.info("自动交易系统已停止")

    # ==================== 数据获取 ====================

    async def fetch_prices(self) -> Dict[str, dict]:
        """获取所有品种价格"""
        page = self.fetcher.page
        if not page:
            return {}

        result = {}

        for symbol in self.symbols:
            try:
                bid = 0.0
                ask = 0.0

                item = await page.query_selector(f'.custom-list-item[data-item="{symbol}"]')
                if item:
                    bid_elem = await item.query_selector('.item-bid span')
                    ask_elem = await item.query_selector('.item-ask span')
                    bid = float((await bid_elem.inner_text()).strip()) if bid_elem else 0
                    ask = float((await ask_elem.inner_text()).strip()) if ask_elem else 0
                else:
                    # 回退策略：如果自选列表不可用（页面布局/慢网/未展开），
                    # 则尝试直接从页面主买/卖按钮读取当前品种报价。
                    can_fallback = bool(self.current_symbol) and (symbol in self.current_symbol.upper())
                    if can_fallback:
                        buy_price = await self.fetcher.get_buy_price()
                        sell_price = await self.fetcher.get_sell_price()
                        ask = float(buy_price) if buy_price else 0.0
                        bid = float(sell_price) if sell_price else 0.0

                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    result[symbol] = {
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'spread': ask - bid,
                        'timestamp': time.time()
                    }

                    # 更新历史
                    self.price_history[symbol].append(mid)
                    self.high_history[symbol].append(ask)
                    self.low_history[symbol].append(bid)

                    for hist in [self.price_history, self.high_history, self.low_history]:
                        if len(hist[symbol]) > self.max_history:
                            hist[symbol].pop(0)

            except Exception as e:
                self.logger.debug(f"获取 {symbol} 失败: {e}")

        self.latest_prices = result
        return result

    def _round_down(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        return (int(value / step)) * step

    def _compute_lot_size(self, symbol: str, entry_price: float) -> float:
        settings = self.symbol_settings.get(symbol, {})
        fixed = float(settings.get("lot_size", 0.01) or 0.01)

        if POSITION_SIZING_CONFIG.get("mode") != "risk_percent":
            return fixed

        v = float(settings.get("value_per_1_price_move_per_1_lot", 0.0) or 0.0)
        if v <= 0 or self.latest_account is None or not entry_price:
            # fallback to fixed if contract value not configured or account unknown
            self._trade_event_logger.log(now_event(
                "sizing_fallback",
                symbol=symbol,
                reason="missing_contract_value_or_account",
                fixed_lot=fixed,
                value_per_1_price_move_per_1_lot=v,
                has_account=bool(self.latest_account),
            ))
            return fixed

        use_equity = bool(POSITION_SIZING_CONFIG.get("use_equity", True))
        base = float(self.latest_account.equity if use_equity else self.latest_account.free_margin)
        risk_frac = float(POSITION_SIZING_CONFIG.get("risk_percent", 0.01) or 0.01)
        risk_amount = max(0.0, base * risk_frac)

        stop_pct = float(self.trade_config.get("stop_loss_percent", 0.05) or 0.05)
        stop_dist = entry_price * stop_pct
        loss_per_1_lot = stop_dist * v
        if loss_per_1_lot <= 0:
            return fixed

        raw_lot = risk_amount / loss_per_1_lot

        min_lot = float(settings.get("min_lot", POSITION_SIZING_CONFIG.get("min_lot", 0.01)) or 0.01)
        max_lot = float(settings.get("max_lot", POSITION_SIZING_CONFIG.get("max_lot", 5.0)) or 5.0)
        step = float(settings.get("lot_step", POSITION_SIZING_CONFIG.get("lot_step", 0.01)) or 0.01)

        lot = max(min_lot, min(max_lot, raw_lot))
        lot = self._round_down(lot, step)
        lot = round(lot, 4)

        self._trade_event_logger.log(now_event(
            "sizing",
            symbol=symbol,
            base_value=base,
            risk_percent=risk_frac,
            risk_amount=risk_amount,
            entry_price=entry_price,
            stop_loss_percent=stop_pct,
            stop_distance=stop_dist,
            value_per_1_price_move_per_1_lot=v,
            loss_per_1_lot=loss_per_1_lot,
            raw_lot=raw_lot,
            final_lot=lot,
        ))

        return lot

    # ==================== 预测与信号 ====================

    def calc_prediction(self, symbol: str) -> dict:
        """计算预测"""
        prices = self.price_history.get(symbol, [])
        if len(prices) < 10:
            result = {"direction": "NEUTRAL", "confidence": 0, "reason": "数据不足"}
            self.latest_predictions[symbol] = result
            return result

        pred = self.predictor.predict(prices)
        result = {
            "direction": pred.direction.value,
            "confidence": pred.confidence,
            "reason": pred.reason,
            "methods_agree": pred.methods_agree,
        }
        self.latest_predictions[symbol] = result
        return result

    def calc_signal(self, symbol: str) -> dict:
        """计算交易信号"""
        prices = self.price_history.get(symbol, [])
        highs = self.high_history.get(symbol, [])
        lows = self.low_history.get(symbol, [])

        base_result = {"type": "NONE", "strength": 0, "reason": "数据不足"}

        close = np.array(prices)
        high = np.array(highs)
        low = np.array(lows)

        # 动量信号始终计算（避免指标计算异常导致“有动量也不下单”）
        momentum = self.calc_momentum_signal(symbol)

        try:
            # 指标交叉类信号：需要更长历史，避免早期噪声
            if len(prices) >= 30:
                macd = Indicators.macd(
                    close,
                    MACD_CONFIG["fast_period"],
                    MACD_CONFIG["slow_period"],
                    MACD_CONFIG["signal_period"]
                )
                histogram = macd.histogram

                kdj = Indicators.kdj(high, low, close)
                k = kdj.k
                d = kdj.d

                signal_type = "NONE"
                strength = 0.0
                reasons = []

                if len(histogram) >= 2:
                    if histogram[-1] > 0 and histogram[-2] <= 0:
                        reasons.append("MACD金叉")
                        strength += 0.35
                        signal_type = "BUY"
                    elif histogram[-1] < 0 and histogram[-2] >= 0:
                        reasons.append("MACD死叉")
                        strength += 0.35
                        signal_type = "SELL"

                if len(k) >= 2 and len(d) >= 2:
                    if k[-1] > d[-1] and k[-2] <= d[-2] and k[-1] < 80:
                        reasons.append("KDJ金叉")
                        strength += 0.35
                        if signal_type == "NONE":
                            signal_type = "BUY"
                    elif k[-1] < d[-1] and k[-2] >= d[-2] and k[-1] > 20:
                        reasons.append("KDJ死叉")
                        strength += 0.35
                        if signal_type == "NONE":
                            signal_type = "SELL"

                base_result = {
                    "type": signal_type,
                    "strength": min(strength, 1.0),
                    "reason": " + ".join(reasons) if reasons else "无信号",
                    "mode": "indicators",
                }
        except Exception as e:
            base_result = {"type": "NONE", "strength": 0, "reason": f"指标计算错误: {e}", "mode": "indicators"}

        # 两者取强者
        if momentum.get("strength", 0) > base_result.get("strength", 0):
            base_result = momentum

        self.latest_signals[symbol] = base_result
        return base_result

    def calc_momentum_signal(self, symbol: str) -> dict:
        """
        动量追涨/追跌信号：
        - 看 lookback 内单边涨跌幅(%)；
        - 要求最近 min_consecutive_ticks 连续同方向；
        - 点差过大则过滤。
        """
        if not MOMENTUM_CONFIG.get("enabled", True):
            return {"type": "NONE", "strength": 0, "reason": "动量策略关闭", "mode": "momentum"}

        prices = self.price_history.get(symbol, [])
        price_info = self.latest_prices.get(symbol, {})
        if not prices or not price_info:
            return {"type": "NONE", "strength": 0, "reason": "无价格数据", "mode": "momentum"}

        lookback = int(MOMENTUM_CONFIG.get("lookback_points", 12))
        min_consecutive = int(MOMENTUM_CONFIG.get("min_consecutive_ticks", 3))
        if len(prices) < max(lookback + 1, min_consecutive + 2):
            return {"type": "NONE", "strength": 0, "reason": "动量数据不足", "mode": "momentum"}

        min_change_map = MOMENTUM_CONFIG.get("min_change_percent", {})
        min_change = float(min_change_map.get(symbol, 0.05))

        max_spread_map = MOMENTUM_CONFIG.get("max_spread_percent", {})
        max_spread_pct = float(max_spread_map.get(symbol, 0.1))

        bid = float(price_info.get("bid", 0) or 0)
        ask = float(price_info.get("ask", 0) or 0)
        mid = float(price_info.get("mid", prices[-1]) or prices[-1])
        spread = (ask - bid) if (bid > 0 and ask > 0) else float(price_info.get("spread", 0) or 0)
        spread_pct = (spread / mid * 100) if mid else 0.0
        if spread_pct > max_spread_pct:
            return {
                "type": "NONE",
                "strength": 0,
                "reason": f"点差过大 {spread_pct:.3f}%>{max_spread_pct:.3f}%",
                "mode": "momentum",
            }

        base = prices[-lookback]
        if not base:
            return {"type": "NONE", "strength": 0, "reason": "基准价异常", "mode": "momentum"}

        change_pct = (prices[-1] - base) / base * 100

        if min_consecutive <= 0:
            consecutive_up = True
            consecutive_down = True
        else:
            recent = prices[-(min_consecutive + 1):]
            diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            consecutive_up = all(d > 0 for d in diffs)
            consecutive_down = all(d < 0 for d in diffs)

        if change_pct >= min_change and consecutive_up:
            strength = min(1.0, max(0.0, change_pct / (min_change * 2)))
            return {
                "type": "BUY",
                "strength": float(strength),
                "reason": f"追涨动量 {change_pct:.3f}% (近{lookback}tick)",
                "mode": "momentum",
                "change_percent": change_pct,
                "spread_percent": spread_pct,
            }

        if change_pct <= -min_change and consecutive_down:
            strength = min(1.0, max(0.0, (-change_pct) / (min_change * 2)))
            return {
                "type": "SELL",
                "strength": float(strength),
                "reason": f"追跌动量 {change_pct:.3f}% (近{lookback}tick)",
                "mode": "momentum",
                "change_percent": change_pct,
                "spread_percent": spread_pct,
            }

        return {
            "type": "NONE",
            "strength": 0,
            "reason": f"动量不足 {change_pct:.3f}% (阈值 {min_change:.3f}%)",
            "mode": "momentum",
            "change_percent": change_pct,
            "spread_percent": spread_pct,
        }

    # ==================== 交易执行 ====================

    def _build_market_data(self, symbol: str) -> dict:
        """
        构建市场数据字典供自适应策略使用

        Returns:
            包含 volatility, trend_strength, momentum 等的字典
        """
        prices = self.price_history.get(symbol, [])
        price_info = self.latest_prices.get(symbol, {})
        signal = self.latest_signals.get(symbol, {})

        # 计算波动率
        volatility = 0.0
        if len(prices) >= 10:
            recent = prices[-10:]
            volatility = np.std(recent) / np.mean(recent) if np.mean(recent) else 0

        # 计算趋势强度
        trend_strength = 0.0
        if len(prices) >= 20:
            first_half = np.mean(prices[-20:-10])
            second_half = np.mean(prices[-10:])
            if first_half:
                trend_strength = (second_half - first_half) / first_half

        # 计算动量
        momentum = 0.0
        if len(prices) >= 5:
            momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] else 0

        # 计算点差
        spread = 0.0
        if price_info:
            bid = float(price_info.get("bid", 0) or 0)
            ask = float(price_info.get("ask", 0) or 0)
            mid = (bid + ask) / 2 if bid and ask else 0
            spread = (ask - bid) / mid if mid else 0

        return {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "spread": spread,
            "recent_prices": prices[-20:] if prices else [],
            "indicators": {},
            "signal_strength": signal.get("strength", 0),
            "signal_type": signal.get("type", "NONE"),
        }

    def should_trade(self, symbol: str) -> tuple:
        """
        判断是否应该交易

        整合自适应进化策略：
        - 使用贝叶斯学习得到的信号阈值
        - 根据市场上下文判断是否值得交易
        - 动态调整交易参数

        Returns:
            (should_trade: bool, direction: str, reason: str)
        """
        if not self.auto_trade:
            return (False, "", "自动交易已关闭")

        # 检查冷却时间
        last_time = self._last_trade_time.get(symbol, 0)
        if time.time() - last_time < self.trade_config["trade_cooldown"]:
            return (False, "", "冷却中")

        # ========== 核心风控：持仓方向与盈亏检查 ==========
        # 先获取信号方向（后面要用来判断是否反向）
        signal = self.latest_signals.get(symbol, {})
        signal_type = signal.get("type", "NONE")
        if signal_type == "BUY":
            new_direction = "long"
        elif signal_type == "SELL":
            new_direction = "short"
        else:
            new_direction = ""

        # 检查是否已有该品种持仓
        if symbol in self.positions:
            pos = self.positions[symbol]
            price_info = self.latest_prices.get(symbol, {})
            pnl_pct = self._unrealized_pnl_percent(pos, price_info) if price_info else 0

            # 情况1：反向信号
            if new_direction and pos.direction != new_direction:
                mode = str(REVERSE_CONFIG.get("mode", "only_when_losing") or "only_when_losing").strip()

                # Rule set A (your latest requirement): if the existing opposite position is winning, keep it.
                if bool(REVERSE_CONFIG.get("enabled", True)) and mode == "only_when_losing":
                    if pnl_pct >= 0:
                        return (False, "", f"反向信号但当前仓盈利({pnl_pct:.2f}%)，放弃反手")
                    # Losing opposite position: allow close-then-open reverse.
                    self._pending_reverse = {
                        "symbol": symbol,
                        "old_direction": pos.direction,
                        "new_direction": new_direction,
                        "pnl_pct": pnl_pct
                    }
                    return (True, new_direction, f"反向信号且亏损({pnl_pct:.2f}%)，先平后开")

                # Rule set B (previous): require a profit cushion to reverse; if losing, close-only.
                if bool(REVERSE_CONFIG.get("enabled", True)) and mode == "profit_cushion":
                    cushion = float(REVERSE_CONFIG.get("min_profit_cushion_percent", 1.0) or 1.0)
                    if pnl_pct < 0:
                        self._pending_close_only = {
                            "symbol": symbol,
                            "direction": pos.direction,
                            "pnl_pct": pnl_pct,
                            "reason": "亏损止损"
                        }
                        return (False, "", f"反向信号但亏损({pnl_pct:.2f}%)，只平仓不反向开仓")
                    if pnl_pct < cushion:
                        return (False, "", f"反向信号但安全垫不足({pnl_pct:.2f}%<{cushion:.2f}%)，不反转")
                    self._pending_reverse = {
                        "symbol": symbol,
                        "old_direction": pos.direction,
                        "new_direction": new_direction,
                        "pnl_pct": pnl_pct
                    }
                    return (True, new_direction, f"反向信号({pos.direction}→{new_direction})，先平后开(安全垫{pnl_pct:.2f}%)")

                # If REVERSE_CONFIG is disabled: never reverse automatically.
                return (False, "", "反向信号但禁止自动反手")

            # 情况2：同向信号 + 亏损 → 严格禁止
            if pos.direction == new_direction and pnl_pct < 0:
                return (False, "", f"同向持仓亏损中({pnl_pct:.2f}%)，禁止加仓")

            # 情况3：同向信号 + 盈利 → 允许开新仓加仓（保留原仓）
            if pos.direction == new_direction and pnl_pct > 0:
                if bool(ADD_ON_SIGNAL_CONFIG.get("enabled", True)):
                    min_profit = float(ADD_ON_SIGNAL_CONFIG.get("min_profit_percent", 0.2) or 0.2)
                    if pnl_pct < min_profit:
                        return (False, "", f"同向盈利但未达加仓阈值({pnl_pct:.2f}%<{min_profit:.2f}%)")

                    # Add-on-signal cooldown
                    cooldown = float(ADD_ON_SIGNAL_CONFIG.get("cooldown_sec", 60) or 60)
                    if pos.last_add_time and (time.time() - pos.last_add_time) < cooldown:
                        return (False, "", "加仓冷却中")

                    max_adds = int(PYRAMID_CONFIG.get("max_adds_per_symbol", {}).get(symbol, 0))
                    if pos.add_count >= max_adds:
                        return (False, "", "已达最大加仓次数")

                    settings = self.symbol_settings.get(symbol, {})
                    base_lot = float(pos.initial_lot_size or settings.get("lot_size", 0.01))
                    ratio = float(ADD_ON_SIGNAL_CONFIG.get("add_lot_ratio", 1.0) or 1.0)
                    add_lot = round(base_lot * ratio, 3)
                    if add_lot <= 0:
                        return (False, "", "加仓手数无效")

                    self._pending_add = {
                        "symbol": symbol,
                        "direction": pos.direction,
                        "pnl_pct": pnl_pct,
                        "add_lot": add_lot,
                    }
                    return (True, pos.direction, f"同向盈利加仓({pnl_pct:.2f}%)")

                # If disabled, fall back to pyramiding logic only.
                return (False, "", f"同向持仓盈利({pnl_pct:.2f}%)，等待加仓时机")

            return (False, "", "已有持仓，跳过开新仓")

        # 检查其他品种是否有亏损仓位（有亏损时禁止开新仓）
        for pos_symbol, pos in self.positions.items():
            price_info = self.latest_prices.get(pos_symbol, {})
            if price_info:
                pnl_pct = self._unrealized_pnl_percent(pos, price_info)
                if pnl_pct < 0:
                    return (False, "", f"存在亏损仓位({pos_symbol} {pnl_pct:.2f}%)")

        # 检查总持仓数
        if len(self.positions) >= self.trade_config["max_total_positions"]:
            return (False, "", "持仓数已满")

        # ========== 核心进化：获取自适应参数 ==========
        market_data = self._build_market_data(symbol)

        # 检查市场上下文是否值得交易（基于历史学习）
        try:
            should_trade_ctx, ctx_reason = self.evolution_engine.should_trade_in_context(market_data)
            if not should_trade_ctx:
                return (False, "", f"进化策略: {ctx_reason}")
        except Exception as e:
            self.logger.debug(f"上下文检查失败: {e}")
            # 失败时使用默认行为

        # ========== 核心进化：检查经验教训警告 ==========
        try:
            should_avoid, warning = self.evolution_engine.check_lessons_warning(market_data)
            if should_avoid:
                return (False, "", warning)
        except Exception as e:
            self.logger.debug(f"教训检查失败: {e}")

        # 获取自适应参数
        try:
            adapted_params = self.evolution_engine.get_adapted_parameters(symbol, market_data)
            signal_threshold = adapted_params.get("signal_threshold", self.trade_config["min_signal_strength"])
        except Exception as e:
            self.logger.debug(f"获取自适应参数失败: {e}")
            signal_threshold = self.trade_config["min_signal_strength"]

        # 获取信号和预测
        signal = self.latest_signals.get(symbol, {})
        pred = self.latest_predictions.get(symbol, {})

        signal_type = signal.get("type", "NONE")
        signal_strength = signal.get("strength", 0)
        pred_direction = pred.get("direction", "NEUTRAL")
        pred_confidence = pred.get("confidence", 0)
        signal_mode = signal.get("mode", "")

        # 使用自适应信号阈值（核心进化点）
        if signal_strength < signal_threshold:
            return (False, "", f"信号强度不足 {signal_strength:.0%} (阈值:{signal_threshold:.0%})")

        # 追涨/追跌动量：允许不依赖预测模型（更贴近操盘手的"看到强就跟"）
        if signal_mode != "momentum":
            # 检查预测置信度
            if pred_confidence < self.trade_config["min_prediction_confidence"]:
                return (False, "", f"预测置信度不足 {pred_confidence:.0%}")

            # 检查信号和预测是否一致
            if self.trade_config["require_signal_prediction_agree"]:
                if signal_type == "BUY" and pred_direction != "UP":
                    return (False, "", "信号与预测不一致")
                if signal_type == "SELL" and pred_direction != "DOWN":
                    return (False, "", "信号与预测不一致")

        # 确定方向
        if signal_type == "BUY":
            if signal_mode == "momentum":
                return (True, "long", f"信号:{signal.get('reason')}")
            return (True, "long", f"信号:{signal.get('reason')} 预测:{pred.get('reason')}")
        elif signal_type == "SELL":
            if signal_mode == "momentum":
                return (True, "short", f"信号:{signal.get('reason')}")
            return (True, "short", f"信号:{signal.get('reason')} 预测:{pred.get('reason')}")

        return (False, "", "无明确方向")

    async def execute_trade(self, symbol: str, direction: str, reason: str) -> bool:
        """执行交易"""
        if self.live_trade:
            if bool(self._manual_trade_stop):
                self._trade_event_logger.log(now_event(
                    "trade_blocked",
                    symbol=symbol,
                    direction=direction,
                    reason="manual_stop_file_present",
                ))
                return False
            if not bool(self.exec_config.get("allow_live_open", True)):
                self._trade_event_logger.log(now_event(
                    "trade_blocked",
                    symbol=symbol,
                    direction=direction,
                    reason="live_open_disabled",
                ))
                return False

        # ========== 同向加仓：允许在已有仓位基础上再开一单 ==========
        if self._pending_add and self._pending_add.get("symbol") == symbol:
            add = dict(self._pending_add)
            self._pending_add = {}
            if symbol not in self.positions:
                return False
            pos = self.positions[symbol]
            if pos.direction != direction:
                return False
            add_lot = float(add.get("add_lot", 0) or 0)
            if add_lot <= 0:
                return False

            price_info = self.latest_prices.get(symbol, {})
            before_count = await self._count_platform_positions(symbol) if self.live_trade else 0

            used_list_trade = False
            if self.current_symbol != symbol:
                allow_switch_for_open = bool(self.exec_config.get("allow_switch_for_open", False))
                if not allow_switch_for_open:
                    if bool(self.exec_config.get("allow_list_trade", True)):
                        ok = await (self.fetcher.click_buy_from_list(symbol) if direction == "long" else self.fetcher.click_sell_from_list(symbol))
                        if not ok:
                            self._trade_event_logger.log(now_event(
                                "add_blocked",
                                symbol=symbol,
                                direction=direction,
                                reason="switch_disabled_and_list_trade_failed",
                                add_lot=add_lot,
                                current_symbol=self.current_symbol,
                            ))
                            return False
                        used_list_trade = True
                        await asyncio.sleep(0.3)
                    else:
                        return False
                else:
                    ok = await self.fetcher.switch_symbol(symbol)
                    if ok:
                        self.current_symbol = symbol
                        await asyncio.sleep(1)
                    else:
                        return False

            # Place the additional order.
            if self.live_trade:
                lot_ok = await self.fetcher.set_lot_size(add_lot)
                if not lot_ok and bool(self.exec_config.get("require_lot_set", False)):
                    self._trade_event_logger.log(now_event(
                        "add_blocked",
                        symbol=symbol,
                        direction=direction,
                        reason="set_lot_size_failed",
                        add_lot=add_lot,
                    ))
                    return False
                ok = True if used_list_trade else (await self.fetcher.click_buy() if direction == "long" else await self.fetcher.click_sell())
                if not ok:
                    return False
                await asyncio.sleep(0.4)
                if bool(self.exec_config.get("require_confirm", True)) and await self._trade_dialog_visible():
                    ok = await self._click_confirm("市价买入" if direction == "long" else "市价卖出")
                    await asyncio.sleep(0.8)
                if not ok:
                    return False
                if bool(self.exec_config.get("verify_open_after_trade", True)):
                    if before_count > 0:
                        ok = await self._wait_platform_position_count_increased(symbol, before=int(before_count))
                    else:
                        ok = await self._wait_platform_position(symbol, should_exist=True)
                    if not ok:
                        return False
            else:
                ok = True
            if not ok:
                return False

            # Merge into aggregated position tracking.
            add_price = float(price_info.get("ask" if direction == "long" else "bid", 0) or 0)
            if add_price > 0 and pos.entry_price > 0:
                total_lot = pos.lot_size + add_lot
                pos.entry_price = (pos.entry_price * pos.lot_size + add_price * add_lot) / total_lot
                pos.lot_size = total_lot
            else:
                pos.lot_size += add_lot
            pos.add_count += 1
            pos.last_add_time = time.time()
            self._save_positions()
            self._last_trade_time[symbol] = time.time()
            self._trade_event_logger.log(now_event(
                "add_executed_signal",
                symbol=symbol,
                direction=direction,
                add_lot=add_lot,
                pnl_percent=float(add.get("pnl_pct", 0) or 0),
                total_lot=pos.lot_size,
                add_count=pos.add_count,
                reason=reason,
            ))
            return True

        # ========== 核心防护：禁止重复开仓 ==========
        # 如果已有持仓且不是反向开仓流程，直接拒绝
        if symbol in self.positions and not self._pending_reverse:
            self.logger.warning(f"[防重复开仓] {symbol} 已有持仓，拒绝开新仓")
            self._trade_event_logger.log(now_event(
                "trade_blocked",
                symbol=symbol,
                direction=direction,
                reason="already_has_position",
                existing_direction=self.positions[symbol].direction,
            ))
            return False

        # 获取当前价格
        price_info = self.latest_prices.get(symbol, {})
        if not price_info:
            return False

        # ========== 反向开仓处理：先平旧仓 ==========
        if self._pending_reverse and self._pending_reverse.get("symbol") == symbol:
            old_dir = self._pending_reverse.get("old_direction")
            new_dir = self._pending_reverse.get("new_direction")
            pnl_pct = self._pending_reverse.get("pnl_pct", 0)

            self.logger.info(f"[反向开仓] {symbol}: {old_dir}→{new_dir}, 当前盈亏: {pnl_pct:.2f}%")
            self._trade_event_logger.log(now_event(
                "reverse_trade",
                symbol=symbol,
                old_direction=old_dir,
                new_direction=new_dir,
                pnl_percent=pnl_pct,
            ))

            # 先平掉旧仓位
            close_ok = await self.close_position(symbol, f"反向信号平仓({old_dir}→{new_dir})")
            if not close_ok:
                self.logger.warning(f"[反向开仓] 平仓失败，取消开新仓: {symbol}")
                self._pending_reverse = {}
                return False

            # 清除反向标记
            self._pending_reverse = {}

            # 等待一下再开新仓
            await asyncio.sleep(1.0)

            # ========== 关键检查：确认仓位已被清空 ==========
            # 必须验证仓位确实被平掉了，否则开新仓会导致锁仓！
            if symbol in self.positions:
                self.logger.error(f"[反向开仓失败] {symbol} 平仓后仓位仍存在，取消开新仓（防止锁仓）")
                self._trade_event_logger.log(now_event(
                    "reverse_trade_aborted",
                    symbol=symbol,
                    reason="position_still_exists_after_close",
                    existing_direction=self.positions[symbol].direction,
                ))
                return False

            # 重新获取价格（平仓后价格可能变化）
            await self.fetch_prices()
            price_info = self.latest_prices.get(symbol, {})
            if not price_info:
                return False

        # Refresh account for risk-based sizing
        await self.refresh_account()

        # 切换到目标品种
        used_list_trade = False
        if self.current_symbol != symbol:
            allow_switch_for_open = bool(self.exec_config.get("allow_switch_for_open", False))
            if not allow_switch_for_open:
                # 不切换模式：尝试从列表行直接下单（快速交易）
                if bool(self.exec_config.get("allow_list_trade", True)):
                    if direction == "long":
                        ok = await self.fetcher.click_buy_from_list(symbol)
                    else:
                        ok = await self.fetcher.click_sell_from_list(symbol)
                    if not ok:
                        self._trade_event_logger.log(now_event(
                            "trade_blocked",
                            symbol=symbol,
                            direction=direction,
                            reason="switch_disabled_and_list_trade_failed",
                            current_symbol=self.current_symbol,
                        ))
                        return False
                    used_list_trade = True
                    await asyncio.sleep(0.3)
                else:
                    self._trade_event_logger.log(now_event(
                        "trade_blocked",
                        symbol=symbol,
                        direction=direction,
                        reason="switch_disabled_for_open",
                        current_symbol=self.current_symbol,
                    ))
                    return False

            if allow_switch_for_open and not used_list_trade:
                self.logger.info(f"切换品种: {self.current_symbol} -> {symbol}")
                success = await self.fetcher.switch_symbol(symbol)
                if success:
                    self.current_symbol = symbol
                    await asyncio.sleep(1)
                else:
                    self.logger.warning(f"切换品种失败: {symbol}")
                    return False

        # Position sizing (fixed or risk_percent)
        entry_px_for_sizing = float(price_info.get("ask") if direction == "long" else price_info.get("bid") or 0)
        lot_size = self._compute_lot_size(symbol, entry_px_for_sizing)

        # ========== 最终防线：开仓前再次检查是否已有仓位 ==========
        # 这是防止锁仓的最后一道防线
        if symbol in self.positions:
            self.logger.error(f"[最终防线] {symbol} 已有仓位，拒绝开仓（防止锁仓）")
            self._trade_event_logger.log(now_event(
                "trade_blocked_final_check",
                symbol=symbol,
                direction=direction,
                reason="position_exists_before_click",
                existing_direction=self.positions[symbol].direction,
            ))
            return False

        # 执行下单：实盘模式下做重试，尽量"要么下进去，要么明确失败并留证据"
        executed = bool(self.live_trade)
        lot_set_ok = False
        lot_value_after_set = ""
        click_ok = False
        confirm_ok = False
        max_retries = 3 if executed else 1
        if direction == "long":
            entry_price = price_info['ask']
            self.logger.info(f"{'[LIVE]' if executed else '[PAPER]'} 执行买入: {symbol} @ {entry_price}, 手数: {lot_size}")

            if self.live_trade:
                before_count = await self._count_platform_positions(symbol)
                require_confirm = bool(self.exec_config.get("require_confirm", True))
                # 如果已经通过列表交易下单，跳过主界面下单（避免重复开仓）
                if used_list_trade:
                    self.logger.info(f"已通过列表快速交易下单，跳过主界面下单")
                    click_ok = True
                    # 列表点击后通常也会弹“新订单”窗口，仍需要设置手数/确认
                    lot_set_ok = await self.fetcher.set_lot_size(lot_size)
                    lot_value_after_set = getattr(self.fetcher, "last_lot_value", "") or ""
                    await asyncio.sleep(0.2)
                    if require_confirm and await self._trade_dialog_visible():
                        confirm_ok = await self._click_confirm("市价买入")
                    else:
                        confirm_ok = True
                else:
                    for _ in range(max_retries):
                        click_ok = await self.fetcher.click_buy()
                        if not click_ok:
                            await asyncio.sleep(0.8)
                            continue
                        await asyncio.sleep(0.5)
                        # 在弹窗中设置手数
                        lot_set_ok = await self.fetcher.set_lot_size(lot_size)
                        lot_value_after_set = getattr(self.fetcher, "last_lot_value", "") or ""
                        if not lot_set_ok:
                            self._trade_event_logger.log(now_event(
                                "trade_warning",
                                symbol=symbol,
                                direction=direction,
                                reason="set_lot_size_failed_using_default",
                                requested_lot=lot_size,
                            ))
                            if bool(self.exec_config.get("require_lot_set", False)):
                                self._trade_event_logger.log(now_event(
                                    "trade_blocked",
                                    symbol=symbol,
                                    direction=direction,
                                    reason="set_lot_size_failed",
                                    requested_lot=lot_size,
                                ))
                                return False
                        await asyncio.sleep(0.3)
                        if require_confirm and await self._trade_dialog_visible():
                            confirm_ok = await self._click_confirm("市价买入")
                            await asyncio.sleep(1)
                            if confirm_ok:
                                break
                            await asyncio.sleep(0.8)
                        else:
                            confirm_ok = True
                            break
                success = click_ok and confirm_ok
            else:
                success = True
                lot_set_ok = True
                lot_value_after_set = str(lot_size)
                click_ok = True
                confirm_ok = True

        else:  # short
            entry_price = price_info['bid']
            self.logger.info(f"{'[LIVE]' if executed else '[PAPER]'} 执行卖出: {symbol} @ {entry_price}, 手数: {lot_size}")

            if self.live_trade:
                before_count = await self._count_platform_positions(symbol)
                require_confirm = bool(self.exec_config.get("require_confirm", True))
                # 如果已经通过列表交易下单，跳过主界面下单（避免重复开仓）
                if used_list_trade:
                    self.logger.info(f"已通过列表快速交易下单，跳过主界面下单")
                    click_ok = True
                    lot_set_ok = await self.fetcher.set_lot_size(lot_size)
                    lot_value_after_set = getattr(self.fetcher, "last_lot_value", "") or ""
                    await asyncio.sleep(0.2)
                    if require_confirm and await self._trade_dialog_visible():
                        confirm_ok = await self._click_confirm("市价卖出")
                    else:
                        confirm_ok = True
                else:
                    for _ in range(max_retries):
                        click_ok = await self.fetcher.click_sell()
                        if not click_ok:
                            await asyncio.sleep(0.8)
                            continue
                        await asyncio.sleep(0.5)
                        # 在弹窗中设置手数
                        lot_set_ok = await self.fetcher.set_lot_size(lot_size)
                        lot_value_after_set = getattr(self.fetcher, "last_lot_value", "") or ""
                        if not lot_set_ok:
                            self._trade_event_logger.log(now_event(
                                "trade_warning",
                                symbol=symbol,
                                direction=direction,
                                reason="set_lot_size_failed_using_default",
                                requested_lot=lot_size,
                            ))
                            if bool(self.exec_config.get("require_lot_set", False)):
                                self._trade_event_logger.log(now_event(
                                    "trade_blocked",
                                    symbol=symbol,
                                    direction=direction,
                                    reason="set_lot_size_failed",
                                    requested_lot=lot_size,
                                ))
                                return False
                        await asyncio.sleep(0.3)
                        if require_confirm and await self._trade_dialog_visible():
                            confirm_ok = await self._click_confirm("市价卖出")
                            await asyncio.sleep(1)
                            if confirm_ok:
                                break
                            await asyncio.sleep(0.8)
                        else:
                            confirm_ok = True
                            break
                success = click_ok and confirm_ok
            else:
                success = True
                lot_set_ok = True
                lot_value_after_set = str(lot_size)
                click_ok = True
                confirm_ok = True

        if not success:
            self._trade_event_logger.log(now_event(
                "trade_failed",
                symbol=symbol,
                direction=direction,
                reason="click_or_confirm_failed",
                requested_lot=float(lot_size),
                lot_set_ok=bool(lot_set_ok),
                lot_value_after_set=str(lot_value_after_set),
                click_ok=bool(click_ok),
                confirm_ok=bool(confirm_ok),
            ))
            await self._screenshot(f"trade_failed_{symbol}")
            return False
        await self._screenshot(f"trade_after_{symbol}")

        # Verify the order actually created a platform position (prevents “clicked but not placed”).
        if self.live_trade and bool(self.exec_config.get("verify_open_after_trade", True)):
            if int(before_count) <= 0:
                ok = await self._wait_platform_position(symbol, should_exist=True)
            else:
                ok = await self._wait_platform_position_count_increased(symbol, before=int(before_count))
            if not ok:
                self._trade_event_logger.log(now_event(
                    "trade_failed",
                    symbol=symbol,
                    direction=direction,
                    reason="platform_position_not_found_after_open",
                    requested_lot=float(lot_size),
                    lot_set_ok=bool(lot_set_ok),
                    lot_value_after_set=str(lot_value_after_set),
                    click_ok=bool(click_ok),
                    confirm_ok=bool(confirm_ok),
                ))
                await self._screenshot(f"trade_failed_{symbol}")
                return False

        # 计算止盈止损（使用盈亏比）
        sl_pct = self.trade_config["stop_loss_percent"]
        reward_risk_ratio = self.trade_config.get("reward_risk_ratio", 0)

        # 如果设置了盈亏比，则止盈 = 止损 * 盈亏比
        if reward_risk_ratio > 0:
            tp_pct = sl_pct * reward_risk_ratio
            self.logger.info(f"使用盈亏比 1:{reward_risk_ratio}，止损:{sl_pct*100:.1f}% 止盈:{tp_pct*100:.1f}%")
        else:
            tp_pct = self.trade_config["take_profit_percent"]

        if direction == "long":
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)

        # 记录持仓
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            lot_size=lot_size,
            entry_time=time.time(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            add_count=0,
            max_unrealized_pnl_percent=0.0,
            last_add_time=0.0,
            initial_entry_price=float(entry_price),
            initial_lot_size=float(lot_size),
            r_distance=float(abs(entry_price - stop_loss)),
        )
        self.positions[symbol] = position
        self._save_positions()  # 持久化持仓

        # 注册到进化引擎，启动5分钟反思检查
        try:
            signal_info = self.latest_signals.get(symbol, {})
            price_info = self.latest_prices.get(symbol, {})
            prices = self.price_history.get(symbol, [])

            # 计算市场数据
            volatility = 0
            trend_strength = 0
            momentum = 0
            if len(prices) >= 10:
                returns = np.diff(prices[-10:]) / np.array(prices[-10:-1])
                volatility = float(np.std(returns)) if len(returns) > 0 else 0
                momentum = float(np.mean(returns)) if len(returns) > 0 else 0
                # 趋势强度
                if len(prices) >= 20:
                    trend_strength = (prices[-1] - prices[-20]) / prices[-20] if prices[-20] else 0

            market_data = {
                "volatility": volatility,
                "trend_strength": trend_strength,
                "momentum": momentum,
                "spread": float(price_info.get("spread", 0) or 0),
                "indicators": {},
                "recent_prices": prices[-20:] if prices else []
            }

            self.evolution_engine.register_position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=time.time(),
                signal_info=signal_info,
                market_data=market_data
            )
        except Exception as e:
            self.logger.warning(f"进化引擎注册失败: {e}")

        # 记录交易
        record = TradeRecord(
            symbol=symbol,
            action="BUY" if direction == "long" else "SELL",
            direction=direction,
            price=entry_price,
            lot_size=lot_size,
            timestamp=time.time(),
            reason=reason,
            executed=executed,
            requested_lot=float(lot_size),
            lot_set_ok=bool(lot_set_ok),
            lot_value_after_set=str(lot_value_after_set),
        )
        self.trade_history.append(record)
        self._save_trade_record(record)

        # 更新冷却时间
        self._last_trade_time[symbol] = time.time()

        # 声音提醒
        try:
            winsound.Beep(1200, 300)
            winsound.Beep(1500, 300)
        except:
            pass

        self.logger.info(f"{'实盘' if executed else '纸面'} 开仓记录: {symbol} {direction} @ {entry_price}")
        return True

    def _unrealized_pnl_percent(self, position: Position, price_info: dict) -> float:
        if not price_info or not position.entry_price:
            return 0.0
        bid = float(price_info.get("bid", 0) or 0)
        ask = float(price_info.get("ask", 0) or 0)
        mid = float(price_info.get("mid", 0) or 0)

        # 用更保守的可成交价估算浮盈（避免“看着盈利其实一扣点差就亏”导致误加仓/误触发）
        if position.direction == "long":
            px = bid or mid
            return (px - position.entry_price) / position.entry_price * 100 if px else 0.0
        px = ask or mid
        return (position.entry_price - px) / position.entry_price * 100 if px else 0.0

    def _reward_add_multiplier(self) -> float:
        if not REWARD_CONFIG.get("enabled", True):
            return 1.0
        bonus = float(REWARD_CONFIG.get("add_lot_bonus_per_point", 0.0))
        cap = float(REWARD_CONFIG.get("max_add_lot_multiplier", 2.0))
        mult = 1.0 + max(0, self.reward_points) * bonus
        return float(min(mult, cap))

    async def maybe_add_position(self, symbol: str, current_mid: float) -> bool:
        if not PYRAMID_CONFIG.get("enabled", True):
            return False
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        settings = self.symbol_settings.get(symbol, {})
        base_lot = float(pos.initial_lot_size or settings.get("lot_size", 0.01))
        max_adds = int(PYRAMID_CONFIG.get("max_adds_per_symbol", {}).get(symbol, 0))
        if pos.add_count >= max_adds:
            return False

        price_info = self.latest_prices.get(symbol, {})
        pnl_pct = self._unrealized_pnl_percent(pos, price_info)

        # 只在“已形成安全垫的盈利单”上加仓：绝不向亏损加仓
        min_profit = float(PYRAMID_CONFIG.get("min_profit_percent_to_add", 0.05))
        arm_profit = float(BREAKEVEN_CLOSE_CONFIG.get("arm_profit_percent", 0.03))
        if pnl_pct <= 0:
            return False
        if pos.max_unrealized_pnl_percent < max(min_profit, arm_profit):
            return False
        if PYRAMID_CONFIG.get("mode", "percent") == "percent":
            if pnl_pct < min_profit:
                return False
        else:
            r_dist = float(pos.r_distance or 0.0)
            if r_dist <= 0:
                return False
            bid = float(price_info.get("bid", 0) or 0)
            ask = float(price_info.get("ask", 0) or 0)
            mid = float(price_info.get("mid", 0) or 0)
            anchor = float(pos.initial_entry_price or pos.entry_price)
            if pos.direction == "long":
                px = bid or mid
                favorable_move = (px - anchor) if px else 0.0
            else:
                px = ask or mid
                favorable_move = (anchor - px) if px else 0.0
            move_r = favorable_move / r_dist if r_dist else 0.0
            first_add_r = float(PYRAMID_CONFIG.get("first_add_r", 0.5))
            add_every_r = float(PYRAMID_CONFIG.get("add_every_r", 0.5))
            next_r = first_add_r + pos.add_count * add_every_r
            if move_r < next_r:
                return False

            if PYRAMID_CONFIG.get("require_momentum_same_direction", True):
                sig = self.latest_signals.get(symbol, {})
                if sig.get("mode") != "momentum":
                    return False
                if float(sig.get("strength", 0) or 0) < float(PYRAMID_CONFIG.get("min_signal_strength_for_add", 0.5)):
                    return False
                if pos.direction == "long" and sig.get("type") != "BUY":
                    return False
                if pos.direction == "short" and sig.get("type") != "SELL":
                    return False

        cooldown = float(PYRAMID_CONFIG.get("add_cooldown_sec", 60))
        if pos.last_add_time and (time.time() - pos.last_add_time) < cooldown:
            return False

        ratio = float(PYRAMID_CONFIG.get("add_lot_ratio", 0.5))
        add_lot = base_lot * ratio * self._reward_add_multiplier()
        add_lot = round(add_lot, 3)
        if add_lot <= 0:
            return False

        # 加仓也属于“开仓”，遵循同样的切换/列表下单策略
        used_list_trade = False
        if self.current_symbol != symbol:
            allow_switch_for_open = bool(self.exec_config.get("allow_switch_for_open", False))
            if not allow_switch_for_open:
                if bool(self.exec_config.get("allow_list_trade", True)):
                    if pos.direction == "long":
                        ok = await self.fetcher.click_buy_from_list(symbol)
                    else:
                        ok = await self.fetcher.click_sell_from_list(symbol)
                    if not ok:
                        self._trade_event_logger.log(now_event(
                            "add_blocked",
                            symbol=symbol,
                            direction=pos.direction,
                            reason="switch_disabled_and_list_trade_failed",
                            add_lot=add_lot,
                            current_symbol=self.current_symbol,
                        ))
                        return False
                    used_list_trade = True
                    await asyncio.sleep(0.3)
                else:
                    self._trade_event_logger.log(now_event(
                        "add_blocked",
                        symbol=symbol,
                        direction=pos.direction,
                        reason="switch_disabled_for_open",
                        add_lot=add_lot,
                        current_symbol=self.current_symbol,
                    ))
                    return False

            if allow_switch_for_open and not used_list_trade:
                success = await self.fetcher.switch_symbol(symbol)
                if success:
                    self.current_symbol = symbol
                    await asyncio.sleep(1)
                else:
                    self._trade_event_logger.log(now_event(
                        "add_blocked",
                        symbol=symbol,
                        direction=pos.direction,
                        reason="switch_symbol_failed",
                        add_lot=add_lot,
                        current_symbol=self.current_symbol,
                    ))
                    return False

        # 顺势加仓：按当前方向再开一单，然后合并为一个“加总持仓”
        if pos.direction == "long":
            add_price = float(price_info.get("ask", 0) or 0)
            if self.live_trade:
                lot_ok = await self.fetcher.set_lot_size(add_lot)
                if not lot_ok:
                    self._trade_event_logger.log(now_event(
                        "add_warning",
                        symbol=symbol,
                        direction=pos.direction,
                        reason="set_lot_size_failed_using_default",
                        add_lot=add_lot,
                    ))
                    if bool(self.exec_config.get("require_lot_set", False)):
                        self._trade_event_logger.log(now_event(
                            "add_blocked",
                            symbol=symbol,
                            direction=pos.direction,
                            reason="set_lot_size_failed",
                            add_lot=add_lot,
                        ))
                        return False
                ok = True if used_list_trade else await self.fetcher.click_buy()
                if ok:
                    await asyncio.sleep(0.5)
                    # 检查是否需要点击确认按钮
                    if self.exec_config.get("require_confirm", True):
                        ok = await self._click_confirm("市价买入")
                        await asyncio.sleep(1)
                    else:
                        self.logger.info("加仓：require_confirm=False，跳过确认按钮")
            else:
                ok = True
        else:
            add_price = float(price_info.get("bid", 0) or 0)
            if self.live_trade:
                lot_ok = await self.fetcher.set_lot_size(add_lot)
                if not lot_ok:
                    self._trade_event_logger.log(now_event(
                        "add_warning",
                        symbol=symbol,
                        direction=pos.direction,
                        reason="set_lot_size_failed_using_default",
                        add_lot=add_lot,
                    ))
                    if bool(self.exec_config.get("require_lot_set", False)):
                        self._trade_event_logger.log(now_event(
                            "add_blocked",
                            symbol=symbol,
                            direction=pos.direction,
                            reason="set_lot_size_failed",
                            add_lot=add_lot,
                        ))
                        return False
                ok = True if used_list_trade else await self.fetcher.click_sell()
                if ok:
                    await asyncio.sleep(0.5)
                    # 检查是否需要点击确认按钮
                    if self.exec_config.get("require_confirm", True):
                        ok = await self._click_confirm("市价卖出")
                        await asyncio.sleep(1)
                    else:
                        self.logger.info("加仓：require_confirm=False，跳过确认按钮")
            else:
                ok = True

        if not ok:
            self._trade_event_logger.log(now_event(
                "add_failed",
                symbol=symbol,
                direction=pos.direction,
                add_lot=add_lot,
                pnl_percent=pnl_pct,
            ))
            return False

        # 更新平均成本
        total_lot = pos.lot_size + add_lot
        pos.entry_price = (pos.entry_price * pos.lot_size + add_price * add_lot) / total_lot
        pos.lot_size = total_lot
        pos.add_count += 1
        pos.last_add_time = time.time()
        self._save_positions()  # 持久化持仓

        self._trade_event_logger.log(now_event(
            "add_executed",
            symbol=symbol,
            direction=pos.direction,
            add_lot=add_lot,
            add_price=add_price,
            new_avg_entry=pos.entry_price,
            total_lot=pos.lot_size,
            add_count=pos.add_count,
            reward_points=self.reward_points,
        ))
        return True

    async def check_positions(self):
        """检查持仓，判断是否需要平仓"""
        # ========== 同步平台持仓：检测用户手动开的仓位 ==========
        await self._sync_platform_positions()

        # ========== 异常仓位检测：清理系统错误导致的无效仓位 ==========
        await self._validate_positions()

        # ========== 处理待平仓标记（亏损时只平仓不反向开仓）==========
        if self._pending_close_only:
            symbol = self._pending_close_only.get("symbol")
            pnl_pct = self._pending_close_only.get("pnl_pct", 0)
            reason = self._pending_close_only.get("reason", "止损")
            self._pending_close_only = {}  # 清除标记

            if symbol in self.positions:
                self.logger.info(f"[只平仓] {symbol} 亏损({pnl_pct:.2f}%)执行平仓，不反向开仓")
                await self.close_position(symbol, f"反向信号止损({pnl_pct:.2f}%)")
                return  # 平仓后本轮不再检查其他逻辑

        # 首先检查所有持仓的总亏损（账户级别止损）
        if self.positions:
            total_pnl_pct = 0.0
            for sym, pos in self.positions.items():
                price_info = self.latest_prices.get(sym, {})
                if price_info:
                    total_pnl_pct += self._unrealized_pnl_percent(pos, price_info)

            # 如果总亏损超过止损线，平掉所有仓位
            sl_pct = float(self.trade_config.get("stop_loss_percent", 0.05) or 0.05)
            if total_pnl_pct <= -sl_pct * 100:  # 转换为百分比比较
                self.logger.warning(f"账户总亏损 {total_pnl_pct:.2f}% 触发止损线 {-sl_pct*100:.1f}%，平掉所有仓位")
                for sym in list(self.positions.keys()):
                    await self.close_position(sym, f"账户总止损({total_pnl_pct:.2f}%)")
                return

        for symbol, position in list(self.positions.items()):
            price_info = self.latest_prices.get(symbol, {})
            if not price_info:
                continue

            current_price = price_info['mid']
            pnl_pct = self._unrealized_pnl_percent(position, price_info)
            if pnl_pct > position.max_unrealized_pnl_percent:
                position.max_unrealized_pnl_percent = pnl_pct

            # 检查止损
            if position.direction == "long":
                if current_price <= position.stop_loss:
                    await self.close_position(symbol, "触发止损")
                elif current_price >= position.take_profit:
                    await self.close_position(symbol, "触发止盈")
            else:  # short
                if current_price >= position.stop_loss:
                    await self.close_position(symbol, "触发止损")
                elif current_price <= position.take_profit:
                    await self.close_position(symbol, "触发止盈")

            # 盈利回撤到接近 0 强平（若曾达到安全垫）
            if BREAKEVEN_CLOSE_CONFIG.get("enabled", True):
                # 注意：arm_profit_percent 是小数（如0.03=3%），需要乘100转换为百分比值
                arm = float(BREAKEVEN_CLOSE_CONFIG.get("arm_profit_percent", 0.03)) * 100  # 0.03 -> 3.0
                close_below = float(BREAKEVEN_CLOSE_CONFIG.get("close_if_pnl_below_percent", 0.0)) * 100  # 0.0 -> 0.0
                if position.max_unrealized_pnl_percent >= arm and pnl_pct <= close_below:
                    await self.close_position(symbol, f"盈利回撤强平({pnl_pct:.3f}%)")
                    continue

            # 盈利后择机加仓（顺势）
            await self.maybe_add_position(symbol, current_price)

    async def close_position(self, symbol: str, reason: str) -> bool:
        """平仓"""
        if self.live_trade:
            if bool(self._manual_trade_stop):
                self._trade_event_logger.log(now_event(
                    "close_blocked",
                    symbol=symbol,
                    reason="manual_stop_file_present",
                    close_reason=reason,
                ))
                return False
            if not bool(self.exec_config.get("allow_live_close", True)):
                self._trade_event_logger.log(now_event(
                    "close_blocked",
                    symbol=symbol,
                    reason="live_close_disabled",
                    close_reason=reason,
                ))
                return False

        # Platform may have multiple positions per symbol; do not rely solely on local tracking.
        position = self.positions.get(symbol)
        price_info = self.latest_prices.get(symbol, {})

        if not price_info:
            return False

        # 切换品种
        if self.current_symbol != symbol:
            if not bool(self.exec_config.get("allow_switch_for_close", True)):
                self._trade_event_logger.log(now_event(
                    "close_blocked",
                    symbol=symbol,
                    reason="switch_disabled_for_close",
                    current_symbol=self.current_symbol,
                    close_reason=reason,
                ))
                return False
            await self.fetcher.switch_symbol(symbol)
            self.current_symbol = symbol
            await asyncio.sleep(1)

        # 平仓操作 - 必须使用平仓按钮，而不是开反向仓！
        # 注意：点击buy/sell是开新仓，不是平仓！这会导致锁仓！
        direction = position.direction if position else ""
        if not direction:
            # fallback from platform snapshot
            try:
                platform_positions = await self.fetcher.get_open_positions()
            except Exception:
                platform_positions = []
            for p in platform_positions or []:
                if (p.get("symbol", "") or "").upper() == symbol.upper():
                    direction = (p.get("direction") or "")  # "long"/"short" if parsed
                    break

        if direction == "long":
            exit_price = price_info['bid']
        else:
            exit_price = price_info['ask']

        before_cnt = 0
        if self.live_trade and bool(self.exec_config.get("verify_close_after_trade", True)):
            before_cnt = await self._count_platform_positions(symbol)
            if before_cnt <= 0:
                self._trade_event_logger.log(now_event(
                    "close_failed",
                    symbol=symbol,
                    reason="no_platform_position_before_close",
                    close_reason=reason,
                ))
                return False

        # 使用正确的平仓方式：点击持仓上的平仓按钮
        close_success = await self.fetcher.close_position_by_symbol(symbol)
        if not close_success:
            self.logger.warning(f"[平仓失败] {symbol} 无法找到平仓按钮，取消平仓操作")
            self._trade_event_logger.log(now_event(
                "close_failed",
                symbol=symbol,
                reason="no_close_button",
                direction=direction,
            ))
            return False

        await asyncio.sleep(0.5)

        # 点击确认
        if self.live_trade:
            if bool(self.exec_config.get("require_confirm", True)) and await self._trade_dialog_visible():
                await self._click_confirm("市价")
                await asyncio.sleep(1)

        # Verify platform position is gone before mutating local state.
        if self.live_trade and bool(self.exec_config.get("verify_close_after_trade", True)):
            ok = await self._wait_platform_position_count_decreased(symbol, before=int(before_cnt))
            if not ok:
                self._trade_event_logger.log(now_event(
                    "close_failed",
                    symbol=symbol,
                    reason="platform_position_still_exists_after_close",
                    direction=direction,
                ))
                return False

        # 计算盈亏
        if direction == "long" and position:
            pnl = (exit_price - position.entry_price) * position.lot_size
            pnl_percent = (exit_price - position.entry_price) / position.entry_price
        elif direction == "short" and position:
            pnl = (position.entry_price - exit_price) * position.lot_size
            pnl_percent = (position.entry_price - exit_price) / position.entry_price
        else:
            pnl = 0.0
            pnl_percent = 0.0

        # 记录
        record = TradeRecord(
            symbol=symbol,
            action="CLOSE",
            direction=direction or (position.direction if position else ""),
            price=exit_price,
            lot_size=(position.lot_size if position else 0.0),
            timestamp=time.time(),
            reason=reason,
            pnl=pnl,
            executed=bool(self.live_trade),
            requested_lot=float(position.lot_size) if position else 0.0,
            lot_set_ok=bool(self.live_trade),
            lot_value_after_set=str(getattr(self.fetcher, "last_lot_value", "") or ""),
        )
        self.trade_history.append(record)
        self._save_trade_record(record)

        # 奖励：盈利平仓增加奖励点数（用于后续加仓力度）
        if REWARD_CONFIG.get("enabled", True) and pnl > 0:
            self.reward_points += int(REWARD_CONFIG.get("win_points", 1))
            self._trade_event_logger.log(now_event(
                "reward",
                symbol=symbol,
                pnl=pnl,
                reward_points=self.reward_points,
                reason="profit_close",
            ))

        # 从进化引擎注销（停止反思检查）+ 发送最终反馈
        try:
            if position:
                self.evolution_engine.unregister_position(
                    symbol=symbol,
                    final_pnl=pnl,
                    final_pnl_percent=pnl_percent
                )
        except Exception as e:
            self.logger.warning(f"进化引擎注销失败: {e}")

        # 移除持仓
        if symbol in self.positions:
            del self.positions[symbol]
            self._save_positions()  # 持久化持仓

        self.logger.info(f"平仓: {symbol} @ {exit_price}, 盈亏: {pnl:.2f}, 原因: {reason}")

        # 声音提醒
        try:
            winsound.Beep(800, 500)
        except:
            pass

        return True

    async def _process_dashboard_commands(self):
        """处理来自镜像仪表板的命令"""
        command_file = Path(__file__).parent / "trade_data" / "dashboard_commands.json"

        if not command_file.exists():
            return

        try:
            with open(command_file, "r", encoding="utf-8") as f:
                commands = json.load(f)
        except:
            return

        if not commands:
            return

        updated = False
        for cmd in commands:
            if cmd.get("status") != "pending":
                continue

            cmd_type = cmd.get("type")

            try:
                if cmd_type == "close_position":
                    symbol = cmd.get("symbol")
                    if symbol:
                        self.logger.info(f"[仪表板命令] 收到平仓命令: {symbol}")
                        if self.live_trade and bool(self._manual_trade_stop):
                            cmd["status"] = "failed"
                            self._trade_event_logger.log(now_event(
                                "dashboard_command_failed",
                                command_type="close_position",
                                symbol=symbol,
                                reason="manual_stop_file_present",
                            ))
                            updated = True
                            continue
                        if self.live_trade and not bool(self.exec_config.get("allow_live_close", True)):
                            cmd["status"] = "failed"
                            self._trade_event_logger.log(now_event(
                                "dashboard_command_failed",
                                command_type="close_position",
                                symbol=symbol,
                                reason="live_close_disabled",
                            ))
                            updated = True
                            continue
                        # 使用fetcher直接平仓（不依赖系统追踪的持仓）
                        success = await self.fetcher.close_position_by_symbol(symbol)
                        if success and self.live_trade and bool(self.exec_config.get("require_confirm", True)) and await self._trade_dialog_visible():
                            await self._click_confirm("市价")
                            await asyncio.sleep(0.8)
                        if success and self.live_trade and bool(self.exec_config.get("verify_close_after_trade", True)):
                            success = await self._wait_platform_position(symbol, should_exist=False)
                        if success:
                            cmd["status"] = "success"
                            self._trade_event_logger.log(now_event(
                                "dashboard_command_executed",
                                command_type="close_position",
                                symbol=symbol,
                                result="success",
                            ))
                            # 如果系统也在追踪这个持仓，从追踪中移除
                            if symbol in self.positions:
                                del self.positions[symbol]
                                self._save_positions()
                        else:
                            cmd["status"] = "failed"
                            self._trade_event_logger.log(now_event(
                                "dashboard_command_failed",
                                command_type="close_position",
                                symbol=symbol,
                                reason="close_button_not_found",
                            ))
                        updated = True

                elif cmd_type == "close_all":
                    self.logger.info("[仪表板命令] 收到全部平仓命令")
                    if self.live_trade and bool(self._manual_trade_stop):
                        cmd["status"] = "failed"
                        self._trade_event_logger.log(now_event(
                            "dashboard_command_failed",
                            command_type="close_all",
                            reason="manual_stop_file_present",
                        ))
                        updated = True
                        continue
                    if self.live_trade and not bool(self.exec_config.get("allow_live_close", True)):
                        cmd["status"] = "failed"
                        self._trade_event_logger.log(now_event(
                            "dashboard_command_failed",
                            command_type="close_all",
                            reason="live_close_disabled",
                        ))
                        updated = True
                        continue
                    closed = await self.fetcher.close_all_positions()
                    if self.live_trade and bool(self.exec_config.get("require_confirm", True)) and await self._trade_dialog_visible():
                        # Some UIs may require repeated confirms; click once as best-effort.
                        await self._click_confirm("市价")
                        await asyncio.sleep(0.8)
                    cmd["status"] = "success"
                    cmd["closed_count"] = closed
                    self._trade_event_logger.log(now_event(
                        "dashboard_command_executed",
                        command_type="close_all",
                        closed_count=closed,
                    ))
                    # 清空系统追踪的持仓
                    self.positions.clear()
                    self._save_positions()
                    updated = True

            except Exception as e:
                cmd["status"] = "failed"
                cmd["error"] = str(e)
                self.logger.error(f"[仪表板命令] 执行失败: {e}")
                updated = True

        # 保存更新后的命令状态
        if updated:
            try:
                with open(command_file, "w", encoding="utf-8") as f:
                    json.dump(commands, f, ensure_ascii=False, indent=2)
            except:
                pass

    async def _sync_platform_positions(self):
        """
        从平台同步实际持仓，检测用户手动开的仓位
        """
        try:
            platform_positions = await self.fetcher.get_open_positions()
            if not platform_positions:
                return

            # Safety: allow multiple positions per symbol ONLY if they are all same direction.
            # If the platform has both long+short for the same symbol (hedged/locked), pause order clicks.
            per_symbol_dirs: dict[str, set[str]] = {}
            per_symbol_counts: dict[str, int] = {}
            for p in platform_positions:
                sym = (p.get("symbol") or "").upper()
                if not sym:
                    continue
                per_symbol_counts[sym] = per_symbol_counts.get(sym, 0) + 1
                d = (p.get("direction") or "").lower()
                if d:
                    per_symbol_dirs.setdefault(sym, set()).add(d)

            mixed = {s: {"count": per_symbol_counts.get(s, 0), "directions": sorted(list(ds))}
                     for s, ds in per_symbol_dirs.items() if ("long" in ds and "short" in ds)}
            if mixed:
                self._trade_event_logger.log(now_event(
                    "platform_mixed_directions",
                    symbols=mixed,
                    action="pause_auto_trade",
                ))
                self._trade_paused_until = max(self._trade_paused_until, time.time() + 600)

            for pos_info in platform_positions:
                symbol = pos_info.get("symbol")
                direction = pos_info.get("direction")
                lot_size = pos_info.get("lot_size", 0.1)
                entry_price = pos_info.get("entry_price", 0)

                # 如果系统没有追踪这个持仓，添加到追踪
                if symbol and symbol not in self.positions:
                    self.logger.info(f"[同步持仓] 检测到平台持仓: {symbol} {direction} @ {entry_price}, lot={lot_size}")

                    # 计算止损止盈
                    sl_pct = self.trade_config.get("stop_loss_percent", 0.05)
                    reward_risk_ratio = self.trade_config.get("reward_risk_ratio", 4.0)
                    tp_pct = sl_pct * reward_risk_ratio if reward_risk_ratio > 0 else 0.03

                    if direction == "long":
                        stop_loss = entry_price * (1 - sl_pct) if entry_price else 0
                        take_profit = entry_price * (1 + tp_pct) if entry_price else 0
                    else:
                        stop_loss = entry_price * (1 + sl_pct) if entry_price else 0
                        take_profit = entry_price * (1 - tp_pct) if entry_price else 0

                    position = Position(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price or self.latest_prices.get(symbol, {}).get("mid", 0),
                        lot_size=lot_size,
                        entry_time=time.time(),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        add_count=0,
                        max_unrealized_pnl_percent=0.0,
                        last_add_time=0.0,
                        initial_entry_price=entry_price,
                        initial_lot_size=lot_size,
                        r_distance=abs(entry_price - stop_loss) if entry_price and stop_loss else 0,
                    )
                    self.positions[symbol] = position
                    self._save_positions()
                    self.logger.info(f"[同步持仓] 已添加到追踪: {symbol} {direction}")

        except Exception as e:
            self.logger.debug(f"同步平台持仓失败: {e}")

    async def _validate_positions(self):
        """
        验证持仓有效性，清理系统错误导致的异常仓位

        检测规则：
        1. 同一品种不应有多个持仓记录（系统bug导致）
        2. 持仓应有对应的开仓记录（孤儿仓位）
        3. 持仓时间异常（入场时间在未来或过于久远）
        """
        if not self.positions:
            return

        positions_to_close = []
        now = time.time()

        for symbol, pos in self.positions.items():
            # 检查1：入场时间是否异常
            if pos.entry_time > now + 60:  # 入场时间在未来（允许1分钟误差）
                positions_to_close.append((symbol, "入场时间异常(未来)"))
                continue

            # 检查2：是否有对应的开仓记录
            has_open_record = False
            for record in reversed(self.trade_history[-50:]):  # 只检查最近50条
                if record.symbol == symbol and record.action in ("BUY", "SELL"):
                    has_open_record = True
                    break

            if not has_open_record and len(self.trade_history) > 0:
                # 如果有交易历史但找不到开仓记录，可能是孤儿仓位
                # 但要注意：启动时加载的持仓可能没有记录，所以检查开仓时间
                if now - pos.entry_time > 300:  # 5分钟前的仓位应该有记录
                    positions_to_close.append((symbol, "无开仓记录(孤儿仓位)"))
                    continue

        # 平掉异常仓位
        for symbol, reason in positions_to_close:
            self.logger.error(f"[异常仓位检测] 即将平仓: {symbol}, 原因: {reason}")
            self._trade_event_logger.log(now_event(
                "invalid_position_detected",
                symbol=symbol,
                reason=reason,
                position_direction=self.positions[symbol].direction,
                position_entry_price=self.positions[symbol].entry_price,
                position_entry_time=self.positions[symbol].entry_time,
            ))
            await self.close_position(symbol, f"系统错误修复-{reason}")

    # ==================== 数据保存 ====================

    def _save_positions(self):
        """保存当前持仓到文件（用于重启恢复）"""
        file = self.data_dir / "open_positions.json"
        data = {
            "saved_at": time.time(),
            "positions": {s: asdict(p) for s, p in self.positions.items()}
        }
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_positions(self):
        """启动时加载未平仓持仓"""
        file = self.data_dir / "open_positions.json"
        if not file.exists():
            return

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            saved_at = data.get("saved_at", 0)
            positions_data = data.get("positions", {})

            # 检查数据是否过期（超过24小时的持仓数据不加载）
            if time.time() - saved_at > 86400:
                self.logger.warning("持仓数据超过24小时，忽略")
                return

            for symbol, pos_dict in positions_data.items():
                pos = Position(
                    symbol=pos_dict["symbol"],
                    direction=pos_dict["direction"],
                    entry_price=pos_dict["entry_price"],
                    lot_size=pos_dict["lot_size"],
                    entry_time=pos_dict["entry_time"],
                    stop_loss=pos_dict["stop_loss"],
                    take_profit=pos_dict["take_profit"],
                    add_count=pos_dict.get("add_count", 0),
                    max_unrealized_pnl_percent=pos_dict.get("max_unrealized_pnl_percent", 0),
                    last_add_time=pos_dict.get("last_add_time", 0),
                    initial_entry_price=pos_dict.get("initial_entry_price", pos_dict["entry_price"]),
                    initial_lot_size=pos_dict.get("initial_lot_size", pos_dict["lot_size"]),
                    r_distance=pos_dict.get("r_distance", 0),
                )
                self.positions[symbol] = pos
                self.logger.info(f"恢复持仓: {symbol} {pos.direction} @ {pos.entry_price}")

                # 注册到进化引擎
                try:
                    self.evolution_engine.register_position(
                        symbol=symbol,
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        entry_time=pos.entry_time,
                        signal_info={
                            "signal_type": "BUY" if pos.direction == "long" else "SELL",
                            "signal_strength": 1.0,
                            "reason": "恢复持仓"
                        },
                        market_data={"volatility": 0, "trend_strength": 0}
                    )
                except Exception as e:
                    self.logger.warning(f"恢复持仓注册进化引擎失败: {e}")

            if self.positions:
                self.logger.info(f"已恢复 {len(self.positions)} 个持仓")

        except Exception as e:
            self.logger.error(f"加载持仓失败: {e}")

    def _save_trade_record(self, record: TradeRecord):
        """保存交易记录"""
        prefix = "trades" if record.executed else "paper_trades"
        file = self.data_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _save_trade_history(self):
        """保存交易历史摘要"""
        summary = {
            "session_end": datetime.now().isoformat(),
            "total_trades": len(self.trade_history),
            "open_positions": {s: asdict(p) for s, p in self.positions.items()},
            "reward_points": self.reward_points,
            "exit_reason": self.exit_reason,
        }
        file = self.data_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # ==================== 主循环 ====================

    async def run(self, duration: int = None):
        """运行"""
        if not await self.start():
            return

        start_time = time.time()
        self.exit_reason = ""

        print("\n" + "="*80)
        print("  自动交易系统已启动")
        print("="*80)
        print(f"  监控品种: {', '.join(self.symbols)}")
        print(f"  手数配置:")
        for s in self.symbols:
            lot = self.symbol_settings.get(s, {}).get("lot_size", 0.01)
            print(f"    - {s}: {lot} 手")
        print(f"  自动交易: {'开启' if self.auto_trade else '关闭'}")
        print(f"  实盘点击: {'开启' if self.live_trade else '关闭(默认)'}")
        if MOMENTUM_CONFIG.get("enabled", True):
            lb = MOMENTUM_CONFIG.get("lookback_points", 12)
            mc = MOMENTUM_CONFIG.get("min_consecutive_ticks", 3)
            print(f"  追涨策略: 开启 | lookback={lb}tick | 连续={mc}tick")
        print("="*80 + "\n")

        try:
            while self._is_running:
                try:
                    # 心跳：至少每 60 秒落一次，确认程序“活着”
                    now_ts = time.time()
                    if (now_ts - self._last_heartbeat_ts) >= 60:
                        self._last_heartbeat_ts = now_ts
                        self._trade_event_logger.log(now_event(
                            "heartbeat",
                            symbols=self.symbols,
                            live_trade=self.live_trade,
                            open_positions=list(self.positions.keys()),
                            reward_points=self.reward_points,
                        ))

                    # 获取价格
                    await self.fetch_prices()
                    await self.refresh_account()

                    # Manual emergency stop (file-based).
                    try:
                        manual_stop = bool(self._manual_stop_file.exists())
                    except Exception:
                        manual_stop = False
                    if manual_stop != bool(self._manual_trade_stop):
                        self._manual_trade_stop = manual_stop
                        self._trade_event_logger.log(now_event(
                            "manual_trade_stop",
                            enabled=bool(manual_stop),
                            file=str(self._manual_stop_file),
                        ))

                    # 计算预测和信号
                    for symbol in self.symbols:
                        if symbol in self.latest_prices:
                            self.calc_prediction(symbol)
                            self.calc_signal(symbol)

                            # 检查是否应该交易
                            if self.auto_trade:
                                if self._manual_trade_stop:
                                    self._log_decision(symbol, False, "", "STOP_TRADING file present, skip order clicks")
                                    continue
                                if self._trade_paused_until and time.time() < self._trade_paused_until:
                                    self._log_decision(symbol, False, "", "异常频发，临时暂停下单(仅监控)")
                                    continue
                                should, direction, reason = self.should_trade(symbol)
                                self._log_decision(symbol, should, direction, reason)
                                if should:
                                    await self.execute_trade(symbol, direction, reason)
                        else:
                            # 价格缺失也落盘，便于定位“为何没出手”
                            if self.auto_trade:
                                self._log_decision(symbol, False, "", "无价格数据(自选列表不可用/页面未就绪/未登录)")

                    # 检查持仓
                    await self.check_positions()

                    # 处理来自镜像仪表板的命令
                    await self._process_dashboard_commands()

                    # 打印状态
                    self._print_status()

                    # 检查时长
                    if duration and (time.time() - start_time) >= duration:
                        self.exit_reason = f"duration_reached:{duration}"
                        self.logger.info(f"达到运行时长 {duration}s，准备停止...")
                        break

                    await asyncio.sleep(self.check_interval)

                except KeyboardInterrupt:
                    self.exit_reason = "keyboard_interrupt"
                    print("\n\n收到停止信号...")
                    break
                except Exception as e:
                    # 防止“闪退”：记录错误、截图、尝试继续
                    tb = traceback.format_exc(limit=20)
                    now_ts = time.time()
                    if now_ts - self._last_error_ts > 60:
                        self._recent_error_count = 0
                    self._last_error_ts = now_ts
                    self._recent_error_count += 1

                    self._trade_event_logger.log(now_event(
                        "runtime_error",
                        error=str(e),
                        traceback=tb,
                        recent_error_count=self._recent_error_count,
                    ))
                    await self._screenshot("runtime_error")

                    # 决不自动停机：但异常过密时短暂暂停“下单动作”，仅保留监控与记录，避免在异常状态下乱点
                    if self._recent_error_count >= 5:
                        self._trade_paused_until = time.time() + 120
                        self._trade_event_logger.log(now_event(
                            "trade_paused",
                            seconds=120,
                            reason="too_many_errors_in_60s",
                            recent_error_count=self._recent_error_count,
                        ))

                    # 轻量恢复：等一下继续
                    await asyncio.sleep(min(10.0, max(1.0, self.check_interval)))
        finally:
            if not self.exit_reason:
                self.exit_reason = "stopped"
            await self.stop()

    def _print_status(self):
        """打印状态"""
        print("\033[2J\033[H", end="")

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("="*85)
        print(f"  自动交易系统 - {now}")
        print("="*85)

        # 品种状态
        print(f"\n  {'品种':<10} {'价格':<12} {'预测':<8} {'置信度':<8} {'信号':<12} {'持仓':<10}")
        print("-"*85)

        for symbol in self.symbols:
            price_info = self.latest_prices.get(symbol, {})
            pred = self.latest_predictions.get(symbol, {})
            signal = self.latest_signals.get(symbol, {})
            position = self.positions.get(symbol)

            price = f"{price_info.get('mid', 0):.4f}" if price_info else "等待..."
            pred_dir = {"UP": "↑涨", "DOWN": "↓跌", "NEUTRAL": "→平"}.get(pred.get("direction", ""), "-")
            conf = f"{pred.get('confidence', 0):.0%}"
            sig = signal.get("type", "-")
            if sig != "NONE" and sig != "-":
                sig = f"{sig}({signal.get('strength', 0):.0%})"

            pos_str = "-"
            if position:
                pos_str = f"{position.direction[:1].upper()} {position.lot_size}"

            print(f"  {symbol:<10} {price:<12} {pred_dir:<8} {conf:<8} {sig:<12} {pos_str:<10}")

        print("-"*85)

        # 持仓详情
        if self.positions:
            print("\n  [持仓详情]")
            for symbol, pos in self.positions.items():
                current = self.latest_prices.get(symbol, {}).get('mid', 0)
                if pos.direction == "long":
                    pnl = (current - pos.entry_price) / pos.entry_price * 100
                else:
                    pnl = (pos.entry_price - current) / pos.entry_price * 100
                print(f"    {symbol}: {pos.direction} @ {pos.entry_price:.4f}, 当前盈亏: {pnl:+.2f}%")

        # 最近交易
        if self.trade_history:
            recent = self.trade_history[-1]
            print(f"\n  [最近交易] {recent.symbol} {recent.action} @ {recent.price:.4f} - {recent.reason[:30]}")

        print("="*85)
        print(f"  自动交易: {'开启' if self.auto_trade else '关闭'} | 按 Ctrl+C 停止")
        print("="*85)


# ==================== 启动入口 ====================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="自动交易系统")
    parser.add_argument("--symbols", type=str, default="USOIL,XAUUSD,ETHUSD", help="交易品种")
    parser.add_argument("--auto", action="store_true", default=True, help="启用自动交易")
    parser.add_argument("--no-auto", dest="auto", action="store_false", help="禁用自动交易(仅监控)")
    parser.add_argument("--live", action="store_true", default=False, help="实盘点击下单(默认关闭，避免误下单)")
    parser.add_argument("--paper", action="store_true", default=False, help="纸面交易(不点击下单/只记录决策与信号)")
    parser.add_argument("--headless", action="store_true", default=False, help="无界面模式：后台运行浏览器(更快更稳)")
    parser.add_argument("--interval", type=float, default=5.0, help="检查间隔(秒)")
    parser.add_argument("--duration", type=int, default=0, help="运行时长(秒), 0=无限")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    duration = args.duration if args.duration > 0 else None

    trader = AutoTrader(
        symbols=symbols,
        auto_trade=args.auto,
        check_interval=args.interval,
        live_trade=(args.live and (not args.paper)),
        headless=bool(args.headless),
    )

    await trader.run(duration)


if __name__ == "__main__":
    asyncio.run(main())
