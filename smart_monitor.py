"""
智能多品种监控器

功能：
1. 波动率报警 - 超过阈值时提醒
2. 自动切换 - 自动切换到波动最大的品种
3. 交易信号 - 结合策略生成买卖信号
4. 历史记录 - 保存数据用于分析
5. 价格预测 - 动量/趋势/突破综合预测方向
"""

import asyncio
import logging
import time
import json
import os
import winsound  # Windows 声音提醒
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

from data_fetcher import DataFetcher
from indicators import Indicators
from config import MULTI_SYMBOL_CONFIG, MACD_CONFIG
from predictor import PricePredictor, CrossMarketPredictor, Direction


# ==================== 数据结构 ====================

@dataclass
class SymbolTick:
    """品种行情"""
    symbol: str
    bid: float
    ask: float
    mid: float
    spread: float
    timestamp: float


@dataclass
class VolatilityData:
    """波动率数据"""
    symbol: str
    volatility: float          # 波动率 %
    price_change: float        # 涨跌幅 %
    atr: float                 # ATR
    trend: str                 # "up" / "down" / "sideways"
    timestamp: float


@dataclass
class TradeSignal:
    """交易信号"""
    symbol: str
    signal_type: str           # "BUY" / "SELL" / "NONE"
    strength: float            # 信号强度 0-1
    reason: str                # 原因
    price: float               # 触发价格
    timestamp: float


@dataclass
class AlertEvent:
    """报警事件"""
    alert_type: str            # "volatility" / "signal" / "switch"
    symbol: str
    message: str
    value: float
    timestamp: float


# ==================== 智能监控器 ====================

class SmartMonitor:
    """
    智能多品种监控器

    整合波动监控、自动切换、交易信号、历史记录
    """

    def __init__(
        self,
        symbols: List[str] = None,
        volatility_threshold: float = 0.1,
        auto_switch: bool = True,
        check_interval: float = 5.0,
        data_dir: str = "monitor_data",
    ):
        """
        初始化

        Args:
            symbols: 监控的品种列表
            volatility_threshold: 波动率报警阈值 (%)
            auto_switch: 是否自动切换到高波动品种
            check_interval: 检查间隔 (秒)
            data_dir: 数据保存目录
        """
        self.symbols = symbols or MULTI_SYMBOL_CONFIG.get("symbols", ["USOIL", "XAUUSD", "ETHUSD"])
        self.volatility_threshold = volatility_threshold
        self.auto_switch = auto_switch
        self.check_interval = check_interval

        # 数据目录
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # 数据获取器
        self.fetcher = DataFetcher(headless=False)

        # 价格历史 (用于计算指标)
        self.price_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.high_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.low_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.max_history = 100

        # 最新数据
        self.latest_ticks: Dict[str, SymbolTick] = {}
        self.latest_volatility: Dict[str, VolatilityData] = {}
        self.latest_signals: Dict[str, TradeSignal] = {}

        # 当前选中的品种
        self.current_symbol: str = ""

        # 报警历史
        self.alerts: List[AlertEvent] = []
        self.max_alerts = 100

        # 历史记录文件
        self.history_file = self.data_dir / f"history_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.alerts_file = self.data_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"

        # 回调
        self.on_alert: Optional[Callable[[AlertEvent], None]] = None
        self.on_signal: Optional[Callable[[TradeSignal], None]] = None

        # 状态
        self._is_running = False
        self._last_alert_time: Dict[str, float] = {}  # 防止重复报警
        self._alert_cooldown = 60  # 报警冷却时间 (秒)

        # 预测器
        self.predictor = PricePredictor(lookback=20)
        self.cross_predictor = CrossMarketPredictor()
        self.latest_predictions: Dict[str, dict] = {}

        self.logger = logging.getLogger("SmartMonitor")

    # ==================== 连接管理 ====================

    async def start(self) -> bool:
        """启动监控"""
        self.logger.info("正在启动智能监控...")

        ok = await self.fetcher.connect()
        if not ok:
            self.logger.error("无法连接到交易页面")
            return False

        await asyncio.sleep(3)

        # 获取当前品种
        self.current_symbol = await self.fetcher.get_symbol() or ""
        self.logger.info(f"当前品种: {self.current_symbol}")

        self._is_running = True
        self.logger.info(f"智能监控已启动! 监控品种: {self.symbols}")
        return True

    async def stop(self):
        """停止监控"""
        self._is_running = False
        await self.fetcher.disconnect()
        self._save_session_summary()
        self.logger.info("智能监控已停止")

    # ==================== 数据获取 ====================

    async def fetch_all_prices(self) -> Dict[str, SymbolTick]:
        """获取所有品种价格"""
        page = self.fetcher.page
        if not page:
            return {}

        result = {}

        for symbol in self.symbols:
            try:
                item = await page.query_selector(f'.custom-list-item[data-item="{symbol}"]')
                if not item:
                    continue

                bid_elem = await item.query_selector('.item-bid span')
                ask_elem = await item.query_selector('.item-ask span')

                bid = float((await bid_elem.inner_text()).strip()) if bid_elem else 0
                ask = float((await ask_elem.inner_text()).strip()) if ask_elem else 0

                if bid > 0 and ask > 0:
                    tick = SymbolTick(
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        mid=(bid + ask) / 2,
                        spread=ask - bid,
                        timestamp=time.time()
                    )
                    result[symbol] = tick
                    self.latest_ticks[symbol] = tick

                    # 更新历史
                    self._update_history(symbol, tick)

            except Exception as e:
                self.logger.debug(f"获取 {symbol} 失败: {e}")

        return result

    def _update_history(self, symbol: str, tick: SymbolTick):
        """更新价格历史"""
        self.price_history[symbol].append(tick.mid)
        # 简化处理：用 mid 近似 high/low
        self.high_history[symbol].append(tick.ask)
        self.low_history[symbol].append(tick.bid)

        # 限制长度
        for hist in [self.price_history, self.high_history, self.low_history]:
            if len(hist[symbol]) > self.max_history:
                hist[symbol].pop(0)

    # ==================== 1. 波动率计算与报警 ====================

    def calc_volatility(self, symbol: str) -> VolatilityData:
        """计算品种波动率"""
        prices = self.price_history.get(symbol, [])
        highs = self.high_history.get(symbol, [])
        lows = self.low_history.get(symbol, [])

        if len(prices) < 5:
            return VolatilityData(
                symbol=symbol, volatility=0, price_change=0,
                atr=0, trend="sideways", timestamp=time.time()
            )

        prices_arr = np.array(prices)
        highs_arr = np.array(highs)
        lows_arr = np.array(lows)

        # 波动率 = 收益率标准差
        returns = np.diff(prices_arr) / prices_arr[:-1] * 100
        volatility = float(np.std(returns))

        # 涨跌幅
        price_change = (prices_arr[-1] - prices_arr[0]) / prices_arr[0] * 100

        # ATR (简化计算)
        tr = highs_arr - lows_arr
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))

        # 趋势判断
        if len(prices) >= 10:
            recent = prices[-10:]
            if recent[-1] > recent[0] * 1.002:
                trend = "up"
            elif recent[-1] < recent[0] * 0.998:
                trend = "down"
            else:
                trend = "sideways"
        else:
            trend = "sideways"

        data = VolatilityData(
            symbol=symbol,
            volatility=volatility,
            price_change=price_change,
            atr=atr,
            trend=trend,
            timestamp=time.time()
        )

        self.latest_volatility[symbol] = data
        return data

    def check_volatility_alert(self, data: VolatilityData) -> Optional[AlertEvent]:
        """检查波动率是否需要报警"""
        if data.volatility < self.volatility_threshold:
            return None

        # 检查冷却时间
        last_time = self._last_alert_time.get(f"vol_{data.symbol}", 0)
        if time.time() - last_time < self._alert_cooldown:
            return None

        alert = AlertEvent(
            alert_type="volatility",
            symbol=data.symbol,
            message=f"{data.symbol} 波动率突破阈值: {data.volatility:.3f}%",
            value=data.volatility,
            timestamp=time.time()
        )

        self._last_alert_time[f"vol_{data.symbol}"] = time.time()
        self._trigger_alert(alert)
        return alert

    # ==================== 2. 自动切换品种 ====================

    async def auto_switch_to_best(self) -> bool:
        """自动切换到波动最大的品种"""
        if not self.auto_switch:
            return False

        # 获取波动率排名
        ranking = self.get_volatility_ranking()
        if not ranking:
            return False

        best = ranking[0]

        # 如果最佳品种不是当前品种，且波动率超过阈值
        if (best.symbol != self.current_symbol and
            best.volatility >= self.volatility_threshold):

            self.logger.info(f"自动切换: {self.current_symbol} -> {best.symbol}")

            success = await self.fetcher.switch_symbol(best.symbol)
            if success:
                old_symbol = self.current_symbol
                self.current_symbol = best.symbol

                alert = AlertEvent(
                    alert_type="switch",
                    symbol=best.symbol,
                    message=f"自动切换品种: {old_symbol} -> {best.symbol} (波动率 {best.volatility:.3f}%)",
                    value=best.volatility,
                    timestamp=time.time()
                )
                self._trigger_alert(alert)
                return True

        return False

    def get_volatility_ranking(self) -> List[VolatilityData]:
        """获取波动率排名"""
        data_list = []
        for symbol in self.symbols:
            data = self.calc_volatility(symbol)
            if data.volatility > 0:
                data_list.append(data)

        return sorted(data_list, key=lambda x: x.volatility, reverse=True)

    # ==================== 3. 交易信号生成 ====================

    def generate_signal(self, symbol: str) -> TradeSignal:
        """生成交易信号"""
        prices = self.price_history.get(symbol, [])
        highs = self.high_history.get(symbol, [])
        lows = self.low_history.get(symbol, [])

        if len(prices) < 30:
            signal = TradeSignal(
                symbol=symbol, signal_type="NONE", strength=0,
                reason="数据不足", price=0, timestamp=time.time()
            )
            self.latest_signals[symbol] = signal
            return signal

        close = np.array(prices)
        high = np.array(highs)
        low = np.array(lows)

        # 计算指标
        try:
            # MACD
            macd = Indicators.macd(
                close,
                MACD_CONFIG["fast_period"],
                MACD_CONFIG["slow_period"],
                MACD_CONFIG["signal_period"]
            )
            histogram = macd.histogram

            # KDJ
            kdj = Indicators.kdj(high, low, close)
            k = kdj.k
            d = kdj.d
            j = kdj.j

            # RSI
            rsi = Indicators.rsi(close, 14)

            # 信号判断
            signal_type = "NONE"
            strength = 0.0
            reasons = []

            # MACD 金叉/死叉
            if len(histogram) >= 2:
                if histogram[-1] > 0 and histogram[-2] <= 0:
                    reasons.append("MACD金叉")
                    strength += 0.3
                    signal_type = "BUY"
                elif histogram[-1] < 0 and histogram[-2] >= 0:
                    reasons.append("MACD死叉")
                    strength += 0.3
                    signal_type = "SELL"

            # KDJ 金叉/死叉
            if len(k) >= 2 and len(d) >= 2:
                if k[-1] > d[-1] and k[-2] <= d[-2] and k[-1] < 80:
                    reasons.append("KDJ金叉")
                    strength += 0.3
                    if signal_type == "NONE":
                        signal_type = "BUY"
                elif k[-1] < d[-1] and k[-2] >= d[-2] and k[-1] > 20:
                    reasons.append("KDJ死叉")
                    strength += 0.3
                    if signal_type == "NONE":
                        signal_type = "SELL"

            # RSI 超买超卖
            if len(rsi) >= 1:
                if rsi[-1] < 30:
                    reasons.append("RSI超卖")
                    strength += 0.2
                    if signal_type == "NONE":
                        signal_type = "BUY"
                elif rsi[-1] > 70:
                    reasons.append("RSI超买")
                    strength += 0.2
                    if signal_type == "NONE":
                        signal_type = "SELL"

            # 趋势确认
            vol_data = self.latest_volatility.get(symbol)
            if vol_data:
                if vol_data.trend == "up" and signal_type == "BUY":
                    reasons.append("趋势确认")
                    strength += 0.2
                elif vol_data.trend == "down" and signal_type == "SELL":
                    reasons.append("趋势确认")
                    strength += 0.2

            signal = TradeSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=min(strength, 1.0),
                reason=" + ".join(reasons) if reasons else "无明确信号",
                price=prices[-1],
                timestamp=time.time()
            )

            self.latest_signals[symbol] = signal

            # 强信号报警
            if signal.strength >= 0.5 and signal.signal_type != "NONE":
                self._check_signal_alert(signal)

            return signal

        except Exception as e:
            self.logger.debug(f"计算 {symbol} 信号失败: {e}")
            signal = TradeSignal(
                symbol=symbol, signal_type="NONE", strength=0,
                reason=f"计算错误: {e}", price=0, timestamp=time.time()
            )
            self.latest_signals[symbol] = signal
            return signal

    def _check_signal_alert(self, signal: TradeSignal):
        """检查信号报警"""
        last_time = self._last_alert_time.get(f"sig_{signal.symbol}", 0)
        if time.time() - last_time < self._alert_cooldown:
            return

        alert = AlertEvent(
            alert_type="signal",
            symbol=signal.symbol,
            message=f"{signal.symbol} {signal.signal_type} 信号 (强度 {signal.strength:.0%}): {signal.reason}",
            value=signal.strength,
            timestamp=time.time()
        )

        self._last_alert_time[f"sig_{signal.symbol}"] = time.time()
        self._trigger_alert(alert)

    # ==================== 4. 历史记录 ====================

    def save_history_record(self):
        """保存历史记录"""
        record = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "ticks": {s: asdict(t) for s, t in self.latest_ticks.items()},
            "volatility": {s: asdict(v) for s, v in self.latest_volatility.items()},
            "signals": {s: asdict(sig) for s, sig in self.latest_signals.items()},
            "current_symbol": self.current_symbol,
        }

        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_alert(self, alert: AlertEvent):
        """保存报警记录"""
        with open(self.alerts_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(alert), ensure_ascii=False) + "\n")

    def _save_session_summary(self):
        """保存会话摘要"""
        summary = {
            "session_end": datetime.now().isoformat(),
            "total_alerts": len(self.alerts),
            "symbols_monitored": self.symbols,
            "final_volatility": {s: asdict(v) for s, v in self.latest_volatility.items()},
        }

        summary_file = self.data_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # ==================== 报警处理 ====================

    def _trigger_alert(self, alert: AlertEvent):
        """触发报警"""
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)

        # 保存报警
        self.save_alert(alert)

        # 声音提醒 (Windows)
        try:
            if alert.alert_type == "signal":
                winsound.Beep(1000, 500)  # 高音
            else:
                winsound.Beep(800, 300)   # 中音
        except:
            pass

        # 回调
        if self.on_alert:
            self.on_alert(alert)

        # 打印
        print(f"\n{'='*60}")
        print(f"  !!! 报警 [{alert.alert_type.upper()}] !!!")
        print(f"  {alert.message}")
        print(f"  时间: {datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

    # ==================== 主循环 ====================

    async def run(self, duration: int = None):
        """运行监控"""
        if not await self.start():
            return

        start_time = time.time()
        check_count = 0
        save_interval = 12  # 每12次检查保存一次历史 (约1分钟)

        print("\n" + "="*70)
        print("  智能多品种监控已启动")
        print("="*70)
        print(f"  监控品种: {', '.join(self.symbols)}")
        print(f"  波动率阈值: {self.volatility_threshold}%")
        print(f"  自动切换: {'开启' if self.auto_switch else '关闭'}")
        print(f"  数据目录: {self.data_dir}")
        print("  注意: 本脚本仅监控/报警，不执行下单；自动下单请用 auto_trader.py")
        print("="*70 + "\n")

        try:
            while self._is_running:
                # 获取价格
                await self.fetch_all_prices()
                check_count += 1

                # 计算波动率、信号和预测
                for symbol in self.symbols:
                    if symbol in self.latest_ticks:
                        vol_data = self.calc_volatility(symbol)
                        self.check_volatility_alert(vol_data)
                        self.generate_signal(symbol)

                        # 价格预测
                        prices = self.price_history.get(symbol, [])
                        if len(prices) >= 10:
                            pred = self.predictor.predict(prices)
                            self.latest_predictions[symbol] = {
                                "direction": pred.direction.value,
                                "confidence": pred.confidence,
                                "predicted_change": pred.predicted_change,
                                "reason": pred.reason,
                                "methods_agree": pred.methods_agree,
                            }
                            # 更新跨市场预测器
                            self.cross_predictor.update_price(symbol, prices[-1])

                # 自动切换
                if self.auto_switch:
                    await self.auto_switch_to_best()

                # 保存历史
                if check_count % save_interval == 0:
                    self.save_history_record()

                # 打印状态
                self._print_status()

                # 检查时长
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n已运行 {duration} 秒，停止监控")
                    break

                await asyncio.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\n收到停止信号...")
        finally:
            await self.stop()

    def _print_status(self):
        """打印状态"""
        # 清屏
        print("\033[2J\033[H", end="")

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("="*80)
        print(f"  智能多品种监控 (含预测) - {now}")
        print("="*80)

        # 波动率排名
        ranking = self.get_volatility_ranking()

        if not ranking:
            print("\n  等待数据...")
        else:
            print(f"\n  {'品种':<10} {'价格':<12} {'波动率%':<8} {'涨跌%':<8} {'预测':<8} {'置信度':<8} {'信号':<12}")
            print("-"*80)

            for vol in ranking:
                tick = self.latest_ticks.get(vol.symbol)
                signal = self.latest_signals.get(vol.symbol)

                price = f"{tick.mid:.4f}" if tick else "N/A"
                volatility = f"{vol.volatility:.4f}"
                change = f"{vol.price_change:+.3f}"

                # 获取预测
                pred = self.latest_predictions.get(vol.symbol, {})
                pred_dir = pred.get("direction", "→")
                pred_conf = pred.get("confidence", 0)
                pred_str = {"UP": "↑涨", "DOWN": "↓跌", "NEUTRAL": "→平"}.get(pred_dir, "→")
                conf_str = f"{pred_conf:.0%}" if pred_conf > 0 else "-"

                sig_str = "-"
                if signal and signal.signal_type != "NONE":
                    sig_str = f"{signal.signal_type}({signal.strength:.0%})"

                # 高波动标记
                flag = " <<<" if vol.volatility >= self.volatility_threshold else ""

                print(f"  {vol.symbol:<10} {price:<12} {volatility:<8} {change:<8} {pred_str:<8} {conf_str:<8} {sig_str:<12}{flag}")

            # 未找到的品种
            found = [v.symbol for v in ranking]
            missing = [s for s in self.symbols if s not in found]
            if missing:
                print(f"\n  [!] 未找到: {', '.join(missing)}")

        print("-"*80)

        # 当前品种和预测
        curr_pred = self.latest_predictions.get(self.current_symbol, {})
        pred_reason = curr_pred.get("reason", "")
        print(f"  当前品种: {self.current_symbol or 'N/A'}")
        if pred_reason:
            print(f"  预测依据: {pred_reason}")

        # 最近报警
        if self.alerts:
            recent = self.alerts[-1]
            print(f"  最近报警: [{recent.alert_type}] {recent.message[:50]}...")

        print("="*80)
        print("  按 Ctrl+C 停止 | 预测仅供参考 | 数据保存到 monitor_data/")
        print("="*80)


# ==================== 启动入口 ====================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="智能多品种监控")
    parser.add_argument("--symbols", type=str, default="USOIL,XAUUSD,ETHUSD", help="监控品种")
    parser.add_argument("--threshold", type=float, default=0.1, help="波动率报警阈值 (%)")
    parser.add_argument("--auto-switch", action="store_true", default=True, help="自动切换品种")
    parser.add_argument("--no-auto-switch", dest="auto_switch", action="store_false", help="禁用自动切换")
    parser.add_argument("--interval", type=float, default=5.0, help="检查间隔 (秒)")
    parser.add_argument("--duration", type=int, default=0, help="运行时长 (秒), 0=无限")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    duration = args.duration if args.duration > 0 else None

    monitor = SmartMonitor(
        symbols=symbols,
        volatility_threshold=args.threshold,
        auto_switch=args.auto_switch,
        check_interval=args.interval,
    )

    await monitor.run(duration)


if __name__ == "__main__":
    asyncio.run(main())
