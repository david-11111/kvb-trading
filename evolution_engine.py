"""
自我进化引擎模块

核心功能：
1. 开仓5分钟后检测持仓盈亏状态
2. 盈利时总结成功原因，奖励加分
3. 亏损时反思失败原因，惩罚减分
4. 形成学习闭环，持续自我进化
5. 记录日志以备查验

进化算法：
- 贝叶斯参数更新：将策略参数建模为Beta分布
- 汤普森采样：在探索与利用之间平衡
- 上下文强盗：根据市场状态选择最优参数
"""

import json
import time
import threading
import logging
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from collections import defaultdict

from config import AUTO_TRADE_CONFIG
from adaptive_strategy import AdaptiveStrategy, create_adaptive_strategy


class ReflectionType(Enum):
    """反思类型"""
    PROFIT = "profit"      # 盈利反思
    LOSS = "loss"          # 亏损反思
    BREAKEVEN = "breakeven"  # 持平反思


class MarketCondition(Enum):
    """市场状态"""
    TRENDING_UP = "trending_up"        # 上升趋势
    TRENDING_DOWN = "trending_down"    # 下降趋势
    RANGING = "ranging"                # 震荡
    VOLATILE = "volatile"              # 高波动
    CALM = "calm"                      # 低波动
    UNKNOWN = "unknown"                # 未知


@dataclass
class EntryContext:
    """开仓时的上下文信息"""
    symbol: str                        # 交易品种
    direction: str                     # 方向: long/short
    entry_price: float                 # 开仓价格
    entry_time: float                  # 开仓时间戳
    signal_type: str                   # 信号类型
    signal_strength: float             # 信号强度
    market_condition: MarketCondition  # 市场状态
    indicators: Dict[str, float]       # 开仓时的指标值
    momentum: float                    # 动量
    volatility: float                  # 波动率
    spread: float                      # 点差
    recent_prices: List[float]         # 最近价格
    reason: str                        # 开仓原因


@dataclass
class ReflectionRecord:
    """反思记录"""
    id: str                            # 唯一ID
    timestamp: float                   # 反思时间
    symbol: str                        # 交易品种
    reflection_type: ReflectionType    # 反思类型
    entry_context: EntryContext        # 开仓上下文
    check_price: float                 # 检查时价格
    pnl: float                         # 盈亏金额
    pnl_percent: float                 # 盈亏百分比
    hold_duration: float               # 持仓时长(秒)
    analysis: str                      # 分析总结
    lessons: List[str]                 # 学到的教训
    score_change: int                  # 分数变化
    suggestions: List[str]             # 改进建议

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['reflection_type'] = self.reflection_type.value
        data['entry_context']['market_condition'] = self.entry_context.market_condition.value
        return data


@dataclass
class LessonLearned:
    """学到的经验教训"""
    id: str
    created_at: float
    updated_at: float
    category: str                      # 分类: entry/exit/risk/timing
    condition: str                     # 适用条件
    lesson: str                        # 教训内容
    success_count: int = 0             # 成功次数
    failure_count: int = 0             # 失败次数
    effectiveness: float = 0.5         # 有效性评分 (0-1)
    is_active: bool = True             # 是否启用


@dataclass
class EvolutionScore:
    """进化评分"""
    total_score: int = 1000            # 总分 (初始1000)
    profit_reflections: int = 0        # 盈利反思次数
    loss_reflections: int = 0          # 亏损反思次数
    win_streak: int = 0                # 连胜次数
    lose_streak: int = 0               # 连败次数
    max_win_streak: int = 0            # 最大连胜
    max_lose_streak: int = 0           # 最大连败
    total_profit_amount: float = 0     # 总盈利金额
    total_loss_amount: float = 0       # 总亏损金额
    lessons_learned: int = 0           # 学到的教训数


class EvolutionEngine:
    """
    自我进化引擎

    核心职责：
    1. 监控持仓状态
    2. 分析盈亏原因
    3. 生成反思报告
    4. 更新评分系统
    5. 积累经验教训
    """

    # 评分规则
    SCORE_RULES = {
        "profit_base": 10,              # 盈利基础加分
        "profit_per_percent": 5,        # 每盈利1%额外加分
        "loss_base": -15,               # 亏损基础减分
        "loss_per_percent": -8,         # 每亏损1%额外减分
        "breakeven": 2,                 # 持平小奖励
        "streak_bonus": 5,              # 连胜奖励
        "streak_penalty": -3,           # 连败惩罚
        "lesson_applied": 8,            # 应用教训成功加分
        "lesson_ignored": -10,          # 忽视教训失败减分
    }

    # 检查时间点 (秒)
    CHECK_INTERVALS = [300, 600, 900]   # 5分钟, 10分钟, 15分钟

    def __init__(self, data_dir: str = None, auto_trader = None):
        """
        初始化进化引擎

        Args:
            data_dir: 数据存储目录
            auto_trader: AutoTrader实例引用
        """
        self.data_dir = Path(data_dir or "evolution_data")
        self.data_dir.mkdir(exist_ok=True)

        self.auto_trader = auto_trader
        self.logger = logging.getLogger("EvolutionEngine")

        # 评分系统
        self.score = EvolutionScore()

        # 经验库
        self.lessons: Dict[str, LessonLearned] = {}

        # 反思历史
        self.reflections: List[ReflectionRecord] = []

        # 待检查的持仓
        self._pending_checks: Dict[str, Dict] = {}  # symbol -> check info
        self._check_timers: Dict[str, threading.Timer] = {}

        # 模式统计
        self._pattern_stats: Dict[str, Dict] = defaultdict(lambda: {
            "profit_count": 0,
            "loss_count": 0,
            "total_pnl": 0,
            "conditions": []
        })

        # 自适应策略 - 真正的进化算法
        self.adaptive_strategy = create_adaptive_strategy(data_dir=str(self.data_dir))

        # 加载历史数据
        self._load_data()

        self.logger.info(f"进化引擎初始化完成，当前评分: {self.score.total_score}")

    def _load_data(self):
        """加载历史数据"""
        # 加载评分
        score_file = self.data_dir / "evolution_score.json"
        if score_file.exists():
            try:
                with open(score_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.score = EvolutionScore(**data)
                self.logger.info(f"加载评分数据: {self.score.total_score}")
            except Exception as e:
                self.logger.error(f"加载评分数据失败: {e}")

        # 加载经验库
        lessons_file = self.data_dir / "lessons_learned.json"
        if lessons_file.exists():
            try:
                with open(lessons_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        lesson = LessonLearned(**item)
                        self.lessons[lesson.id] = lesson
                self.logger.info(f"加载经验库: {len(self.lessons)} 条教训")
            except Exception as e:
                self.logger.error(f"加载经验库失败: {e}")

        # 加载模式统计
        patterns_file = self.data_dir / "pattern_stats.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    self._pattern_stats = defaultdict(lambda: {
                        "profit_count": 0, "loss_count": 0,
                        "total_pnl": 0, "conditions": []
                    }, json.load(f))
                self.logger.info(f"加载模式统计: {len(self._pattern_stats)} 个模式")
            except Exception as e:
                self.logger.error(f"加载模式统计失败: {e}")

    def _save_data(self):
        """保存数据"""
        try:
            # 保存评分
            with open(self.data_dir / "evolution_score.json", 'w', encoding='utf-8') as f:
                json.dump(asdict(self.score), f, indent=2, ensure_ascii=False)

            # 保存经验库
            with open(self.data_dir / "lessons_learned.json", 'w', encoding='utf-8') as f:
                lessons_data = [asdict(l) for l in self.lessons.values()]
                json.dump(lessons_data, f, indent=2, ensure_ascii=False)

            # 保存模式统计
            with open(self.data_dir / "pattern_stats.json", 'w', encoding='utf-8') as f:
                json.dump(dict(self._pattern_stats), f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")

    def register_position(self, symbol: str, direction: str, entry_price: float,
                         entry_time: float, signal_info: dict, market_data: dict):
        """
        注册新开仓，启动5分钟检查定时器

        Args:
            symbol: 交易品种
            direction: 方向
            entry_price: 开仓价格
            entry_time: 开仓时间
            signal_info: 信号信息
            market_data: 市场数据
        """
        # ========== 核心进化：记录使用的自适应参数 ==========
        # 这会触发汤普森采样，选择当前上下文最优参数
        try:
            adapted_params = self.adaptive_strategy.get_adapted_parameters(symbol, market_data)
            self.logger.debug(f"[{symbol}] 使用自适应参数: {adapted_params}")
        except Exception as e:
            self.logger.warning(f"获取自适应参数失败: {e}")
            adapted_params = {}

        # 构建开仓上下文
        context = EntryContext(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            signal_type=signal_info.get("type", "unknown"),
            signal_strength=signal_info.get("strength", 0),
            market_condition=self._analyze_market_condition(market_data),
            indicators=market_data.get("indicators", {}),
            momentum=market_data.get("momentum", 0),
            volatility=market_data.get("volatility", 0),
            spread=market_data.get("spread", 0),
            recent_prices=market_data.get("recent_prices", [])[-20:],
            reason=signal_info.get("reason", "")
        )

        # 保存待检查信息
        self._pending_checks[symbol] = {
            "context": context,
            "checked_intervals": [],
            "reflections": [],
            "adapted_params": adapted_params  # 保存使用的参数以便后续分析
        }

        # 设置5分钟检查定时器
        self._schedule_check(symbol, 300)  # 5分钟 = 300秒

        self.logger.info(f"注册持仓反思检查: {symbol} {direction} @ {entry_price}")

        # 记录开仓日志
        self._log_event("POSITION_REGISTERED", {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "signal_type": context.signal_type,
            "market_condition": context.market_condition.value
        })

    def _schedule_check(self, symbol: str, delay: float):
        """调度检查任务"""
        # 取消旧定时器
        if symbol in self._check_timers:
            self._check_timers[symbol].cancel()

        # 创建新定时器
        timer = threading.Timer(delay, self._perform_check, args=[symbol, delay])
        timer.daemon = True
        timer.start()
        self._check_timers[symbol] = timer

        self.logger.debug(f"调度 {symbol} 检查: {delay}秒后执行")

    def _perform_check(self, symbol: str, interval: float):
        """执行持仓检查"""
        if symbol not in self._pending_checks:
            return

        check_info = self._pending_checks[symbol]
        context = check_info["context"]

        # 获取当前价格
        current_price = self._get_current_price(symbol)
        if current_price is None:
            self.logger.warning(f"无法获取 {symbol} 当前价格，跳过检查")
            return

        # 计算盈亏
        pnl, pnl_percent = self._calculate_pnl(
            context.direction, context.entry_price, current_price
        )

        # 生成反思
        reflection = self._generate_reflection(
            context, current_price, pnl, pnl_percent, interval
        )

        # 更新评分
        score_change = self._update_score(reflection)
        reflection.score_change = score_change

        # 保存反思
        self.reflections.append(reflection)
        check_info["reflections"].append(reflection)
        check_info["checked_intervals"].append(interval)

        # 记录日志
        self._log_reflection(reflection)

        # 更新经验库
        self._update_lessons(reflection)

        # ========== 核心进化：将结果反馈给自适应策略 ==========
        # 这是真正的机器学习反馈：贝叶斯更新 + 汤普森采样
        # is_final=False 表示这是中间检查，不移除交易记录
        try:
            self.adaptive_strategy.report_trade_result(
                symbol=symbol,
                pnl=pnl,
                pnl_percent=pnl_percent,
                hold_duration=interval,
                is_final=False  # 中间检查，保留交易记录以便后续继续学习
            )
            self.logger.debug(f"[{symbol}] 中间检查进化反馈已提交")
        except Exception as e:
            self.logger.warning(f"自适应策略反馈失败: {e}")

        # 保存数据
        self._save_data()

        self.logger.info(
            f"[{symbol}] {int(interval/60)}分钟检查: "
            f"{'盈利' if pnl >= 0 else '亏损'} {pnl:.2f} ({pnl_percent*100:.2f}%), "
            f"评分变化: {score_change:+d}, 当前总分: {self.score.total_score}"
        )

        # 如果仍有持仓，调度下一次检查
        if self._has_position(symbol):
            next_intervals = [i for i in self.CHECK_INTERVALS if i > interval]
            if next_intervals:
                next_delay = next_intervals[0] - interval
                self._schedule_check(symbol, next_delay)

    def _generate_reflection(self, context: EntryContext, check_price: float,
                            pnl: float, pnl_percent: float, interval: float) -> ReflectionRecord:
        """生成反思记录"""
        # 确定反思类型
        if pnl > 0:
            reflection_type = ReflectionType.PROFIT
        elif pnl < 0:
            reflection_type = ReflectionType.LOSS
        else:
            reflection_type = ReflectionType.BREAKEVEN

        # 生成分析
        analysis = self._analyze_trade(context, check_price, pnl, pnl_percent, reflection_type)

        # 提取教训
        lessons = self._extract_lessons(context, reflection_type, pnl_percent)

        # 生成建议
        suggestions = self._generate_suggestions(context, reflection_type, pnl_percent)

        # 创建反思记录
        reflection = ReflectionRecord(
            id=f"REF_{context.symbol}_{int(time.time()*1000)}",
            timestamp=time.time(),
            symbol=context.symbol,
            reflection_type=reflection_type,
            entry_context=context,
            check_price=check_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            hold_duration=interval,
            analysis=analysis,
            lessons=lessons,
            score_change=0,  # 稍后更新
            suggestions=suggestions
        )

        return reflection

    def _analyze_trade(self, context: EntryContext, check_price: float,
                      pnl: float, pnl_percent: float,
                      reflection_type: ReflectionType) -> str:
        """分析交易"""
        lines = []

        # 基本信息
        lines.append(f"【交易分析】{context.symbol} {context.direction.upper()}")
        lines.append(f"开仓价: {context.entry_price} -> 当前价: {check_price}")
        lines.append(f"盈亏: {pnl:+.2f} ({pnl_percent*100:+.2f}%)")
        lines.append(f"市场状态: {context.market_condition.value}")
        lines.append(f"信号类型: {context.signal_type}, 强度: {context.signal_strength:.2f}")

        if reflection_type == ReflectionType.PROFIT:
            lines.append("\n【成功原因分析】")

            # 分析成功因素
            if context.market_condition in [MarketCondition.TRENDING_UP, MarketCondition.TRENDING_DOWN]:
                if (context.direction == "long" and context.market_condition == MarketCondition.TRENDING_UP) or \
                   (context.direction == "short" and context.market_condition == MarketCondition.TRENDING_DOWN):
                    lines.append("✓ 顺势交易：方向与市场趋势一致")

            if context.signal_strength > 0.7:
                lines.append("✓ 强信号入场：信号强度较高")

            if context.momentum > 0 and context.direction == "long":
                lines.append("✓ 动量支持：正向动量支持多头")
            elif context.momentum < 0 and context.direction == "short":
                lines.append("✓ 动量支持：负向动量支持空头")

            if context.spread < 0.05:
                lines.append("✓ 低点差入场：交易成本低")

        elif reflection_type == ReflectionType.LOSS:
            lines.append("\n【失败原因分析】")

            # 分析失败因素
            if context.market_condition == MarketCondition.RANGING:
                lines.append("✗ 震荡市操作：震荡市中趋势策略效果差")

            if context.market_condition == MarketCondition.VOLATILE:
                lines.append("✗ 高波动市场：波动过大导致止损")

            if context.signal_strength < 0.5:
                lines.append("✗ 弱信号入场：信号强度不足")

            if (context.direction == "long" and context.momentum < 0) or \
               (context.direction == "short" and context.momentum > 0):
                lines.append("✗ 逆动量交易：方向与动量相反")

            if context.spread > 0.1:
                lines.append("✗ 高点差入场：交易成本过高")

        return "\n".join(lines)

    def _extract_lessons(self, context: EntryContext,
                        reflection_type: ReflectionType,
                        pnl_percent: float) -> List[str]:
        """提取教训"""
        lessons = []

        if reflection_type == ReflectionType.PROFIT:
            # 盈利教训
            if context.signal_strength > 0.7:
                lessons.append(f"在{context.market_condition.value}市场中，高强度信号({context.signal_strength:.2f})入场效果好")

            if context.momentum * (1 if context.direction == "long" else -1) > 0:
                lessons.append("顺动量方向交易提高成功率")

            if pnl_percent > 0.02:  # 盈利超过2%
                lessons.append(f"{context.signal_type}信号在当前市况下表现优秀")

        elif reflection_type == ReflectionType.LOSS:
            # 亏损教训
            if context.signal_strength < 0.5:
                lessons.append("避免在弱信号时入场，等待更强确认")

            if context.market_condition == MarketCondition.RANGING:
                lessons.append("震荡市慎用趋势策略，考虑区间策略")

            if context.market_condition == MarketCondition.VOLATILE:
                lessons.append("高波动时减小仓位或暂停交易")

            if context.spread > 0.1:
                lessons.append("避免在高点差时入场")

            if abs(pnl_percent) > 0.03:  # 亏损超过3%
                lessons.append("需要更严格的止损或更好的入场时机")

        return lessons

    def _generate_suggestions(self, context: EntryContext,
                             reflection_type: ReflectionType,
                             pnl_percent: float) -> List[str]:
        """生成改进建议"""
        suggestions = []

        if reflection_type == ReflectionType.LOSS:
            suggestions.append("复盘入场时机，是否可以等待更好的价位")
            suggestions.append("检查止损设置是否合理")

            if context.market_condition == MarketCondition.VOLATILE:
                suggestions.append("高波动时考虑减半仓位")

            if context.signal_strength < 0.6:
                suggestions.append("提高信号强度阈值至0.6以上")

        elif reflection_type == ReflectionType.PROFIT:
            suggestions.append("总结成功模式，纳入策略库")

            if pnl_percent < 0.01:
                suggestions.append("考虑延长持仓时间以获取更多利润")

        return suggestions

    def _update_score(self, reflection: ReflectionRecord) -> int:
        """更新评分"""
        score_change = 0
        rules = self.SCORE_RULES

        if reflection.reflection_type == ReflectionType.PROFIT:
            # 盈利加分
            score_change = rules["profit_base"]
            score_change += int(reflection.pnl_percent * 100 * rules["profit_per_percent"])

            # 更新连胜
            self.score.win_streak += 1
            self.score.lose_streak = 0
            if self.score.win_streak > self.score.max_win_streak:
                self.score.max_win_streak = self.score.win_streak

            # 连胜奖励
            if self.score.win_streak >= 3:
                score_change += rules["streak_bonus"] * (self.score.win_streak - 2)

            self.score.profit_reflections += 1
            self.score.total_profit_amount += reflection.pnl

        elif reflection.reflection_type == ReflectionType.LOSS:
            # 亏损减分
            score_change = rules["loss_base"]
            score_change += int(abs(reflection.pnl_percent) * 100 * rules["loss_per_percent"])

            # 更新连败
            self.score.lose_streak += 1
            self.score.win_streak = 0
            if self.score.lose_streak > self.score.max_lose_streak:
                self.score.max_lose_streak = self.score.lose_streak

            # 连败惩罚
            if self.score.lose_streak >= 3:
                score_change += rules["streak_penalty"] * (self.score.lose_streak - 2)

            self.score.loss_reflections += 1
            self.score.total_loss_amount += abs(reflection.pnl)

        else:
            # 持平
            score_change = rules["breakeven"]

        # 更新总分
        self.score.total_score += score_change

        # 防止负分
        if self.score.total_score < 0:
            self.score.total_score = 0

        return score_change

    def _update_lessons(self, reflection: ReflectionRecord):
        """更新经验库"""
        for lesson_text in reflection.lessons:
            # 生成教训ID
            lesson_id = f"LESSON_{hash(lesson_text) % 100000}"

            if lesson_id in self.lessons:
                # 更新现有教训
                lesson = self.lessons[lesson_id]
                lesson.updated_at = time.time()
                if reflection.reflection_type == ReflectionType.PROFIT:
                    lesson.success_count += 1
                else:
                    lesson.failure_count += 1
                # 重新计算有效性
                total = lesson.success_count + lesson.failure_count
                lesson.effectiveness = lesson.success_count / total if total > 0 else 0.5
            else:
                # 创建新教训
                category = self._categorize_lesson(lesson_text)
                lesson = LessonLearned(
                    id=lesson_id,
                    created_at=time.time(),
                    updated_at=time.time(),
                    category=category,
                    condition=reflection.entry_context.market_condition.value,
                    lesson=lesson_text,
                    success_count=1 if reflection.reflection_type == ReflectionType.PROFIT else 0,
                    failure_count=1 if reflection.reflection_type == ReflectionType.LOSS else 0,
                    effectiveness=1.0 if reflection.reflection_type == ReflectionType.PROFIT else 0.0,
                    is_active=True
                )
                self.lessons[lesson_id] = lesson
                self.score.lessons_learned += 1

        # 更新模式统计
        pattern_key = f"{reflection.entry_context.signal_type}_{reflection.entry_context.market_condition.value}"
        stats = self._pattern_stats[pattern_key]
        if reflection.reflection_type == ReflectionType.PROFIT:
            stats["profit_count"] += 1
        else:
            stats["loss_count"] += 1
        stats["total_pnl"] += reflection.pnl

    def _categorize_lesson(self, lesson_text: str) -> str:
        """对教训进行分类"""
        if any(word in lesson_text for word in ["入场", "信号", "时机"]):
            return "entry"
        elif any(word in lesson_text for word in ["止损", "止盈", "平仓"]):
            return "exit"
        elif any(word in lesson_text for word in ["仓位", "风险", "波动"]):
            return "risk"
        elif any(word in lesson_text for word in ["时间", "周期", "持仓"]):
            return "timing"
        return "general"

    def _log_reflection(self, reflection: ReflectionRecord):
        """记录反思日志"""
        log_file = self.data_dir / f"reflections_{datetime.now().strftime('%Y%m%d')}.jsonl"

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(reflection.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"写入反思日志失败: {e}")

    def _log_event(self, event_type: str, data: dict):
        """记录事件日志"""
        log_file = self.data_dir / f"evolution_events_{datetime.now().strftime('%Y%m%d')}.jsonl"

        event = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"写入事件日志失败: {e}")

    def _analyze_market_condition(self, market_data: dict) -> MarketCondition:
        """分析市场状态"""
        volatility = market_data.get("volatility", 0)
        trend_strength = market_data.get("trend_strength", 0)
        momentum = market_data.get("momentum", 0)

        # 高波动
        if volatility > 0.02:
            return MarketCondition.VOLATILE

        # 低波动
        if volatility < 0.005:
            return MarketCondition.CALM

        # 强趋势
        if abs(trend_strength) > 0.7:
            if momentum > 0:
                return MarketCondition.TRENDING_UP
            else:
                return MarketCondition.TRENDING_DOWN

        # 震荡
        if abs(trend_strength) < 0.3:
            return MarketCondition.RANGING

        return MarketCondition.UNKNOWN

    def _calculate_pnl(self, direction: str, entry_price: float,
                      current_price: float) -> tuple:
        """计算盈亏"""
        if direction == "long":
            pnl = current_price - entry_price
        else:
            pnl = entry_price - current_price

        pnl_percent = pnl / entry_price if entry_price else 0

        return pnl, pnl_percent

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        if self.auto_trader and hasattr(self.auto_trader, 'latest_prices'):
            price_info = self.auto_trader.latest_prices.get(symbol, {})
            return price_info.get("mid") or price_info.get("bid")
        return None

    def _has_position(self, symbol: str) -> bool:
        """检查是否仍有持仓"""
        if self.auto_trader and hasattr(self.auto_trader, 'positions'):
            return symbol in self.auto_trader.positions
        return False

    def unregister_position(self, symbol: str, final_pnl: float = None,
                            final_pnl_percent: float = None):
        """
        注销持仓（平仓时调用）

        Args:
            symbol: 交易品种
            final_pnl: 最终盈亏金额
            final_pnl_percent: 最终盈亏百分比
        """
        # 获取持仓时长
        hold_duration = 0
        if symbol in self._pending_checks:
            context = self._pending_checks[symbol].get("context")
            if context:
                hold_duration = time.time() - context.entry_time
            del self._pending_checks[symbol]

        # 取消定时器
        if symbol in self._check_timers:
            self._check_timers[symbol].cancel()
            del self._check_timers[symbol]

        # ========== 核心进化：发送最终平仓反馈 ==========
        # 这是最重要的学习信号，权重最高
        if final_pnl is not None and final_pnl_percent is not None:
            try:
                self.adaptive_strategy.report_trade_result(
                    symbol=symbol,
                    pnl=final_pnl,
                    pnl_percent=final_pnl_percent,
                    hold_duration=hold_duration,
                    is_final=True  # 最终平仓，移除交易记录
                )
                self.logger.info(
                    f"[{symbol}] 最终平仓进化反馈: "
                    f"{'盈利' if final_pnl > 0 else '亏损'} {final_pnl:.2f} ({final_pnl_percent*100:.2f}%)"
                )
            except Exception as e:
                self.logger.warning(f"最终平仓进化反馈失败: {e}")

        self.logger.info(f"注销持仓反思检查: {symbol}")

    def get_applicable_lessons(self, market_condition: str,
                              signal_type: str) -> List[LessonLearned]:
        """获取适用的经验教训"""
        applicable = []
        for lesson in self.lessons.values():
            if not lesson.is_active:
                continue
            if lesson.effectiveness < 0.3:
                continue
            if lesson.condition == market_condition or lesson.condition == "general":
                applicable.append(lesson)

        # 按有效性排序
        applicable.sort(key=lambda x: x.effectiveness, reverse=True)
        return applicable[:5]  # 最多返回5条

    def check_lessons_warning(self, market_data: dict) -> tuple:
        """
        检查经验教训是否发出警告

        根据已学到的教训，判断当前市场条件是否应该避免交易

        Args:
            market_data: 市场数据

        Returns:
            (should_avoid: bool, warning_reason: str)
        """
        # 提取市场条件
        spread = market_data.get("spread", 0)
        volatility = market_data.get("volatility", 0)
        signal_type = market_data.get("signal_type", "")

        # 获取当前上下文
        context = self.adaptive_strategy.classify_context(market_data)

        # 获取适用教训
        lessons = self.get_applicable_lessons(context, signal_type)

        warnings = []
        for lesson in lessons:
            text = lesson.lesson_text.lower()

            # 检查高点差警告
            if "点差" in text and "避免" in text:
                # 如果点差超过0.1%，应用此教训
                price = market_data.get("price", 1)
                if price > 0:
                    spread_pct = spread / price
                    if spread_pct > 0.001:  # 0.1%
                        warnings.append(f"教训警告: 高点差({spread_pct:.2%})")

            # 检查高波动警告
            if "波动" in text and ("减小" in text or "暂停" in text):
                if volatility > 0.02:  # 2%
                    warnings.append(f"教训警告: 高波动({volatility:.2%})")

            # 检查震荡市警告
            if "震荡" in text and "慎用" in text:
                if "ranging" in context:
                    warnings.append("教训警告: 震荡市慎用趋势策略")

        if warnings:
            return (True, "; ".join(warnings[:2]))  # 最多返回2条警告

        return (False, "")

    def get_pattern_stats(self, signal_type: str = None) -> Dict:
        """获取模式统计"""
        if signal_type:
            return {k: v for k, v in self._pattern_stats.items() if signal_type in k}
        return dict(self._pattern_stats)

    def get_score_summary(self) -> Dict:
        """获取评分摘要"""
        win_rate = 0
        if self.score.profit_reflections + self.score.loss_reflections > 0:
            win_rate = self.score.profit_reflections / (
                self.score.profit_reflections + self.score.loss_reflections
            )

        return {
            "total_score": self.score.total_score,
            "win_rate": win_rate,
            "profit_reflections": self.score.profit_reflections,
            "loss_reflections": self.score.loss_reflections,
            "win_streak": self.score.win_streak,
            "lose_streak": self.score.lose_streak,
            "max_win_streak": self.score.max_win_streak,
            "max_lose_streak": self.score.max_lose_streak,
            "total_profit": self.score.total_profit_amount,
            "total_loss": self.score.total_loss_amount,
            "net_pnl": self.score.total_profit_amount - self.score.total_loss_amount,
            "lessons_learned": self.score.lessons_learned
        }

    def generate_daily_report(self) -> str:
        """生成每日报告"""
        summary = self.get_score_summary()

        lines = [
            "=" * 50,
            "【每日进化报告】",
            f"日期: {datetime.now().strftime('%Y-%m-%d')}",
            "=" * 50,
            "",
            f"📊 当前评分: {summary['total_score']}",
            f"📈 胜率: {summary['win_rate']*100:.1f}%",
            f"🎯 盈利次数: {summary['profit_reflections']}",
            f"❌ 亏损次数: {summary['loss_reflections']}",
            "",
            f"🔥 当前连胜: {summary['win_streak']}",
            f"❄️ 当前连败: {summary['lose_streak']}",
            f"🏆 最大连胜: {summary['max_win_streak']}",
            f"💔 最大连败: {summary['max_lose_streak']}",
            "",
            f"💰 总盈利: {summary['total_profit']:.2f}",
            f"💸 总亏损: {summary['total_loss']:.2f}",
            f"📊 净盈亏: {summary['net_pnl']:.2f}",
            "",
            f"📚 累计学到教训: {summary['lessons_learned']} 条",
            "",
            "【最有效的教训】"
        ]

        # 添加最有效的教训
        effective_lessons = sorted(
            self.lessons.values(),
            key=lambda x: x.effectiveness,
            reverse=True
        )[:5]

        for i, lesson in enumerate(effective_lessons, 1):
            lines.append(f"{i}. [{lesson.effectiveness*100:.0f}%] {lesson.lesson}")

        lines.extend([
            "",
            "=" * 50
        ])

        return "\n".join(lines)

    # ==================== 自适应策略接口 ====================
    # 这些方法将真正的进化算法暴露给 auto_trader 使用

    def get_adapted_parameters(self, symbol: str, market_data: dict) -> Dict[str, float]:
        """
        获取自适应参数 - 核心进化接口

        使用贝叶斯推断和汤普森采样生成最优参数：
        - 对不确定的参数探索更多（汤普森采样）
        - 对确定有效的参数利用更多（贝叶斯后验）
        - 根据市场上下文选择参数（上下文强盗）

        Args:
            symbol: 交易品种
            market_data: 市场数据（含 volatility, trend_strength, momentum）

        Returns:
            适应后的参数字典：
            - signal_threshold: 信号强度阈值
            - momentum_threshold: 动量阈值
            - max_spread_percent: 最大点差百分比
            - stop_loss_atr_mult: 止损ATR倍数
            - take_profit_atr_mult: 止盈ATR倍数
            - min_hold_minutes: 最小持仓分钟
            - lookback_points: 回看数据点数
        """
        return self.adaptive_strategy.get_adapted_parameters(symbol, market_data)

    def should_trade_in_context(self, market_data: dict) -> tuple:
        """
        判断当前市场上下文是否值得交易

        基于历史学习判断：如果该上下文的胜率置信下界太低，暂停交易
        使用 Wilson 置信区间防止过早放弃或过度自信

        Args:
            market_data: 市场数据

        Returns:
            (是否交易, 原因)
        """
        context = self.adaptive_strategy.classify_context(market_data)
        return self.adaptive_strategy.should_trade_with_current_params(context)

    def get_evolution_insights(self) -> Dict:
        """
        获取进化洞察 - 了解系统学到了什么

        Returns:
            每个上下文的学习结果和最优参数
        """
        return self.adaptive_strategy.get_parameter_insights()

    def get_recommended_config(self) -> Dict[str, float]:
        """
        获取综合推荐配置

        基于所有上下文的加权平均，返回全局最优参数
        """
        return self.adaptive_strategy.get_recommended_config()

    def generate_evolution_report(self) -> str:
        """
        生成进化报告 - 详细的策略进化情况
        """
        return self.adaptive_strategy.generate_evolution_report()

    def shutdown(self):
        """关闭引擎"""
        # 取消所有定时器
        for timer in self._check_timers.values():
            timer.cancel()
        self._check_timers.clear()

        # 保存数据
        self._save_data()

        self.logger.info("进化引擎已关闭")


# 便捷函数
def create_evolution_engine(auto_trader=None, data_dir: str = None) -> EvolutionEngine:
    """创建进化引擎实例"""
    return EvolutionEngine(data_dir=data_dir, auto_trader=auto_trader)
