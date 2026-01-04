"""
自适应策略模块 - 真正的自我进化

核心算法：
1. 贝叶斯参数更新 - 根据交易结果动态调整策略参数
2. 汤普森采样 - 在探索与利用之间平衡
3. 滑动窗口评估 - 近期表现权重更高
4. 上下文强盗(Contextual Bandit) - 根据市场状态选择最优参数

进化机制：
- 每次交易结果反馈到参数先验分布
- 成功的参数组合强化，失败的弱化
- 自动发现最优参数区间
"""

import json
import time
import math
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats


@dataclass
class ParameterBelief:
    """参数的贝叶斯信念（Beta分布）"""
    name: str
    alpha: float = 1.0      # 成功次数 + 1
    beta: float = 1.0       # 失败次数 + 1
    min_value: float = 0.0  # 参数最小值
    max_value: float = 1.0  # 参数最大值
    current_value: float = 0.5  # 当前使用的值

    @property
    def mean(self) -> float:
        """期望值"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """方差（不确定性）"""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))

    @property
    def confidence(self) -> float:
        """置信度 (0-1)，样本越多越高"""
        total = self.alpha + self.beta - 2  # 减去初始值
        return 1 - 1 / (1 + total / 10)  # 10次交易达到约50%置信度

    def sample(self) -> float:
        """汤普森采样 - 从后验分布采样"""
        sampled_prob = np.random.beta(self.alpha, self.beta)
        # 映射到参数范围
        return self.min_value + sampled_prob * (self.max_value - self.min_value)

    def update(self, success: bool, weight: float = 1.0):
        """贝叶斯更新"""
        if success:
            self.alpha += weight
        else:
            self.beta += weight

    def decay(self, factor: float = 0.99):
        """遗忘因子 - 让旧数据影响逐渐减弱"""
        # 保持最小值，避免完全遗忘
        self.alpha = max(1.0, self.alpha * factor)
        self.beta = max(1.0, self.beta * factor)


@dataclass
class ContextualArm:
    """上下文强盗的臂（策略配置）"""
    context: str           # 市场上下文（如 "trending_high_vol"）
    parameters: Dict[str, ParameterBelief] = field(default_factory=dict)
    total_trades: int = 0
    total_profit: float = 0.0
    win_count: int = 0
    last_update: float = 0.0


class AdaptiveStrategy:
    """
    自适应策略 - 真正的自我进化

    进化原理：
    1. 将策略参数建模为Beta分布
    2. 使用贝叶斯推断从交易结果学习
    3. 汤普森采样平衡探索与利用
    4. 上下文感知 - 不同市场状态用不同参数
    """

    # 需要进化的参数及其范围
    EVOLVABLE_PARAMETERS = {
        "signal_threshold": {"min": 0.3, "max": 0.9, "default": 0.6},
        "momentum_threshold": {"min": 0.01, "max": 0.10, "default": 0.05},
        "max_spread_percent": {"min": 0.02, "max": 0.15, "default": 0.08},
        "stop_loss_atr_mult": {"min": 1.0, "max": 4.0, "default": 2.0},
        "take_profit_atr_mult": {"min": 1.5, "max": 6.0, "default": 3.0},
        "min_hold_minutes": {"min": 1, "max": 30, "default": 5},
        "lookback_points": {"min": 5, "max": 30, "default": 12},
    }

    # 市场上下文分类
    CONTEXTS = [
        "trending_low_vol",    # 趋势 + 低波动
        "trending_high_vol",   # 趋势 + 高波动
        "ranging_low_vol",     # 震荡 + 低波动
        "ranging_high_vol",    # 震荡 + 高波动
    ]

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or "evolution_data")
        self.data_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("AdaptiveStrategy")

        # 每个上下文的参数信念
        self.context_arms: Dict[str, ContextualArm] = {}

        # 当前活跃的交易及其使用的参数
        self._active_trades: Dict[str, Dict] = {}

        # 全局统计
        self.total_adaptations = 0
        self.exploration_rate = 0.2  # 探索率，随时间衰减

        # 加载历史数据
        self._load_beliefs()

        self.logger.info(f"自适应策略初始化，已学习 {self._count_experiences()} 次交易经验")

    def _init_context_arm(self, context: str) -> ContextualArm:
        """初始化上下文的参数信念"""
        parameters = {}
        for name, config in self.EVOLVABLE_PARAMETERS.items():
            parameters[name] = ParameterBelief(
                name=name,
                alpha=1.0,
                beta=1.0,
                min_value=config["min"],
                max_value=config["max"],
                current_value=config["default"]
            )
        return ContextualArm(context=context, parameters=parameters)

    def _count_experiences(self) -> int:
        """统计学习的经验数量"""
        total = 0
        for arm in self.context_arms.values():
            total += arm.total_trades
        return total

    def classify_context(self, market_data: dict) -> str:
        """
        将市场数据分类为上下文

        Args:
            market_data: 包含 volatility, trend_strength, momentum 等

        Returns:
            上下文字符串
        """
        volatility = market_data.get("volatility", 0.01)
        trend_strength = abs(market_data.get("trend_strength", 0))

        # 判断趋势/震荡
        is_trending = trend_strength > 0.4

        # 判断波动率高低
        is_high_vol = volatility > 0.015

        if is_trending:
            return "trending_high_vol" if is_high_vol else "trending_low_vol"
        else:
            return "ranging_high_vol" if is_high_vol else "ranging_low_vol"

    def get_adapted_parameters(self, symbol: str, market_data: dict) -> Dict[str, float]:
        """
        获取自适应参数（核心进化接口）

        使用汤普森采样在探索与利用之间平衡：
        - 对不确定的参数探索更多
        - 对确定有效的参数利用更多

        Args:
            symbol: 交易品种
            market_data: 市场数据

        Returns:
            适应后的参数字典
        """
        context = self.classify_context(market_data)

        # 获取或创建该上下文的参数信念
        if context not in self.context_arms:
            self.context_arms[context] = self._init_context_arm(context)

        arm = self.context_arms[context]
        adapted_params = {}

        for name, belief in arm.parameters.items():
            # 汤普森采样：从后验分布采样
            if random.random() < self.exploration_rate:
                # 探索：使用采样值
                value = belief.sample()
            else:
                # 利用：使用期望值
                ratio = belief.mean
                value = belief.min_value + ratio * (belief.max_value - belief.min_value)

            adapted_params[name] = value
            belief.current_value = value

        # 记录活跃交易使用的参数
        self._active_trades[symbol] = {
            "context": context,
            "parameters": adapted_params.copy(),
            "entry_time": time.time()
        }

        self.logger.debug(f"[{symbol}] 上下文:{context}, 自适应参数: {adapted_params}")

        return adapted_params

    def report_trade_result(self, symbol: str, pnl: float, pnl_percent: float,
                           hold_duration: float, is_final: bool = False):
        """
        报告交易结果 - 进化的核心反馈

        Args:
            symbol: 交易品种
            pnl: 盈亏金额
            pnl_percent: 盈亏百分比
            hold_duration: 持仓时长（秒）
            is_final: 是否为最终平仓（True则移除交易记录）
        """
        if symbol not in self._active_trades:
            self.logger.warning(f"未找到 {symbol} 的活跃交易记录")
            return

        # 中间检查不移除，最终平仓才移除
        if is_final:
            trade_info = self._active_trades.pop(symbol)
        else:
            trade_info = self._active_trades[symbol]

        context = trade_info["context"]
        used_params = trade_info["parameters"]

        if context not in self.context_arms:
            return

        arm = self.context_arms[context]

        # 判断成功与否
        success = pnl > 0

        # 计算更新权重
        # 中间检查权重较小，最终平仓权重较大
        base_weight = min(abs(pnl_percent) * 20 + 0.5, 3.0)
        weight = base_weight * (1.5 if is_final else 0.5)  # 中间检查权重减半

        # 贝叶斯更新所有使用的参数
        for name, belief in arm.parameters.items():
            belief.update(success, weight)

        # 只在最终平仓时更新交易统计
        if is_final:
            arm.total_trades += 1
            arm.total_profit += pnl
            if success:
                arm.win_count += 1

        arm.last_update = time.time()
        self.total_adaptations += 1

        # 衰减探索率
        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)

        # 定期遗忘旧数据
        if arm.total_trades % 50 == 0:
            for belief in arm.parameters.values():
                belief.decay(0.95)

        # 保存
        self._save_beliefs()

        check_type = "最终平仓" if is_final else f"{int(hold_duration/60)}分钟检查"
        self.logger.info(
            f"[{symbol}] {check_type}学习: {'盈利' if success else '亏损'} {pnl:.2f} "
            f"(权重:{weight:.2f}), 上下文:{context}, 累计适应:{self.total_adaptations}次"
        )

    def get_parameter_insights(self) -> Dict:
        """
        获取参数进化洞察

        Returns:
            每个参数在各上下文下的学习结果
        """
        insights = {}

        for context, arm in self.context_arms.items():
            if arm.total_trades == 0:
                continue

            context_insights = {
                "total_trades": arm.total_trades,
                "win_rate": arm.win_count / arm.total_trades if arm.total_trades > 0 else 0,
                "total_profit": arm.total_profit,
                "parameters": {}
            }

            for name, belief in arm.parameters.items():
                optimal_value = belief.min_value + belief.mean * (belief.max_value - belief.min_value)
                context_insights["parameters"][name] = {
                    "optimal_value": round(optimal_value, 4),
                    "confidence": round(belief.confidence, 2),
                    "uncertainty": round(math.sqrt(belief.variance), 4),
                    "current_value": round(belief.current_value, 4),
                }

            insights[context] = context_insights

        return insights

    def get_recommended_config(self) -> Dict[str, float]:
        """
        获取综合推荐配置

        基于所有上下文的加权平均，权重为交易次数
        """
        weighted_params = defaultdict(float)
        total_weight = 0

        for context, arm in self.context_arms.items():
            if arm.total_trades == 0:
                continue

            weight = arm.total_trades * (arm.win_count / arm.total_trades if arm.total_trades > 0 else 0.5)

            for name, belief in arm.parameters.items():
                optimal = belief.min_value + belief.mean * (belief.max_value - belief.min_value)
                weighted_params[name] += optimal * weight

            total_weight += weight

        if total_weight == 0:
            # 返回默认值
            return {name: config["default"] for name, config in self.EVOLVABLE_PARAMETERS.items()}

        return {name: round(value / total_weight, 4) for name, value in weighted_params.items()}

    def should_trade_with_current_params(self, context: str) -> Tuple[bool, str]:
        """
        判断当前参数是否值得交易

        基于置信区间判断：如果参数胜率的置信下界太低，暂停交易

        Returns:
            (是否交易, 原因)
        """
        if context not in self.context_arms:
            return True, "新上下文，需要探索"

        arm = self.context_arms[context]

        if arm.total_trades < 5:
            return True, f"样本不足({arm.total_trades}次)，继续探索"

        win_rate = arm.win_count / arm.total_trades

        # 使用Wilson置信区间的下界
        n = arm.total_trades
        z = 1.96  # 95%置信度
        denominator = 1 + z**2 / n
        center = (win_rate + z**2 / (2*n)) / denominator
        margin = z * math.sqrt((win_rate * (1 - win_rate) + z**2 / (4*n)) / n) / denominator
        lower_bound = center - margin

        if lower_bound < 0.35:
            return False, f"胜率置信下界过低({lower_bound:.1%})，暂停该上下文"

        return True, f"胜率{win_rate:.1%}，置信下界{lower_bound:.1%}"

    def generate_evolution_report(self) -> str:
        """生成进化报告"""
        lines = [
            "=" * 60,
            "【策略进化报告】",
            f"累计学习次数: {self._count_experiences()}",
            f"当前探索率: {self.exploration_rate:.1%}",
            "=" * 60,
            ""
        ]

        insights = self.get_parameter_insights()

        for context, data in insights.items():
            lines.append(f"📊 上下文: {context}")
            lines.append(f"   交易次数: {data['total_trades']}, 胜率: {data['win_rate']:.1%}")
            lines.append(f"   累计盈亏: {data['total_profit']:.2f}")
            lines.append("   进化后的最优参数:")

            for name, param_data in data["parameters"].items():
                confidence_bar = "●" * int(param_data["confidence"] * 10) + "○" * (10 - int(param_data["confidence"] * 10))
                lines.append(
                    f"      {name}: {param_data['optimal_value']:.4f} "
                    f"[置信度: {confidence_bar} {param_data['confidence']:.0%}]"
                )
            lines.append("")

        # 综合推荐
        recommended = self.get_recommended_config()
        lines.append("🎯 综合推荐配置:")
        for name, value in recommended.items():
            default = self.EVOLVABLE_PARAMETERS[name]["default"]
            change = ((value - default) / default) * 100 if default else 0
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            lines.append(f"   {name}: {value:.4f} ({arrow}{abs(change):.1f}% vs 默认)")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _save_beliefs(self):
        """保存信念状态"""
        data = {}
        for context, arm in self.context_arms.items():
            arm_data = {
                "context": arm.context,
                "total_trades": arm.total_trades,
                "total_profit": arm.total_profit,
                "win_count": arm.win_count,
                "last_update": arm.last_update,
                "parameters": {}
            }
            for name, belief in arm.parameters.items():
                arm_data["parameters"][name] = {
                    "alpha": belief.alpha,
                    "beta": belief.beta,
                    "min_value": belief.min_value,
                    "max_value": belief.max_value,
                    "current_value": belief.current_value
                }
            data[context] = arm_data

        with open(self.data_dir / "adaptive_beliefs.json", "w", encoding="utf-8") as f:
            json.dump({
                "beliefs": data,
                "exploration_rate": self.exploration_rate,
                "total_adaptations": self.total_adaptations,
                "saved_at": time.time()
            }, f, indent=2, ensure_ascii=False)

    def _load_beliefs(self):
        """加载信念状态"""
        path = self.data_dir / "adaptive_beliefs.json"
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.exploration_rate = data.get("exploration_rate", 0.2)
            self.total_adaptations = data.get("total_adaptations", 0)

            for context, arm_data in data.get("beliefs", {}).items():
                arm = ContextualArm(
                    context=context,
                    total_trades=arm_data.get("total_trades", 0),
                    total_profit=arm_data.get("total_profit", 0),
                    win_count=arm_data.get("win_count", 0),
                    last_update=arm_data.get("last_update", 0)
                )

                for name, belief_data in arm_data.get("parameters", {}).items():
                    arm.parameters[name] = ParameterBelief(
                        name=name,
                        alpha=belief_data.get("alpha", 1.0),
                        beta=belief_data.get("beta", 1.0),
                        min_value=belief_data.get("min_value", 0.0),
                        max_value=belief_data.get("max_value", 1.0),
                        current_value=belief_data.get("current_value", 0.5)
                    )

                self.context_arms[context] = arm

            self.logger.info(f"加载 {len(self.context_arms)} 个上下文的进化数据")

        except Exception as e:
            self.logger.error(f"加载进化数据失败: {e}")


# 便捷函数
def create_adaptive_strategy(data_dir: str = None) -> AdaptiveStrategy:
    """创建自适应策略实例"""
    return AdaptiveStrategy(data_dir=data_dir)
