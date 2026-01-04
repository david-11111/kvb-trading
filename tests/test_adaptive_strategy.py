"""
自适应策略测试模块

测试覆盖：
1. 贝叶斯参数更新
2. 汤普森采样
3. 上下文分类
4. 参数进化
5. 置信度计算
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from adaptive_strategy import (
    AdaptiveStrategy,
    ParameterBelief,
    ContextualArm,
    create_adaptive_strategy,
)


class TestParameterBelief:
    """测试贝叶斯参数信念"""

    def test_initial_belief(self):
        """测试初始信念"""
        belief = ParameterBelief(name="test", alpha=1.0, beta=1.0)
        assert belief.mean == 0.5  # 均匀先验
        assert belief.confidence < 0.1  # 低置信度

    def test_bayesian_update_success(self):
        """测试成功后的贝叶斯更新"""
        belief = ParameterBelief(name="test", alpha=1.0, beta=1.0)
        initial_mean = belief.mean

        belief.update(success=True, weight=1.0)

        assert belief.alpha == 2.0
        assert belief.beta == 1.0
        assert belief.mean > initial_mean  # 成功后均值上升

    def test_bayesian_update_failure(self):
        """测试失败后的贝叶斯更新"""
        belief = ParameterBelief(name="test", alpha=1.0, beta=1.0)
        initial_mean = belief.mean

        belief.update(success=False, weight=1.0)

        assert belief.alpha == 1.0
        assert belief.beta == 2.0
        assert belief.mean < initial_mean  # 失败后均值下降

    def test_weighted_update(self):
        """测试带权重的更新"""
        belief = ParameterBelief(name="test", alpha=1.0, beta=1.0)

        belief.update(success=True, weight=2.0)

        assert belief.alpha == 3.0  # 1.0 + 2.0

    def test_thompson_sampling(self):
        """测试汤普森采样"""
        belief = ParameterBelief(
            name="test",
            alpha=10.0,
            beta=2.0,
            min_value=0.3,
            max_value=0.9
        )

        # 采样多次验证范围
        samples = [belief.sample() for _ in range(100)]

        assert all(0.3 <= s <= 0.9 for s in samples)
        # 高alpha应该导致采样值偏高
        assert np.mean(samples) > 0.6

    def test_confidence_increases_with_data(self):
        """测试置信度随数据增加而提高"""
        belief = ParameterBelief(name="test", alpha=1.0, beta=1.0)
        initial_confidence = belief.confidence

        for _ in range(10):
            belief.update(success=True, weight=1.0)

        assert belief.confidence > initial_confidence

    def test_decay(self):
        """测试遗忘因子"""
        belief = ParameterBelief(name="test", alpha=10.0, beta=10.0)

        belief.decay(factor=0.5)

        assert belief.alpha == 5.0
        assert belief.beta == 5.0

    def test_decay_preserves_minimum(self):
        """测试遗忘保持最小值"""
        belief = ParameterBelief(name="test", alpha=1.5, beta=1.5)

        belief.decay(factor=0.1)

        assert belief.alpha >= 1.0
        assert belief.beta >= 1.0

    def test_variance_calculation(self):
        """测试方差计算"""
        belief = ParameterBelief(name="test", alpha=1.0, beta=1.0)

        # Beta(1,1)的方差 = 1*1 / (4 * 3) = 1/12
        expected_variance = 1/12
        assert abs(belief.variance - expected_variance) < 0.001


class TestContextualArm:
    """测试上下文强盗臂"""

    def test_arm_creation(self):
        """测试臂创建"""
        arm = ContextualArm(context="trending_high_vol")

        assert arm.context == "trending_high_vol"
        assert arm.total_trades == 0
        assert arm.win_count == 0

    def test_arm_with_parameters(self):
        """测试带参数的臂"""
        params = {
            "signal_threshold": ParameterBelief(name="signal_threshold"),
            "momentum_threshold": ParameterBelief(name="momentum_threshold"),
        }
        arm = ContextualArm(context="ranging_low_vol", parameters=params)

        assert len(arm.parameters) == 2
        assert "signal_threshold" in arm.parameters


class TestAdaptiveStrategy:
    """测试自适应策略"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def strategy(self, temp_dir):
        """策略实例fixture"""
        return AdaptiveStrategy(data_dir=temp_dir)

    def test_strategy_initialization(self, strategy):
        """测试策略初始化"""
        assert strategy.exploration_rate == 0.2
        assert strategy.total_adaptations == 0
        assert len(strategy.context_arms) == 0

    def test_context_classification_trending_high_vol(self, strategy):
        """测试趋势+高波动分类"""
        market_data = {
            "volatility": 0.02,
            "trend_strength": 0.6,
            "momentum": 0.05
        }

        context = strategy.classify_context(market_data)

        assert context == "trending_high_vol"

    def test_context_classification_trending_low_vol(self, strategy):
        """测试趋势+低波动分类"""
        market_data = {
            "volatility": 0.01,
            "trend_strength": 0.6,
            "momentum": 0.05
        }

        context = strategy.classify_context(market_data)

        assert context == "trending_low_vol"

    def test_context_classification_ranging_high_vol(self, strategy):
        """测试震荡+高波动分类"""
        market_data = {
            "volatility": 0.02,
            "trend_strength": 0.2,
            "momentum": 0.01
        }

        context = strategy.classify_context(market_data)

        assert context == "ranging_high_vol"

    def test_context_classification_ranging_low_vol(self, strategy):
        """测试震荡+低波动分类"""
        market_data = {
            "volatility": 0.01,
            "trend_strength": 0.2,
            "momentum": 0.01
        }

        context = strategy.classify_context(market_data)

        assert context == "ranging_low_vol"

    def test_get_adapted_parameters(self, strategy):
        """测试获取自适应参数"""
        market_data = {
            "volatility": 0.01,
            "trend_strength": 0.5,
            "momentum": 0.02
        }

        params = strategy.get_adapted_parameters("USOIL", market_data)

        # 验证返回所有可进化参数
        assert "signal_threshold" in params
        assert "momentum_threshold" in params
        assert "stop_loss_atr_mult" in params
        assert "take_profit_atr_mult" in params

        # 验证参数在合理范围内
        assert 0.3 <= params["signal_threshold"] <= 0.9
        assert 0.01 <= params["momentum_threshold"] <= 0.10

    def test_adapted_parameters_creates_arm(self, strategy):
        """测试获取参数时创建臂"""
        market_data = {"volatility": 0.01, "trend_strength": 0.5}

        strategy.get_adapted_parameters("USOIL", market_data)

        assert "trending_low_vol" in strategy.context_arms

    def test_report_trade_result_profit(self, strategy):
        """测试盈利交易反馈"""
        market_data = {"volatility": 0.01, "trend_strength": 0.5}

        # 先获取参数（注册交易）
        strategy.get_adapted_parameters("USOIL", market_data)

        # 报告盈利结果（is_final=True 表示最终平仓）
        strategy.report_trade_result(
            symbol="USOIL",
            pnl=100.0,
            pnl_percent=0.02,
            hold_duration=300,
            is_final=True  # 最终平仓时才更新统计
        )

        # 验证更新
        arm = strategy.context_arms["trending_low_vol"]
        assert arm.total_trades == 1
        assert arm.win_count == 1

    def test_report_trade_result_loss(self, strategy):
        """测试亏损交易反馈"""
        market_data = {"volatility": 0.01, "trend_strength": 0.5}

        strategy.get_adapted_parameters("USOIL", market_data)
        strategy.report_trade_result(
            symbol="USOIL",
            pnl=-50.0,
            pnl_percent=-0.01,
            hold_duration=300,
            is_final=True  # 最终平仓时才更新统计
        )

        arm = strategy.context_arms["trending_low_vol"]
        assert arm.total_trades == 1
        assert arm.win_count == 0

    def test_exploration_rate_decay(self, strategy):
        """测试探索率衰减"""
        market_data = {"volatility": 0.01, "trend_strength": 0.5}
        initial_rate = strategy.exploration_rate

        # 多次交易（is_final=True 才会触发探索率衰减）
        for _ in range(10):
            strategy.get_adapted_parameters("USOIL", market_data)
            strategy.report_trade_result("USOIL", 10.0, 0.01, 300, is_final=True)

        assert strategy.exploration_rate < initial_rate
        assert strategy.exploration_rate >= 0.05  # 最小值

    def test_should_trade_new_context(self, strategy):
        """测试新上下文应该交易"""
        should, reason = strategy.should_trade_with_current_params("trending_high_vol")

        assert should is True
        assert "新上下文" in reason or "探索" in reason

    def test_should_trade_insufficient_samples(self, strategy):
        """测试样本不足时应该交易"""
        # 创建一个只有3次交易的臂
        arm = strategy._init_context_arm("trending_low_vol")
        arm.total_trades = 3
        arm.win_count = 2
        strategy.context_arms["trending_low_vol"] = arm

        should, reason = strategy.should_trade_with_current_params("trending_low_vol")

        assert should is True
        assert "样本不足" in reason

    def test_should_not_trade_low_confidence(self, strategy):
        """测试低置信度时不应交易"""
        # 创建一个胜率很低的臂
        arm = strategy._init_context_arm("ranging_high_vol")
        arm.total_trades = 20
        arm.win_count = 3  # 15% 胜率
        strategy.context_arms["ranging_high_vol"] = arm

        should, reason = strategy.should_trade_with_current_params("ranging_high_vol")

        assert should is False
        assert "置信下界过低" in reason

    def test_get_parameter_insights(self, strategy):
        """测试获取参数洞察"""
        market_data = {"volatility": 0.01, "trend_strength": 0.5}

        # 做一些交易（is_final=True 才会更新统计）
        for i in range(5):
            strategy.get_adapted_parameters("USOIL", market_data)
            strategy.report_trade_result("USOIL", 10.0 if i % 2 == 0 else -5.0, 0.01, 300, is_final=True)

        insights = strategy.get_parameter_insights()

        assert "trending_low_vol" in insights
        assert "parameters" in insights["trending_low_vol"]
        assert "total_trades" in insights["trending_low_vol"]

    def test_get_recommended_config(self, strategy):
        """测试获取推荐配置"""
        # 无交易时返回默认值
        config = strategy.get_recommended_config()

        for name, spec in AdaptiveStrategy.EVOLVABLE_PARAMETERS.items():
            assert name in config
            assert config[name] == spec["default"]

    def test_save_and_load_beliefs(self, temp_dir):
        """测试保存和加载信念"""
        strategy1 = AdaptiveStrategy(data_dir=temp_dir)
        market_data = {"volatility": 0.01, "trend_strength": 0.5}

        # 做一些交易（is_final=True 才会更新统计）
        for _ in range(5):
            strategy1.get_adapted_parameters("USOIL", market_data)
            strategy1.report_trade_result("USOIL", 10.0, 0.01, 300, is_final=True)

        # 新实例加载
        strategy2 = AdaptiveStrategy(data_dir=temp_dir)

        assert strategy2.total_adaptations == 5
        assert "trending_low_vol" in strategy2.context_arms
        assert strategy2.context_arms["trending_low_vol"].total_trades == 5

    def test_generate_evolution_report(self, strategy):
        """测试生成进化报告"""
        market_data = {"volatility": 0.01, "trend_strength": 0.5}

        for _ in range(3):
            strategy.get_adapted_parameters("USOIL", market_data)
            strategy.report_trade_result("USOIL", 10.0, 0.01, 300, is_final=True)

        report = strategy.generate_evolution_report()

        assert "策略进化报告" in report
        assert "trending_low_vol" in report
        assert "综合推荐配置" in report


class TestEvolutionMechanics:
    """测试进化机制"""

    @pytest.fixture
    def temp_dir(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    def test_parameters_evolve_with_success(self, temp_dir):
        """测试参数随成功交易进化"""
        strategy = AdaptiveStrategy(data_dir=temp_dir)
        market_data = {"volatility": 0.01, "trend_strength": 0.5}

        # 获取初始参数
        initial_params = strategy.get_adapted_parameters("USOIL", market_data)
        strategy.report_trade_result("USOIL", 100.0, 0.05, 300, is_final=True)

        # 多次成功交易后（is_final=True 才会累计学习）
        for _ in range(20):
            strategy.get_adapted_parameters("USOIL", market_data)
            strategy.report_trade_result("USOIL", 100.0, 0.05, 300, is_final=True)

        # 获取进化后的参数
        final_params = strategy.get_adapted_parameters("USOIL", market_data)

        # 验证置信度提高（21次交易后 alpha 足够高）
        arm = strategy.context_arms["trending_low_vol"]
        for belief in arm.parameters.values():
            assert belief.alpha > 2.0  # alpha 应该显著增加

    def test_different_contexts_evolve_independently(self, temp_dir):
        """测试不同上下文独立进化"""
        strategy = AdaptiveStrategy(data_dir=temp_dir)

        # 在趋势市场盈利（is_final=True 才会更新统计）
        trending_data = {"volatility": 0.01, "trend_strength": 0.6}
        for _ in range(10):
            strategy.get_adapted_parameters("USOIL", trending_data)
            strategy.report_trade_result("USOIL", 100.0, 0.05, 300, is_final=True)

        # 在震荡市场亏损
        ranging_data = {"volatility": 0.01, "trend_strength": 0.2}
        for _ in range(10):
            strategy.get_adapted_parameters("XAUUSD", ranging_data)
            strategy.report_trade_result("XAUUSD", -50.0, -0.02, 300, is_final=True)

        # 验证趋势市场胜率高
        trending_arm = strategy.context_arms["trending_low_vol"]
        ranging_arm = strategy.context_arms["ranging_low_vol"]

        assert trending_arm.win_count == 10
        assert ranging_arm.win_count == 0

    def test_weight_based_on_pnl_magnitude(self, temp_dir):
        """测试权重基于盈亏幅度"""
        strategy = AdaptiveStrategy(data_dir=temp_dir)
        market_data = {"volatility": 0.01, "trend_strength": 0.5}

        # 小盈利（使用 is_final 来触发贝叶斯更新）
        strategy.get_adapted_parameters("USOIL", market_data)
        strategy.report_trade_result("USOIL", 10.0, 0.005, 300, is_final=True)
        small_alpha = strategy.context_arms["trending_low_vol"].parameters["signal_threshold"].alpha

        # 大盈利
        strategy.get_adapted_parameters("USOIL", market_data)
        strategy.report_trade_result("USOIL", 200.0, 0.10, 300, is_final=True)
        large_alpha = strategy.context_arms["trending_low_vol"].parameters["signal_threshold"].alpha

        # 大盈利应该增加更多alpha
        assert large_alpha - small_alpha > 1.0


class TestIntegrationWithEvolutionEngine:
    """测试与进化引擎的集成"""

    @pytest.fixture
    def temp_dir(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    def test_create_adaptive_strategy(self, temp_dir):
        """测试便捷创建函数"""
        strategy = create_adaptive_strategy(data_dir=temp_dir)

        assert isinstance(strategy, AdaptiveStrategy)
        assert strategy.data_dir == Path(temp_dir)

    def test_active_trades_tracking(self, temp_dir):
        """测试活跃交易跟踪"""
        strategy = AdaptiveStrategy(data_dir=temp_dir)
        market_data = {"volatility": 0.01, "trend_strength": 0.5}

        # 获取参数会注册活跃交易
        strategy.get_adapted_parameters("USOIL", market_data)
        assert "USOIL" in strategy._active_trades

        # 中间检查不会移除活跃交易
        strategy.report_trade_result("USOIL", 50.0, 0.02, 150, is_final=False)
        assert "USOIL" in strategy._active_trades

        # 最终平仓才会移除活跃交易
        strategy.report_trade_result("USOIL", 100.0, 0.05, 300, is_final=True)
        assert "USOIL" not in strategy._active_trades
