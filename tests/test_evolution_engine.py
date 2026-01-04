"""
进化引擎测试
测试覆盖：反思生成、评分系统、经验学习、持久化
"""

import pytest
import time
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evolution_engine import (
    EvolutionEngine,
    ReflectionType,
    MarketCondition,
    EntryContext,
    ReflectionRecord,
    LessonLearned,
    EvolutionScore,
    create_evolution_engine
)


class TestEvolutionEngineInit:
    """初始化测试"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_init_creates_data_dir(self, temp_dir):
        """初始化创建数据目录"""
        engine = EvolutionEngine(data_dir=temp_dir)
        assert Path(temp_dir).exists()

    def test_init_default_score(self, temp_dir):
        """初始评分为1000"""
        engine = EvolutionEngine(data_dir=temp_dir)
        assert engine.score.total_score == 1000

    def test_init_empty_lessons(self, temp_dir):
        """初始无教训"""
        engine = EvolutionEngine(data_dir=temp_dir)
        assert len(engine.lessons) == 0

    def test_create_evolution_engine_function(self, temp_dir):
        """便捷创建函数"""
        engine = create_evolution_engine(data_dir=temp_dir)
        assert isinstance(engine, EvolutionEngine)


class TestMarketConditionAnalysis:
    """市场状态分析测试"""

    @pytest.fixture
    def engine(self):
        temp = tempfile.mkdtemp()
        engine = EvolutionEngine(data_dir=temp)
        yield engine
        shutil.rmtree(temp, ignore_errors=True)

    def test_volatile_market(self, engine):
        """高波动市场"""
        market_data = {"volatility": 0.03, "trend_strength": 0.5, "momentum": 0}
        result = engine._analyze_market_condition(market_data)
        assert result == MarketCondition.VOLATILE

    def test_calm_market(self, engine):
        """低波动市场"""
        market_data = {"volatility": 0.003, "trend_strength": 0.5, "momentum": 0}
        result = engine._analyze_market_condition(market_data)
        assert result == MarketCondition.CALM

    def test_trending_up(self, engine):
        """上升趋势"""
        market_data = {"volatility": 0.01, "trend_strength": 0.8, "momentum": 0.5}
        result = engine._analyze_market_condition(market_data)
        assert result == MarketCondition.TRENDING_UP

    def test_trending_down(self, engine):
        """下降趋势"""
        market_data = {"volatility": 0.01, "trend_strength": 0.8, "momentum": -0.5}
        result = engine._analyze_market_condition(market_data)
        assert result == MarketCondition.TRENDING_DOWN

    def test_ranging_market(self, engine):
        """震荡市场"""
        market_data = {"volatility": 0.01, "trend_strength": 0.2, "momentum": 0}
        result = engine._analyze_market_condition(market_data)
        assert result == MarketCondition.RANGING


class TestPnLCalculation:
    """盈亏计算测试"""

    @pytest.fixture
    def engine(self):
        temp = tempfile.mkdtemp()
        engine = EvolutionEngine(data_dir=temp)
        yield engine
        shutil.rmtree(temp, ignore_errors=True)

    def test_long_profit(self, engine):
        """多头盈利"""
        pnl, pnl_percent = engine._calculate_pnl("long", 100.0, 110.0)
        assert pnl == 10.0
        assert pnl_percent == 0.1

    def test_long_loss(self, engine):
        """多头亏损"""
        pnl, pnl_percent = engine._calculate_pnl("long", 100.0, 90.0)
        assert pnl == -10.0
        assert pnl_percent == -0.1

    def test_short_profit(self, engine):
        """空头盈利"""
        pnl, pnl_percent = engine._calculate_pnl("short", 100.0, 90.0)
        assert pnl == 10.0
        assert pnl_percent == 0.1

    def test_short_loss(self, engine):
        """空头亏损"""
        pnl, pnl_percent = engine._calculate_pnl("short", 100.0, 110.0)
        assert pnl == -10.0
        assert pnl_percent == -0.1


class TestScoreSystem:
    """评分系统测试"""

    @pytest.fixture
    def engine(self):
        temp = tempfile.mkdtemp()
        engine = EvolutionEngine(data_dir=temp)
        yield engine
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def profit_reflection(self, engine):
        """盈利反思记录"""
        context = EntryContext(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            entry_time=time.time() - 300,
            signal_type="momentum",
            signal_strength=0.8,
            market_condition=MarketCondition.TRENDING_UP,
            indicators={},
            momentum=0.5,
            volatility=0.01,
            spread=1.0,
            recent_prices=[3000 + i for i in range(20)],
            reason="强势上涨信号"
        )
        return ReflectionRecord(
            id="REF_TEST_001",
            timestamp=time.time(),
            symbol="ETHUSD",
            reflection_type=ReflectionType.PROFIT,
            entry_context=context,
            check_price=3050.0,
            pnl=50.0,
            pnl_percent=0.0167,  # 1.67%
            hold_duration=300,
            analysis="成功的趋势交易",
            lessons=["顺势交易效果好"],
            score_change=0,
            suggestions=[]
        )

    @pytest.fixture
    def loss_reflection(self, engine):
        """亏损反思记录"""
        context = EntryContext(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            entry_time=time.time() - 300,
            signal_type="momentum",
            signal_strength=0.4,
            market_condition=MarketCondition.RANGING,
            indicators={},
            momentum=-0.2,
            volatility=0.02,
            spread=5.0,
            recent_prices=[3000 - i for i in range(20)],
            reason="弱势信号"
        )
        return ReflectionRecord(
            id="REF_TEST_002",
            timestamp=time.time(),
            symbol="ETHUSD",
            reflection_type=ReflectionType.LOSS,
            entry_context=context,
            check_price=2950.0,
            pnl=-50.0,
            pnl_percent=-0.0167,
            hold_duration=300,
            analysis="逆势交易失败",
            lessons=["避免震荡市逆势操作"],
            score_change=0,
            suggestions=["等待趋势明确"]
        )

    def test_profit_score_increase(self, engine, profit_reflection):
        """盈利加分"""
        initial_score = engine.score.total_score
        score_change = engine._update_score(profit_reflection)

        assert score_change > 0
        assert engine.score.total_score > initial_score
        assert engine.score.profit_reflections == 1

    def test_loss_score_decrease(self, engine, loss_reflection):
        """亏损减分"""
        initial_score = engine.score.total_score
        score_change = engine._update_score(loss_reflection)

        assert score_change < 0
        assert engine.score.total_score < initial_score
        assert engine.score.loss_reflections == 1

    def test_win_streak_tracking(self, engine, profit_reflection):
        """连胜追踪"""
        for i in range(3):
            reflection = ReflectionRecord(
                id=f"REF_TEST_{i}",
                timestamp=time.time(),
                symbol="ETHUSD",
                reflection_type=ReflectionType.PROFIT,
                entry_context=profit_reflection.entry_context,
                check_price=3050.0,
                pnl=50.0,
                pnl_percent=0.0167,
                hold_duration=300,
                analysis="成功",
                lessons=[],
                score_change=0,
                suggestions=[]
            )
            engine._update_score(reflection)

        assert engine.score.win_streak == 3
        assert engine.score.max_win_streak == 3
        assert engine.score.lose_streak == 0

    def test_lose_streak_tracking(self, engine, loss_reflection):
        """连败追踪"""
        for i in range(3):
            reflection = ReflectionRecord(
                id=f"REF_TEST_{i}",
                timestamp=time.time(),
                symbol="ETHUSD",
                reflection_type=ReflectionType.LOSS,
                entry_context=loss_reflection.entry_context,
                check_price=2950.0,
                pnl=-50.0,
                pnl_percent=-0.0167,
                hold_duration=300,
                analysis="失败",
                lessons=[],
                score_change=0,
                suggestions=[]
            )
            engine._update_score(reflection)

        assert engine.score.lose_streak == 3
        assert engine.score.max_lose_streak == 3
        assert engine.score.win_streak == 0

    def test_streak_bonus(self, engine, profit_reflection):
        """连胜奖励"""
        scores = []
        for i in range(5):
            reflection = ReflectionRecord(
                id=f"REF_TEST_{i}",
                timestamp=time.time(),
                symbol="ETHUSD",
                reflection_type=ReflectionType.PROFIT,
                entry_context=profit_reflection.entry_context,
                check_price=3050.0,
                pnl=50.0,
                pnl_percent=0.0167,
                hold_duration=300,
                analysis="成功",
                lessons=[],
                score_change=0,
                suggestions=[]
            )
            score_change = engine._update_score(reflection)
            scores.append(score_change)

        # 连胜后应该有额外奖励
        assert scores[3] > scores[1]  # 第4次比第2次多（因为连胜奖励）


class TestLessonLearning:
    """经验学习测试"""

    @pytest.fixture
    def engine(self):
        temp = tempfile.mkdtemp()
        engine = EvolutionEngine(data_dir=temp)
        yield engine
        shutil.rmtree(temp, ignore_errors=True)

    def test_lesson_extraction(self, engine):
        """教训提取"""
        context = EntryContext(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            entry_time=time.time(),
            signal_type="momentum",
            signal_strength=0.8,
            market_condition=MarketCondition.TRENDING_UP,
            indicators={},
            momentum=0.5,
            volatility=0.01,
            spread=1.0,
            recent_prices=[],
            reason="test"
        )

        lessons = engine._extract_lessons(context, ReflectionType.PROFIT, 0.02)

        assert len(lessons) > 0
        assert any("信号" in l or "动量" in l for l in lessons)

    def test_lesson_categorization(self, engine):
        """教训分类"""
        assert engine._categorize_lesson("入场时机太早") == "entry"
        assert engine._categorize_lesson("止损设置过紧") == "exit"
        assert engine._categorize_lesson("仓位过重") == "risk"
        assert engine._categorize_lesson("持仓时间过短") == "timing"
        assert engine._categorize_lesson("其他问题") == "general"

    def test_lesson_persistence(self, engine):
        """教训持久化"""
        context = EntryContext(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            entry_time=time.time(),
            signal_type="momentum",
            signal_strength=0.8,
            market_condition=MarketCondition.TRENDING_UP,
            indicators={},
            momentum=0.5,
            volatility=0.01,
            spread=1.0,
            recent_prices=[],
            reason="test"
        )

        reflection = ReflectionRecord(
            id="REF_TEST_001",
            timestamp=time.time(),
            symbol="ETHUSD",
            reflection_type=ReflectionType.PROFIT,
            entry_context=context,
            check_price=3050.0,
            pnl=50.0,
            pnl_percent=0.0167,
            hold_duration=300,
            analysis="成功",
            lessons=["顺势交易效果好"],
            score_change=10,
            suggestions=[]
        )

        engine._update_lessons(reflection)

        assert len(engine.lessons) > 0
        assert engine.score.lessons_learned > 0


class TestPositionRegistration:
    """持仓注册测试"""

    @pytest.fixture
    def engine_with_trader(self):
        temp = tempfile.mkdtemp()

        # 创建模拟的AutoTrader
        mock_trader = MagicMock()
        mock_trader.latest_prices = {"ETHUSD": {"mid": 3050, "bid": 3049, "ask": 3051}}
        mock_trader.positions = {"ETHUSD": MagicMock()}

        engine = EvolutionEngine(data_dir=temp, auto_trader=mock_trader)
        yield engine
        engine.shutdown()
        shutil.rmtree(temp, ignore_errors=True)

    def test_register_position(self, engine_with_trader):
        """注册持仓"""
        engine = engine_with_trader

        signal_info = {"type": "momentum", "strength": 0.7, "reason": "上涨趋势"}
        market_data = {
            "volatility": 0.01,
            "trend_strength": 0.6,
            "momentum": 0.3,
            "spread": 2.0,
            "indicators": {},
            "recent_prices": [3000 + i for i in range(20)]
        }

        engine.register_position(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            entry_time=time.time(),
            signal_info=signal_info,
            market_data=market_data
        )

        assert "ETHUSD" in engine._pending_checks
        assert "ETHUSD" in engine._check_timers

    def test_unregister_position(self, engine_with_trader):
        """注销持仓"""
        engine = engine_with_trader

        signal_info = {"type": "momentum", "strength": 0.7, "reason": "test"}
        market_data = {"volatility": 0.01, "trend_strength": 0.6, "momentum": 0.3,
                       "spread": 2.0, "indicators": {}, "recent_prices": []}

        engine.register_position("ETHUSD", "long", 3000.0, time.time(),
                                signal_info, market_data)

        engine.unregister_position("ETHUSD")

        assert "ETHUSD" not in engine._pending_checks
        assert "ETHUSD" not in engine._check_timers


class TestReportGeneration:
    """报告生成测试"""

    @pytest.fixture
    def engine_with_history(self):
        temp = tempfile.mkdtemp()
        engine = EvolutionEngine(data_dir=temp)

        # 添加一些模拟数据
        engine.score.profit_reflections = 10
        engine.score.loss_reflections = 5
        engine.score.total_profit_amount = 500
        engine.score.total_loss_amount = 200
        engine.score.max_win_streak = 5
        engine.score.max_lose_streak = 2

        yield engine
        shutil.rmtree(temp, ignore_errors=True)

    def test_daily_report(self, engine_with_history):
        """每日报告生成"""
        report = engine_with_history.generate_daily_report()

        assert "每日进化报告" in report
        assert "当前评分" in report
        assert "胜率" in report
        assert "总盈利" in report

    def test_score_summary(self, engine_with_history):
        """评分摘要"""
        summary = engine_with_history.get_score_summary()

        assert "total_score" in summary
        assert "win_rate" in summary
        assert summary["profit_reflections"] == 10
        assert summary["loss_reflections"] == 5


class TestDataPersistence:
    """数据持久化测试"""

    def test_save_and_load_score(self):
        """评分保存和加载"""
        temp = tempfile.mkdtemp()
        try:
            # 创建引擎并修改评分
            engine1 = EvolutionEngine(data_dir=temp)
            engine1.score.total_score = 1200
            engine1.score.profit_reflections = 5
            engine1._save_data()

            # 创建新引擎加载数据
            engine2 = EvolutionEngine(data_dir=temp)

            assert engine2.score.total_score == 1200
            assert engine2.score.profit_reflections == 5
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_save_and_load_lessons(self):
        """教训保存和加载"""
        temp = tempfile.mkdtemp()
        try:
            engine1 = EvolutionEngine(data_dir=temp)

            # 添加教训
            lesson = LessonLearned(
                id="LESSON_TEST_001",
                created_at=time.time(),
                updated_at=time.time(),
                category="entry",
                condition="trending_up",
                lesson="顺势交易效果好",
                success_count=5,
                failure_count=1,
                effectiveness=0.83,
                is_active=True
            )
            engine1.lessons[lesson.id] = lesson
            engine1._save_data()

            # 加载
            engine2 = EvolutionEngine(data_dir=temp)

            assert "LESSON_TEST_001" in engine2.lessons
            assert engine2.lessons["LESSON_TEST_001"].effectiveness == 0.83
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class TestEdgeCases:
    """边界条件测试"""

    @pytest.fixture
    def engine(self):
        temp = tempfile.mkdtemp()
        engine = EvolutionEngine(data_dir=temp)
        yield engine
        shutil.rmtree(temp, ignore_errors=True)

    def test_zero_entry_price(self, engine):
        """零开仓价"""
        pnl, pnl_percent = engine._calculate_pnl("long", 0.0, 100.0)
        assert pnl == 100.0
        assert pnl_percent == 0  # 避免除零

    def test_negative_score_prevention(self, engine):
        """防止负分"""
        engine.score.total_score = 10

        context = EntryContext(
            symbol="ETHUSD",
            direction="long",
            entry_price=3000.0,
            entry_time=time.time(),
            signal_type="momentum",
            signal_strength=0.4,
            market_condition=MarketCondition.VOLATILE,
            indicators={},
            momentum=-0.5,
            volatility=0.03,
            spread=10.0,
            recent_prices=[],
            reason="test"
        )

        # 创建大亏损的反思
        reflection = ReflectionRecord(
            id="REF_TEST_BIG_LOSS",
            timestamp=time.time(),
            symbol="ETHUSD",
            reflection_type=ReflectionType.LOSS,
            entry_context=context,
            check_price=2500.0,
            pnl=-500.0,
            pnl_percent=-0.167,  # 16.7%亏损
            hold_duration=300,
            analysis="大亏损",
            lessons=[],
            score_change=0,
            suggestions=[]
        )

        engine._update_score(reflection)

        # 评分不应该变成负数
        assert engine.score.total_score >= 0

    def test_empty_market_data(self, engine):
        """空市场数据"""
        result = engine._analyze_market_condition({})
        # 空数据时volatility=0，被判断为CALM（低波动）
        assert result == MarketCondition.CALM

    def test_applicable_lessons_empty(self, engine):
        """无适用教训"""
        lessons = engine.get_applicable_lessons("trending_up", "momentum")
        assert isinstance(lessons, list)


class TestConcurrency:
    """并发测试"""

    def test_multiple_position_registration(self):
        """多持仓注册"""
        temp = tempfile.mkdtemp()
        try:
            engine = EvolutionEngine(data_dir=temp)

            signal_info = {"type": "momentum", "strength": 0.7, "reason": "test"}
            market_data = {"volatility": 0.01, "trend_strength": 0.6, "momentum": 0.3,
                          "spread": 2.0, "indicators": {}, "recent_prices": []}

            # 注册多个持仓
            for symbol in ["ETHUSD", "XAUUSD", "USOIL"]:
                engine.register_position(symbol, "long", 1000.0, time.time(),
                                        signal_info, market_data)

            assert len(engine._pending_checks) == 3
            assert len(engine._check_timers) == 3

            engine.shutdown()
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_shutdown_cancels_timers(self):
        """关闭取消定时器"""
        temp = tempfile.mkdtemp()
        try:
            engine = EvolutionEngine(data_dir=temp)

            signal_info = {"type": "momentum", "strength": 0.7, "reason": "test"}
            market_data = {"volatility": 0.01, "trend_strength": 0.6, "momentum": 0.3,
                          "spread": 2.0, "indicators": {}, "recent_prices": []}

            engine.register_position("ETHUSD", "long", 1000.0, time.time(),
                                    signal_info, market_data)

            engine.shutdown()

            assert len(engine._check_timers) == 0
        finally:
            shutil.rmtree(temp, ignore_errors=True)
