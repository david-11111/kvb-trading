"""
智能交易平台 - 主程序入口

自动化交易流程：
1. 获取网页数据
2. 分析市场状态（大脑思考）
3. 生成交易决策
4. 执行交易
5. 风控监控
"""

import asyncio
import logging
import time
import sys
from datetime import datetime
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import WEB_CONFIG, LOG_CONFIG, RISK_CONFIG
from data_fetcher import DataFetcher, MockDataFetcher
from decision_engine import TradingBrain, DecisionResult
from trader import Trader
from risk_control import RiskManager


# 配置日志
def setup_logging():
    """配置日志系统"""
    log_format = '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s'
    log_file = Path("logs")
    log_file.mkdir(exist_ok=True)

    # 文件处理器
    file_handler = logging.FileHandler(
        log_file / f"trading_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("Main")


class SmartTradingPlatform:
    """
    智能交易平台

    整合所有模块，实现自动化交易
    """

    def __init__(self, initial_capital: float = 10000, use_mock: bool = True):
        """
        初始化交易平台

        Args:
            initial_capital: 初始资金
            use_mock: 是否使用模拟数据（网页不可用时）
        """
        self.initial_capital = initial_capital
        self.use_mock = use_mock

        # 初始化各模块
        self.brain = TradingBrain(initial_capital)
        self.trader = Trader(initial_capital=initial_capital)
        self.risk_manager = RiskManager(initial_capital)

        # 数据获取器
        if use_mock:
            self.data_fetcher = MockDataFetcher()
        else:
            self.data_fetcher = DataFetcher()

        # 运行状态
        self.is_running = False
        self.loop_interval = 5  # 循环间隔（秒）

        self.logger = logging.getLogger("Platform")

    async def start(self):
        """启动交易平台"""
        self.logger.info("=" * 60)
        self.logger.info("智能交易平台启动")
        self.logger.info(f"初始资金: {self.initial_capital}")
        self.logger.info(f"运行模式: {'模拟' if self.use_mock else '实盘'}")
        self.logger.info(f"风控阈值: {RISK_CONFIG['max_loss_percent']*100}%")
        self.logger.info("=" * 60)

        # 连接到网页（非模拟模式）
        if not self.use_mock:
            if isinstance(self.data_fetcher, DataFetcher):
                connected = await self.data_fetcher.connect()
                if not connected:
                    self.logger.error("无法连接到交易网页，切换到模拟模式")
                    self.use_mock = True
                    self.data_fetcher = MockDataFetcher()
                else:
                    self.trader.set_page(self.data_fetcher.page)

        self.is_running = True

        # 启动主循环
        await self._main_loop()

    async def stop(self):
        """停止交易平台"""
        self.is_running = False

        # 断开网页连接
        if not self.use_mock and isinstance(self.data_fetcher, DataFetcher):
            await self.data_fetcher.disconnect()

        # 打印交易统计
        self._print_summary()

        self.logger.info("交易平台已停止")

    async def _main_loop(self):
        """主交易循环"""
        self.logger.info("进入主交易循环...")

        tick_count = 0

        while self.is_running:
            try:
                tick_count += 1

                # 1. 获取最新数据
                if self.use_mock:
                    price_data = self.data_fetcher.tick()
                    current_price = price_data.current_price
                else:
                    price_data = await self.data_fetcher.fetch_price()
                    if price_data is None:
                        self.logger.warning("获取价格失败，跳过本次循环")
                        await asyncio.sleep(self.loop_interval)
                        continue
                    current_price = price_data.current_price

                # 获取价格数组
                high, low, close = self.data_fetcher.get_price_arrays()

                # 2. 大脑思考，生成决策
                decision = self.brain.think(high, low, close, current_price)

                # 3. 检查是否需要止盈
                if self.trader.current_position:
                    pos = self.trader.current_position
                    if self.brain.should_take_profit(current_price, pos.entry_price, pos.take_profit / pos.entry_price - 1 if pos.direction == "long" else 1 - pos.take_profit / pos.entry_price):
                        self.logger.info("触发止盈条件")
                        decision = await self._create_take_profit_decision()

                # 4. 执行决策
                await self._execute_decision(decision, current_price)

                # 5. 定期输出状态
                if tick_count % 20 == 0:
                    self._print_status(current_price)

                # 等待下一个周期
                await asyncio.sleep(self.loop_interval)

            except KeyboardInterrupt:
                self.logger.info("收到中断信号，正在停止...")
                break
            except Exception as e:
                self.logger.error(f"主循环异常: {e}")
                await asyncio.sleep(self.loop_interval)

        await self.stop()

    async def _execute_decision(self, decision, current_price: float):
        """执行交易决策"""
        if decision.result == DecisionResult.NO_ACTION:
            return

        if decision.result == DecisionResult.BUY:
            record = await self.trader.buy(
                price=current_price,
                quantity=1,
                take_profit_percent=decision.take_profit,
                position_size=decision.position_size
            )
            if record:
                self.brain.notify_trade_executed("long", current_price)
                self.risk_manager.open_position("long", current_price, record.quantity)

        elif decision.result == DecisionResult.SELL:
            record = await self.trader.sell(
                price=current_price,
                quantity=1,
                take_profit_percent=decision.take_profit,
                position_size=decision.position_size
            )
            if record:
                self.brain.notify_trade_executed("short", current_price)
                self.risk_manager.open_position("short", current_price, record.quantity)

        elif decision.result in [DecisionResult.CLOSE_LONG, DecisionResult.CLOSE_SHORT, DecisionResult.STOP_LOSS]:
            record = await self.trader.close_position(current_price)
            if record:
                self.brain.notify_trade_executed(None, current_price)
                self.risk_manager.close_position()

    async def _create_take_profit_decision(self):
        """创建止盈决策"""
        from decision_engine import Decision, DecisionResult
        pos = self.trader.current_position
        if pos.direction == "long":
            result = DecisionResult.CLOSE_LONG
        else:
            result = DecisionResult.CLOSE_SHORT

        return Decision(
            result=result,
            confidence=1.0,
            market_state="止盈",
            macd_signal="N/A",
            take_profit=0,
            position_size=0,
            reasoning="达到止盈目标，获利了结"
        )

    def _print_status(self, current_price: float):
        """打印当前状态"""
        summary = self.trader.get_trade_summary()
        position = self.trader.get_position_info()

        self.logger.info("-" * 40)
        self.logger.info(f"当前价格: {current_price:.4f}")
        self.logger.info(f"当前资金: {summary['current_capital']:.2f}")
        self.logger.info(f"收益率: {summary['return_percent']:.2f}%")
        if position:
            self.logger.info(f"持仓: {position['direction']} @ {position['entry_price']:.4f}")
            pnl, pnl_pct = self.risk_manager.get_pnl(current_price)
            self.logger.info(f"浮动盈亏: {pnl:.2f} ({pnl_pct*100:.2f}%)")
        self.logger.info("-" * 40)

    def _print_summary(self):
        """打印交易统计"""
        summary = self.trader.get_trade_summary()

        print("\n" + "=" * 60)
        print("交易统计摘要")
        print("=" * 60)
        print(f"初始资金: {self.initial_capital:.2f}")
        print(f"最终资金: {summary['current_capital']:.2f}")
        print(f"总收益: {summary['total_pnl']:.2f}")
        print(f"收益率: {summary['return_percent']:.2f}%")
        print(f"总交易次数: {summary['total_trades']}")
        print(f"盈利次数: {summary['winning_trades']}")
        print(f"亏损次数: {summary['losing_trades']}")
        print(f"胜率: {summary['win_rate']:.1f}%")
        print("=" * 60 + "\n")


async def main():
    """主函数"""
    logger = setup_logging()

    # 解析命令行参数
    use_mock = True  # 默认使用模拟模式（网页暂不可用）
    initial_capital = 10000

    if len(sys.argv) > 1:
        if "--real" in sys.argv:
            use_mock = False
        for arg in sys.argv[1:]:
            if arg.startswith("--capital="):
                initial_capital = float(arg.split("=")[1])

    # 创建并启动平台
    platform = SmartTradingPlatform(
        initial_capital=initial_capital,
        use_mock=use_mock
    )

    try:
        await platform.start()
    except KeyboardInterrupt:
        logger.info("收到退出信号")
    finally:
        await platform.stop()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    智能交易平台 v1.0                          ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  功能特性:                                                    ║
    ║  - MACD金叉/死叉信号检测                                     ║
    ║  - 投资哲学体系（震荡/单边市场判断）                         ║
    ║  - 自动化交易执行                                            ║
    ║  - 5%止损风控                                                ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  使用方法:                                                    ║
    ║  python main.py              # 模拟模式                       ║
    ║  python main.py --real       # 实盘模式（需网页可用）         ║
    ║  python main.py --capital=20000  # 设置初始资金               ║
    ║  按 Ctrl+C 停止运行                                          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    asyncio.run(main())
