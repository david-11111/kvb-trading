"""
WebSocket 数据获取器

通过 WebSocket 直接连接 KVB 服务器获取实时行情
比网页读取更稳定、更快速、更轻量
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

try:
    import websockets
except ImportError:
    print("请安装 websockets: pip install websockets")
    websockets = None


@dataclass
class TickData:
    """行情数据"""
    symbol: str                      # 品种代码
    bid: float                       # 卖价
    ask: float                       # 买价
    spread: float                    # 点差
    timestamp: float                 # 时间戳

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class SymbolInfo:
    """品种信息"""
    symbol: str
    name: str = ""
    digits: int = 2                  # 小数位数
    pip_value: float = 0.01          # 点值


class WSDataFetcher:
    """
    WebSocket 数据获取器

    功能：
    1. 连接 KVB WebSocket 服务器
    2. 订阅多个品种的实时行情
    3. 接收并解析行情数据
    4. 提供回调接口
    """

    WS_URL = "wss://ws.mykvb.com/api/trade-api/v1/ws"

    def __init__(self, symbols: List[str] = None):
        """
        初始化

        Args:
            symbols: 要订阅的品种列表
        """
        self.symbols = symbols or ["USOIL", "XAUUSD", "ETHUSD"]
        self.ws = None
        self.is_connected = False

        # 行情数据缓存
        self.ticks: Dict[str, TickData] = {}

        # 价格历史（用于计算波动率）
        self.price_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.max_history = 100

        # 回调函数
        self.on_tick: Optional[Callable[[TickData], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None

        self.logger = logging.getLogger("WSDataFetcher")

    async def connect(self, token: str = None) -> bool:
        """
        连接 WebSocket

        Args:
            token: 认证 token（从浏览器登录态获取）
        """
        if websockets is None:
            self.logger.error("websockets 模块未安装")
            return False

        try:
            self.logger.info(f"正在连接 WebSocket: {self.WS_URL}")

            # 构建连接头
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            self.ws = await websockets.connect(
                self.WS_URL,
                extra_headers=headers if headers else None,
                ping_interval=30,
                ping_timeout=10,
            )

            self.is_connected = True
            self.logger.info("WebSocket 连接成功!")

            if self.on_connect:
                self.on_connect()

            return True

        except Exception as e:
            self.logger.error(f"WebSocket 连接失败: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """断开连接"""
        if self.ws:
            await self.ws.close()
            self.ws = None
        self.is_connected = False
        self.logger.info("WebSocket 已断开")

        if self.on_disconnect:
            self.on_disconnect()

    async def subscribe(self, symbols: List[str] = None):
        """
        订阅品种行情

        Args:
            symbols: 要订阅的品种列表
        """
        if not self.is_connected or not self.ws:
            self.logger.error("未连接到 WebSocket")
            return

        symbols = symbols or self.symbols

        # KVB 订阅消息格式（需要根据实际协议调整）
        subscribe_msg = {
            "type": "subscribe",
            "channels": ["quotes"],
            "symbols": symbols,
        }

        try:
            await self.ws.send(json.dumps(subscribe_msg))
            self.logger.info(f"已订阅品种: {symbols}")
        except Exception as e:
            self.logger.error(f"订阅失败: {e}")

    async def listen(self):
        """
        监听并处理消息
        """
        if not self.is_connected or not self.ws:
            self.logger.error("未连接到 WebSocket")
            return

        try:
            async for message in self.ws:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.warning(f"WebSocket 连接已关闭: {e}")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"监听异常: {e}")

    async def _handle_message(self, message: str):
        """处理接收到的消息"""
        try:
            data = json.loads(message)

            # 根据消息类型处理（需要根据实际协议调整）
            msg_type = data.get("type", "")

            if msg_type == "quote" or "symbol" in data:
                await self._handle_quote(data)
            elif msg_type == "ping":
                await self._send_pong()
            else:
                self.logger.debug(f"未知消息类型: {data}")

        except json.JSONDecodeError:
            self.logger.debug(f"非 JSON 消息: {message[:100]}")
        except Exception as e:
            self.logger.error(f"处理消息异常: {e}")

    async def _handle_quote(self, data: dict):
        """处理行情数据"""
        symbol = data.get("symbol", data.get("s", ""))

        if not symbol or symbol not in self.symbols:
            return

        bid = float(data.get("bid", data.get("b", 0)))
        ask = float(data.get("ask", data.get("a", 0)))

        if bid <= 0 or ask <= 0:
            return

        tick = TickData(
            symbol=symbol,
            bid=bid,
            ask=ask,
            spread=ask - bid,
            timestamp=time.time(),
        )

        self.ticks[symbol] = tick

        # 更新价格历史
        self.price_history[symbol].append(tick.mid_price)
        if len(self.price_history[symbol]) > self.max_history:
            self.price_history[symbol].pop(0)

        # 触发回调
        if self.on_tick:
            self.on_tick(tick)

    async def _send_pong(self):
        """发送心跳响应"""
        if self.ws:
            await self.ws.send(json.dumps({"type": "pong"}))

    def get_tick(self, symbol: str) -> Optional[TickData]:
        """获取指定品种的最新行情"""
        return self.ticks.get(symbol)

    def get_all_ticks(self) -> Dict[str, TickData]:
        """获取所有品种的最新行情"""
        return self.ticks.copy()

    def get_volatility(self, symbol: str) -> float:
        """
        计算品种波动率

        Returns:
            波动率(%)
        """
        history = self.price_history.get(symbol, [])
        if len(history) < 2:
            return 0.0

        import numpy as np
        prices = np.array(history)
        returns = np.diff(prices) / prices[:-1] * 100
        return float(np.std(returns))


class WSMultiMonitor:
    """
    基于 WebSocket 的多品种监控器
    """

    def __init__(self, symbols: List[str] = None, check_interval: float = 1.0):
        self.symbols = symbols or ["USOIL", "XAUUSD", "ETHUSD"]
        self.check_interval = check_interval

        self.fetcher = WSDataFetcher(self.symbols)
        self._is_running = False

        self.logger = logging.getLogger("WSMultiMonitor")

    async def start(self, token: str = None) -> bool:
        """启动监控"""
        connected = await self.fetcher.connect(token)
        if not connected:
            return False

        await self.fetcher.subscribe()
        self._is_running = True
        return True

    async def stop(self):
        """停止监控"""
        self._is_running = False
        await self.fetcher.disconnect()

    async def run(self, duration: int = None):
        """
        运行监控

        Args:
            duration: 运行时长（秒），None 表示无限
        """
        self.logger.info("启动 WebSocket 多品种监控...")

        # 启动监听任务
        listen_task = asyncio.create_task(self.fetcher.listen())

        # 启动状态打印任务
        print_task = asyncio.create_task(self._print_loop(duration))

        try:
            await asyncio.gather(listen_task, print_task)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _print_loop(self, duration: int = None):
        """定期打印状态"""
        start_time = time.time()

        while self._is_running:
            await asyncio.sleep(self.check_interval)

            if duration and (time.time() - start_time) >= duration:
                self._is_running = False
                break

            self._print_status()

    def _print_status(self):
        """打印当前状态"""
        print("\n" + "="*70)
        print(f"WebSocket 多品种监控 - {datetime.now().strftime('%H:%M:%S')}")
        print("="*70)
        print(f"{'品种':<10} {'买价':<12} {'卖价':<12} {'点差':<10} {'波动率%':<10}")
        print("-"*70)

        volatilities = []

        for symbol in self.symbols:
            tick = self.fetcher.get_tick(symbol)
            vol = self.fetcher.get_volatility(symbol)

            if tick:
                bid = f"{tick.bid:.4f}"
                ask = f"{tick.ask:.4f}"
                spread = f"{tick.spread:.4f}"
                vol_str = f"{vol:.4f}"
                volatilities.append((symbol, vol))
            else:
                bid = ask = spread = vol_str = "等待数据..."

            print(f"{symbol:<10} {bid:<12} {ask:<12} {spread:<10} {vol_str:<10}")

        print("-"*70)

        # 排名
        if volatilities:
            volatilities.sort(key=lambda x: x[1], reverse=True)
            best = volatilities[0]
            print(f">>> 波动率最高: {best[0]} ({best[1]:.4f}%)")

        print("="*70)


# ==================== 从浏览器提取 Token ====================

async def get_token_from_browser() -> Optional[str]:
    """
    从浏览器登录态中提取认证 token
    """
    from data_fetcher import DataFetcher

    fetcher = DataFetcher(headless=True)
    connected = await fetcher.connect()

    if not connected:
        return None

    try:
        page = fetcher.page

        # 尝试从 localStorage 获取 token
        token = await page.evaluate("""
            () => {
                return localStorage.getItem('token') ||
                       localStorage.getItem('access_token') ||
                       sessionStorage.getItem('token') ||
                       sessionStorage.getItem('access_token');
            }
        """)

        if token:
            return token

        # 尝试从 cookies 获取
        cookies = await page.context.cookies()
        for cookie in cookies:
            if 'token' in cookie['name'].lower():
                return cookie['value']

        return None

    finally:
        await fetcher.disconnect()


# ==================== 主函数 ====================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket 多品种监控")
    parser.add_argument(
        "--symbols",
        type=str,
        default="USOIL,XAUUSD,ETHUSD",
        help="要监控的品种，逗号分隔",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="监控时长（秒），0 表示无限",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
    )

    symbols = [s.strip() for s in args.symbols.split(",")]
    duration = args.duration if args.duration > 0 else None

    # 尝试获取 token
    print("正在从浏览器获取认证信息...")
    token = await get_token_from_browser()

    if token:
        print(f"获取到 token: {token[:20]}...")
    else:
        print("未获取到 token，尝试无认证连接...")

    # 启动监控
    monitor = WSMultiMonitor(symbols)

    if await monitor.start(token):
        await monitor.run(duration)
    else:
        print("启动失败，请检查网络连接")


if __name__ == "__main__":
    asyncio.run(main())
