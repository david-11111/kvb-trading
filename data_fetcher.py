"""
数据获取模块
使用 Playwright 自动化获取网页数据
"""

import asyncio
import logging
import json
import time
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

from config import WEB_CONFIG, LOGIN_CONFIG


@dataclass
class PriceData:
    """价格数据"""
    current_price: float         # 当前价格
    high: float                  # 最高价
    low: float                   # 最低价
    open: float                  # 开盘价
    close: float                 # 收盘价（当前）
    timestamp: float             # 时间戳
    volume: Optional[float] = None  # 成交量


@dataclass
class AccountInfo:
    """账户信息"""
    balance: float               # 账户余额
    equity: float                # 权益
    margin: float                # 保证金
    free_margin: float           # 可用保证金
    profit: float                # 浮动盈亏
    positions: List[Dict] = field(default_factory=list)  # 持仓列表


@dataclass
class MarketData:
    """市场数据（含历史K线）"""
    current: PriceData           # 当前价格
    history: List[PriceData]     # 历史K线数据
    account: Optional[AccountInfo] = None  # 账户信息


class DataFetcher:
    """
    数据获取器

    通过 Playwright 控制浏览器，从网页获取：
    1. 实时价格
    2. 历史K线数据（用于计算指标）
    3. 账户信息
    """

    def __init__(self, url: str = None, headless: bool = None):
        """
        初始化数据获取器

        Args:
            url: 目标网址
            headless: 是否无头模式
        """
        self.url = url or WEB_CONFIG["url"]
        self.headless = headless if headless is not None else WEB_CONFIG["headless"]
        self.browser = None
        self.page = None
        self.context = None
        self._playwright = None
        self.last_lot_value: Optional[str] = None
        self.is_connected = False
        self._attached_over_cdp = False

        # 数据缓存
        self._price_history: List[PriceData] = []
        self._max_history = 500  # 最多保存500条历史数据

        # 回调函数
        self._on_price_update: Optional[Callable] = None

        self.logger = logging.getLogger("DataFetcher")

        # 元素选择器配置 - 针对 mykvb.com/trade 页面
        # 基于实际页面结构分析得出的选择器
        self.selectors = {
            # OHLC价格 - .top-price-num 按顺序为 O, H, L, C
            "ohlc_prices": ".top-price-num",              # 获取所有OHLC价格
            "price_change": ".top-price-chg",             # 价格变动
            # 交易按钮
            "buy_button": "button.price-btn-bid",         # 绿色买入按钮
            "sell_button": "button.price-btn-ask",        # 红色卖出按钮
            "spread": ".price-btn-spread",                # 点差
            # 品种信息
            "symbol": ".kline-toolbar-symbol",            # 当前交易品种
            # 账户信息
            "account_labels": "span.label",               # 标签
            "account_values": "span.value",               # 数值
        }

    async def ensure_positions_tab(self) -> bool:
        """
        Best-effort: ensure the bottom holdings table (“持仓”) is visible.
        This improves reliability for close/sync operations.
        """
        if not self.is_connected or not self.page:
            return False

        candidates = [
            "text=持仓",
            "button:has-text('持仓')",
            "[role='tab']:has-text('持仓')",
            "text=Positions",
            "button:has-text('Positions')",
            "[role='tab']:has-text('Positions')",
        ]
        try:
            clicked = False
            for sel in candidates:
                try:
                    el = await self.page.query_selector(sel)
                    if el and await el.is_visible():
                        try:
                            await el.click(timeout=1000)
                        except Exception:
                            await el.click()
                        await asyncio.sleep(0.3)
                        clicked = True
                        break
                except Exception:
                    continue
            # If a table exists with rows beyond header, consider it “ready”.
            for rows_sel in ["tbody tr", "table tr", "[role='row']"]:
                try:
                    rows = await self.page.query_selector_all(rows_sel)
                    if rows and len(rows) >= 2:
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return clicked

    async def _scroll_row_actions_into_view(self, row) -> dict:
        """
        Best-effort: hover the row and scroll the nearest horizontal scroll container to the far right.
        Returns a small dict for debugging.
        """
        if not self.page:
            return {"ok": False, "reason": "no_page"}

        info: dict = {"ok": True, "hovered": False, "h_scrolled": False, "container": None}
        try:
            try:
                await row.scroll_into_view_if_needed(timeout=1200)
            except Exception:
                pass

            # Hover (some UIs only show the trailing X on hover).
            try:
                await row.hover(timeout=1200)
                info["hovered"] = True
            except Exception:
                pass

            # Choose the "best" horizontal scroll container among ancestors (largest scrollable distance).
            res = await row.evaluate(
                """
                (row) => {
                  function isScrollableX(el) {
                    try {
                      const st = window.getComputedStyle(el);
                      const ox = (st.overflowX || '').toLowerCase();
                      if (!(ox === 'auto' || ox === 'scroll')) return false;
                      return (el.scrollWidth - el.clientWidth) > 5;
                    } catch (e) { return false; }
                  }
                  const candidates = [];
                  let cur = row;
                  while (cur) {
                    if (isScrollableX(cur)) {
                      candidates.push({
                        el: cur,
                        tag: cur.tagName,
                        cls: (cur.className||'')+'',
                        dist: (cur.scrollWidth - cur.clientWidth),
                        w: cur.clientWidth,
                        h: cur.clientHeight,
                      });
                    }
                    cur = cur.parentElement;
                  }
                  // also scan a small neighborhood for nested scroll containers (common in complex tables)
                  const root = row.closest('div') || row.parentElement;
                  if (root) {
                    const divs = root.querySelectorAll('div');
                    for (const d of divs) {
                      if (!isScrollableX(d)) continue;
                      candidates.push({
                        el: d,
                        tag: d.tagName,
                        cls: (d.className||'')+'',
                        dist: (d.scrollWidth - d.clientWidth),
                        w: d.clientWidth,
                        h: d.clientHeight,
                      });
                    }
                  }
                  if (!candidates.length) return { scrolled: false, reason: 'no_scrollable_x' };
                  candidates.sort((a,b) => (b.dist - a.dist) || (b.w - a.w));
                  const best = candidates[0];
                  try { best.el.scrollLeft = best.el.scrollWidth; } catch(e) {}
                  return { scrolled: true, best: {tag: best.tag, cls: best.cls, dist: best.dist, w: best.w, h: best.h}, candidates: candidates.slice(0,5).map(c => ({tag:c.tag, dist:c.dist, w:c.w, h:c.h, cls:c.cls})) };
                }
                """
            )
            info["h_scrolled"] = bool(res.get("scrolled"))
            info["container"] = res

            # Hover again after scroll (some UIs re-render / lose hover state while scrolling).
            try:
                await row.hover(timeout=1200)
                info["hovered"] = True
            except Exception:
                pass
        except Exception as e:
            info["ok"] = False
            info["error"] = str(e)
        return info

    async def _click_close_x_in_row(self, row) -> dict:
        """
        Click the trailing “X/×” close button inside a holdings row.
        Returns a dict describing what happened.
        """
        if not self.page:
            return {"clicked": False, "reason": "no_page"}

        selectors = [
            "button:has-text('×')",
            "button:has-text('x')",
            "button:has-text('X')",
            "[role='button']:has-text('×')",
            "[class*='icon-x']",
            "[class*='icon-close']",
            "[class*='close-icon']",
            "[class*='close']",
            "svg[class*='close']",
            "i.close",
            "span:has-text('×')",
        ]
        for sel in selectors:
            try:
                loc = row.locator(sel).first
                if await loc.count() > 0 and await loc.is_visible():
                    try:
                        await loc.click(force=True, timeout=1200)
                    except Exception:
                        await loc.click(force=True)
                    return {"clicked": True, "method": "row_selector", "selector": sel}
            except Exception:
                continue

        try:
            res = await row.evaluate(
                """
                (row) => {
                  const rect = row.getBoundingClientRect();
                  const y = rect.top + rect.height / 2;
                  const xs = [rect.right - 8, rect.right - 16, rect.right - 26, rect.right - 40, rect.right - 55];
                  const hints = ['平仓', '关闭', 'close', 'delete', 'icon-x', 'icon-close'];

                  function describe(el) {
                    if (!el) return null;
                    const txt = ((el.innerText || el.textContent || '') + '').trim();
                    const cls = (el.className || '') + '';
                    const aria = el.getAttribute ? (el.getAttribute('aria-label') || '') : '';
                    const title = el.getAttribute ? (el.getAttribute('title') || '') : '';
                    return { tag: el.tagName, txt, cls, aria, title };
                  }

                  for (const x of xs) {
                    const el0 = document.elementFromPoint(x, y);
                    if (!el0) continue;
                    const inRow = row.contains(el0) || (el0.closest && el0.closest('tr') === row);
                    if (!inRow) continue;
                    const target = (el0.closest && el0.closest('button,[role=button],a,[tabindex]')) || el0;
                    const d = describe(target) || {};
                    const hay = ((d.txt||'') + ' ' + (d.cls||'') + ' ' + (d.aria||'') + ' ' + (d.title||'')).toLowerCase();

                    const hasHint =
                      (d.txt === '×' || d.txt === 'x' || d.txt === 'X') ||
                      hints.some(h => hay.indexOf(h) !== -1);

                    const tag = (d.tag || '').toUpperCase();
                    const allowByTag = (tag === 'SVG' || tag === 'PATH' || tag === 'I' || tag === 'SPAN');

                    if (hasHint || allowByTag) {
                      try { target.click(); } catch (e) {}
                      return { clicked: true, method: 'elementFromPoint', x, y, target: d, raw: describe(el0) };
                    }
                  }
                  return { clicked: false, method: 'elementFromPoint' };
                }
                """
            )
            if res and res.get("clicked"):
                return res
        except Exception as e:
            return {"clicked": False, "method": "elementFromPoint", "error": str(e)}

        return {"clicked": False, "reason": "no_close_x_found"}

    async def connect(self):
        """连接到网页"""
        from playwright.async_api import async_playwright
        import os

        max_attempts = int(WEB_CONFIG.get("connect_retries", 3) or 3)
        sleep_sec = float(WEB_CONFIG.get("connect_retry_sleep_sec", 2) or 2)
        last_err: Exception | None = None
        cdp_url = os.environ.get("SMART_TRADING_CDP_URL") or (WEB_CONFIG.get("cdp_url") or "")

        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"正在连接到 {self.url} (尝试 {attempt}/{max_attempts})")

                self._playwright = await async_playwright().start()

                if cdp_url:
                    # 观察者模式：附着到你已经打开的 Quark/Chromium（不再启动新浏览器）
                    self._attached_over_cdp = True
                    browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
                    self.browser = browser
                    contexts = browser.contexts
                    self.context = contexts[0] if contexts else await browser.new_context()

                    # 复用已有页面：优先找已打开的 trade 页面，否则新开一个
                    pages = list(self.context.pages)
                    trade_page = None
                    for p in pages:
                        try:
                            if "mykvb.com/trade" in (p.url or ""):
                                trade_page = p
                                break
                        except Exception:
                            continue
                    if trade_page:
                        self.page = trade_page
                    else:
                        self.page = await self.context.new_page()
                        await self.page.goto(self.url, wait_until="domcontentloaded")
                else:
                    # 启动浏览器 - 使用持久化上下文以保持登录状态
                    env_profile_dir = os.environ.get("SMART_TRADING_PROFILE_DIR")
                    user_data_dir = env_profile_dir or self._default_user_data_dir()
                    browser_exe = os.environ.get("SMART_TRADING_BROWSER_EXE") or WEB_CONFIG.get("browser_executable_path")

                    launch_kwargs = {}
                    if browser_exe:
                        launch_kwargs["executable_path"] = browser_exe

                    context_kwargs = {}
                    try:
                        mobile_on = bool(WEB_CONFIG.get("emulate_mobile", False)) or (os.environ.get("SMART_TRADING_MOBILE") == "1")
                        if mobile_on:
                            # Use a common iPhone-like viewport/UA to trigger lighter UI.
                            context_kwargs.update({
                                "is_mobile": True,
                                "device_scale_factor": 2,
                                "has_touch": True,
                                "viewport": {"width": 390, "height": 844},
                                "user_agent": (
                                    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                                    "Mobile/15E148 Safari/604.1"
                                ),
                            })
                            self.logger.info("已启用手机端模拟(SMART_TRADING_MOBILE=1)")
                    except Exception:
                        pass

                    async def launch_with_profile(profile_dir: str):
                        return await self._playwright.chromium.launch_persistent_context(
                            user_data_dir=profile_dir,
                            headless=self.headless,
                            args=[
                                '--no-sandbox',
                                '--disable-setuid-sandbox',
                                '--disable-session-crashed-bubble',  # 禁用崩溃恢复提示
                                '--disable-infobars',                # 禁用信息栏
                                '--no-first-run',                    # 禁用首次运行提示
                                '--disable-restore-session-state',   # 禁用恢复会话状态
                                '--hide-crash-restore-bubble',       # 隐藏崩溃恢复气泡
                            ],
                            **launch_kwargs,
                            **context_kwargs,
                        )

                    try:
                        self.context = await launch_with_profile(user_data_dir)
                    except Exception as e:
                        if env_profile_dir:
                            self.logger.warning(
                                f"使用指定Profile启动失败，将回退到项目内Profile以继续运行。"
                                f"（若想用Quark登录态，请先关闭所有Quark窗口再重试）错误: {e}"
                            )
                            user_data_dir = self._default_user_data_dir()
                            self.context = await launch_with_profile(user_data_dir)
                        else:
                            raise

                    self.browser = self.context

                    # 获取或创建页面
                    pages = self.context.pages
                    if pages:
                        self.page = pages[0]
                    else:
                        self.page = await self.context.new_page()

                # 加速：阻止部分大资源加载（图片/字体/视频等）
                try:
                    block_types = set(WEB_CONFIG.get("block_resource_types") or [])
                    if block_types:
                        async def _route_handler(route, request):
                            try:
                                if request.resource_type in block_types:
                                    await route.abort()
                                    return
                            except Exception:
                                pass
                            await route.continue_()

                        await self.context.route("**/*", _route_handler)
                except Exception:
                    pass

                # 统一超时设置（香港/海外线路、挑战页会明显变慢）
                nav_timeout = int(WEB_CONFIG.get("nav_timeout_ms", 300000))
                ready_timeout = int(WEB_CONFIG.get("ready_timeout_ms", 240000))
                try:
                    self.page.set_default_navigation_timeout(nav_timeout)
                    self.page.set_default_timeout(ready_timeout)
                except Exception:
                    pass

                # 访问目标网址（CDP 模式若已在 trade 页面则不强跳）
                if not cdp_url:
                    last_nav_err: Exception | None = None

                    async def _safe_goto():
                        nonlocal last_nav_err
                        for nav_try in range(1, 4):
                            try:
                                if (not self.page) or (hasattr(self.page, "is_closed") and self.page.is_closed()):
                                    self.page = await self.context.new_page()
                                await self.page.goto(self.url, wait_until="domcontentloaded", timeout=nav_timeout)
                                return
                            except Exception as e:
                                last_nav_err = e
                                msg = str(e)
                                if nav_try == 1:
                                    self.logger.warning(f"页面导航警告: {e}")
                                # 挑战页/重定向时 frame 重建：新建 page 重试更稳
                                if ("Frame has been detached" in msg) or ("Target closed" in msg) or ("has been closed" in msg):
                                    try:
                                        if self.page and (not self.page.is_closed()):
                                            await self.page.close()
                                    except Exception:
                                        pass
                                    self.page = None
                                await asyncio.sleep(0.6)

                        raise last_nav_err if last_nav_err else RuntimeError("page.goto failed")

                    await _safe_goto()

                try:
                    # 用 attached（而非 visible）兼容动画/遮罩导致的"可见性"误判
                    await self.page.wait_for_selector(self.selectors["buy_button"], timeout=ready_timeout, state="attached")
                except Exception as e:
                    self.logger.warning(f"关键元素未就绪(可能仍在加载/挑战页/未登录): {e}")

                    # 检测是否在登录页面，如果是则尝试自动登录
                    if await self._is_login_page():
                        self.logger.info("检测到登录页面，尝试自动登录...")
                        login_success = await self._auto_login()
                        if login_success:
                            self.is_connected = True
                            self.logger.info("自动登录成功，连接完成")
                            return True
                        else:
                            self.logger.error("自动登录失败")
                            return False

                    # 非登录页但也没有交易按钮，保存诊断截图
                    try:
                        from pathlib import Path
                        out_dir = Path(os.path.dirname(__file__)) / "trade_data"
                        out_dir.mkdir(exist_ok=True)
                        path = out_dir / f"connect_not_ready_{int(time.time())}.png"
                        await self.page.screenshot(path=str(path), full_page=True)
                        self.logger.warning(f"已保存诊断截图: {path}")
                    except Exception as screenshot_err:
                        self.logger.warning(f"保存诊断截图失败: {screenshot_err}")
                    return False

                self.is_connected = True
                self.logger.info("连接成功")
                return True

            except Exception as e:
                last_err = e
                self.logger.error(f"连接失败(尝试 {attempt}/{max_attempts}): {e}")
                self.is_connected = False
                try:
                    await self.disconnect()
                except Exception:
                    pass
                try:
                    if getattr(self, "_playwright", None):
                        await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
                self._attached_over_cdp = False
                if attempt < max_attempts:
                    await asyncio.sleep(sleep_sec)

        if last_err:
            self.logger.error(f"连接失败: {last_err}")
        return False

    def _default_user_data_dir(self) -> str:
        """
        默认将 Playwright 持久化 profile 放在可写目录。

        说明：
        - Prefer keeping all project artifacts inside the repo by default.
        - You can override with env var SMART_TRADING_PROFILE_DIR.
        - If the repo location is not writable (Windows protected folders), set SMART_TRADING_PROFILE_DIR to a writable path.
        """
        import os

        return os.path.join(os.path.dirname(__file__), "browser_data")

    async def disconnect(self):
        """断开连接"""
        # CDP 附着模式下不要关闭用户自己的浏览器，只断开连接即可
        if self.browser and not self._attached_over_cdp:
            try:
                await self.browser.close()
            except Exception:
                pass
        self.browser = None
        self.context = None
        self.page = None
        self.is_connected = False
        self._attached_over_cdp = False
        if self._playwright:
            try:
                await self._playwright.stop()
            finally:
                self._playwright = None
            self.logger.info("已断开连接")

    async def fetch_price(self) -> Optional[PriceData]:
        """
        获取当前价格

        Returns:
            PriceData对象
        """
        if not self.is_connected or not self.page:
            self.logger.warning("未连接到网页")
            return None

        try:
            # 获取价格元素（需要根据实际网页结构调整选择器）
            price_data = await self._extract_price_from_page()

            if price_data:
                # 添加到历史记录
                self._price_history.append(price_data)
                if len(self._price_history) > self._max_history:
                    self._price_history.pop(0)

                # 触发回调
                if self._on_price_update:
                    self._on_price_update(price_data)

            return price_data

        except Exception as e:
            self.logger.error(f"获取价格失败: {e}")
            return None

    async def _extract_price_from_page(self) -> Optional[PriceData]:
        """从页面提取价格数据 - 适配KVB交易页面"""
        try:
            # 获取所有OHLC价格元素 (.top-price-num 按顺序为 O, H, L, C)
            price_elements = await self.page.query_selector_all(self.selectors["ohlc_prices"])

            if len(price_elements) < 4:
                self.logger.warning(f"OHLC价格元素不足，找到 {len(price_elements)} 个")
                return None

            # 提取OHLC数值
            ohlc = []
            for elem in price_elements[:4]:
                text = await elem.inner_text()
                cleaned = ''.join(c for c in text if c.isdigit() or c == '.' or c == '-')
                ohlc.append(float(cleaned) if cleaned else 0)

            open_price, high, low, close = ohlc

            return PriceData(
                current_price=close,
                high=high,
                low=low,
                open=open_price,
                close=close,
                timestamp=time.time()
            )

        except Exception as e:
            self.logger.error(f"提取价格数据失败: {e}")
            return None

    async def _get_element_value(self, selector: str) -> Optional[float]:
        """获取元素的数值"""
        try:
            element = await self.page.query_selector(selector)
            if element:
                text = await element.inner_text()
                # 清理文本并转换为数字
                cleaned = ''.join(c for c in text if c.isdigit() or c == '.' or c == '-')
                return float(cleaned) if cleaned else None
        except:
            pass
        return None

    async def fetch_account(self) -> Optional[AccountInfo]:
        """
        获取账户信息 - 适配KVB交易页面

        Returns:
            AccountInfo对象
        """
        if not self.is_connected or not self.page:
            return None

        try:
            # 使用精确选择器获取账户信息容器
            profit_items = await self.page.query_selector_all("li.profit__item")

            account_data = {}
            for item in profit_items:
                label_elem = await item.query_selector("span.label")
                value_elem = await item.query_selector("span.value")
                
                if label_elem and value_elem:
                    label_text = await label_elem.inner_text()
                    value_text = await value_elem.inner_text()
                    # 清理label（去除冒号）和数值
                    label_clean = label_text.strip().rstrip(':').rstrip('：')
                    cleaned = ''.join(c for c in value_text if c.isdigit() or c == '.' or c == '-')
                    account_data[label_clean] = float(cleaned) if cleaned else 0

            profit = account_data.get("未结算盈亏", 0)
            equity = account_data.get("净值", 0)
            free_margin = account_data.get("可用保证金", 0)

            self.logger.debug(f"账户数据: {account_data}")

            return AccountInfo(
                balance=free_margin,  # 使用可用保证金作为余额
                equity=equity,
                margin=equity - free_margin if equity > free_margin else 0,
                free_margin=free_margin,
                profit=profit,
                positions=[]
            )
        except Exception as e:
            self.logger.error(f"获取账户信息失败: {e}")
            return None

    async def get_open_positions(self) -> List[Dict]:
        """
        从平台获取当前持仓列表

        Returns:
            持仓列表，每个元素包含 symbol, direction, lot_size, entry_price, profit 等
        """
        if not self.is_connected or not self.page:
            return []

        positions = []
        seen_keys = set()

        try:
            await self.ensure_positions_tab()

            # 首先获取持仓数量（从"持仓(N)"标签）
            position_tab = await self.page.query_selector("text=持仓")
            if position_tab:
                tab_text = await position_tab.inner_text()
                self.logger.info(f"持仓标签: {tab_text}")

            # KVB平台的持仓表格行选择器
            # 基于截图分析：持仓区域是一个表格，每行包含品种、订单号、时间、方向等
            row_selectors = [
                # Prefer the FMUI table wrapper used by KVB positions table
                ".fmui-table-wrapper tbody tr",
                ".fmui-table-wrapper tr",
                # 表格行
                "table tr",
                "tbody tr",
                ".trade-table tr",
                "[class*='table'] tr",
                "[class*='list'] [class*='item']",
                "[class*='position'] [class*='row']",
                "[class*='order'] [class*='row']",
            ]

            for selector in row_selectors:
                try:
                    rows = await self.page.query_selector_all(selector)
                    if rows and len(rows) > 1:  # 跳过表头
                        self.logger.info(f"找到 {len(rows)} 行: {selector}")
                        for row in rows:
                            try:
                                text = await row.inner_text()
                                # Holdings rows have an order ticket like "#67159350"
                                if "#" not in text:
                                    continue
                                # 跳过表头行
                                if "品种" in text or "Symbol" in text or "订单" in text:
                                    continue
                                pos_info = self._parse_position_row(text)
                                if pos_info:
                                    # De-dup: prefer ticket if available, else normalize by a short fingerprint.
                                    ticket = str(pos_info.get("ticket") or "")
                                    key = ticket if ticket else f"{pos_info.get('symbol')}|{pos_info.get('direction')}|{pos_info.get('lot_size')}|{pos_info.get('open_time','')}"
                                    if key in seen_keys:
                                        continue
                                    seen_keys.add(key)
                                    positions.append(pos_info)
                                    self.logger.info(f"解析持仓: {pos_info}")
                            except Exception as e:
                                pass
                        if positions:
                            break
                except Exception as e:
                    pass

            # 如果上面没找到，尝试直接从页面获取包含持仓信息的文本
            if not positions:
                try:
                    # 获取整个页面文本并解析
                    body_text = await self.page.inner_text("body")
                    positions = self._parse_positions_from_text(body_text)
                except Exception as e:
                    self.logger.debug(f"从页面文本解析持仓失败: {e}")

        except Exception as e:
            self.logger.error(f"获取持仓列表失败: {e}")

        if positions:
            self.logger.info(f"共检测到 {len(positions)} 个持仓")
            # 保存到文件供仪表板读取
            self._save_platform_positions(positions)
        return positions

    def _save_platform_positions(self, positions: List[Dict]):
        """保存平台持仓到文件供仪表板读取"""
        try:
            from pathlib import Path
            data_dir = Path(__file__).parent / "trade_data"
            data_dir.mkdir(exist_ok=True)
            filepath = data_dir / "platform_positions.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(positions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.debug(f"保存平台持仓失败: {e}")

    def _parse_positions_from_text(self, text: str) -> List[Dict]:
        """从页面文本中解析所有持仓"""
        import re
        positions = []

        seen = set()

        def add_pos(p: Dict):
            ticket = str(p.get("ticket") or "")
            key = ticket if ticket else json.dumps(p, ensure_ascii=False, sort_keys=True)
            if key in seen:
                return
            seen.add(key)
            positions.append(p)

        # 匹配模式: 品种 订单号 日期时间 Buy/Sell 手数 开仓价 现价
        # 例如: XAUUSD #67150001 02/01/2026 20:15:09 Sell 0.01 4314.10 4331.66
        pattern = r'(ETHUSD|XAUUSD|USOIL)\s+#?(?P<ticket>\d+)\s+(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(Buy|Sell|买入|卖出)\s+(?P<lot>\d+\.?\d*)\s+(?P<entry>\d+\.?\d*)'

        for m in re.finditer(pattern, text, re.IGNORECASE):
            symbol = m.group(1).upper()
            direction_raw = m.group(4)
            direction = "long" if direction_raw.lower() in ("buy", "买入") else "short"
            lot = float(m.group("lot"))
            entry_price = float(m.group("entry"))
            add_pos({
                "symbol": symbol.upper(),
                "direction": direction,
                "lot_size": lot,
                "entry_price": entry_price,
                "ticket": m.group("ticket"),
                "open_time": f"{m.group('date')} {m.group('time')}",
            })

        # 也尝试更简单的模式
        if not positions:
            pattern2 = r'(ETHUSD|XAUUSD|USOIL)\s+(Buy|Sell|买入|卖出)\s+(\d+\.?\d*)\s+(\d+\.?\d*)'
            matches = re.findall(pattern2, text, re.IGNORECASE)
            for m in matches:
                symbol, direction, lot, entry_price = m
                direction = "long" if direction.lower() in ("buy", "买入") else "short"
                add_pos({
                    "symbol": symbol.upper(),
                    "direction": direction,
                    "lot_size": float(lot),
                    "entry_price": float(entry_price),
                })

        return positions

    def _parse_position_row(self, text: str) -> dict:
        """解析持仓行文本"""
        import re

        # 尝试匹配多种格式
        patterns = [
            # 格式0: ETHUSD #67159350 03/01/2026 08:22:47 Sell 0.28 3110.23 ...
            r'(?P<symbol>ETHUSD|XAUUSD|USOIL)\s+#?(?P<ticket>\d+)\s+(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<side>Buy|Sell|买入|卖出)\s+(?P<lot>\d+\.?\d*)\s+(?P<entry>\d+\.?\d*)',
            # 格式1: ETHUSD 买入 0.1 3110.50
            r'(ETHUSD|XAUUSD|USOIL)\s+(买入|卖出|BUY|SELL|多|空)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
            # 格式2: ETHUSD 买入 0.1 (没有价格)
            r'(ETHUSD|XAUUSD|USOIL)\s+(买入|卖出|BUY|SELL|多|空)\s+(\d+\.?\d*)',
            # 格式3: ETHUSD #123456 01/03/2026 12:00:00 Buy 0.1 3110.50
            r'(ETHUSD|XAUUSD|USOIL)\s+#?\d+.*?(Buy|Sell|买入|卖出)\s+(\d+\.?\d*)',
        ]

        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if i == 0:
                    symbol = match.group("symbol").upper()
                    ticket = match.group("ticket")
                    date = match.group("date")
                    t = match.group("time")
                    side = match.group("side").lower()
                    lot = float(match.group("lot"))
                    entry_price = float(match.group("entry"))
                    direction = "long" if side in ("buy", "买入") else "short"
                    return {
                        "symbol": symbol,
                        "direction": direction,
                        "lot_size": lot,
                        "entry_price": entry_price,
                        "ticket": ticket,
                        "open_time": f"{date} {t}",
                    }
                else:
                    symbol = match.group(1).upper()
                    dir_text = match.group(2).lower()
                    lot = float(match.group(3))
                    entry_price = float(match.group(4)) if len(match.groups()) >= 4 else 0

                    if dir_text in ("买入", "buy", "多"):
                        direction = "long"
                    else:
                        direction = "short"

                    result = {
                        "symbol": symbol,
                        "direction": direction,
                        "lot_size": lot,
                    }
                    if entry_price > 0:
                        result["entry_price"] = entry_price
                    return result

        return None

    def get_market_data(self) -> MarketData:
        """
        获取完整的市场数据（同步方法，用于指标计算）

        Returns:
            MarketData对象
        """
        if not self._price_history:
            # 返回模拟数据用于测试
            return self._get_mock_data()

        current = self._price_history[-1] if self._price_history else None

        return MarketData(
            current=current,
            history=self._price_history.copy()
        )

    def get_price_arrays(self) -> tuple:
        """
        获取用于指标计算的价格数组

        Returns:
            (high_array, low_array, close_array)
        """
        import numpy as np

        if not self._price_history:
            # 返回模拟数据
            mock = self._get_mock_data()
            history = mock.history
        else:
            history = self._price_history

        high = np.array([p.high for p in history])
        low = np.array([p.low for p in history])
        close = np.array([p.close for p in history])

        return high, low, close

    def _get_mock_data(self) -> MarketData:
        """生成模拟数据用于测试"""
        import numpy as np

        # 生成100条模拟K线数据
        np.random.seed(42)
        base_price = 100
        prices = []

        for i in range(100):
            change = np.random.randn() * 2  # 随机波动
            base_price += change

            high = base_price + abs(np.random.randn())
            low = base_price - abs(np.random.randn())
            open_price = base_price + np.random.randn() * 0.5
            close = base_price

            prices.append(PriceData(
                current_price=close,
                high=high,
                low=low,
                open=open_price,
                close=close,
                timestamp=time.time() - (100 - i) * 60  # 每分钟一条
            ))

        return MarketData(
            current=prices[-1],
            history=prices
        )

    def set_price_callback(self, callback: Callable[[PriceData], None]):
        """设置价格更新回调"""
        self._on_price_update = callback

    def update_selectors(self, selectors: Dict[str, str]):
        """
        更新元素选择器

        Args:
            selectors: 选择器字典
        """
        self.selectors.update(selectors)
        self.logger.info(f"选择器已更新: {list(selectors.keys())}")

    # ==================== 自动登录方法 ====================

    def _get_login_credentials(self) -> tuple:
        """
        获取登录凭证（优先从环境变量，其次从配置文件）

        Returns:
            (phone, password) 元组
        """
        import os

        phone = LOGIN_CONFIG.get("phone") or os.environ.get("SMART_TRADING_PHONE", "")
        password = LOGIN_CONFIG.get("password") or os.environ.get("SMART_TRADING_PASSWORD", "")

        return phone.strip(), password.strip()

    async def _is_login_page(self) -> bool:
        """
        检测当前页面是否是登录页面

        Returns:
            True 如果是登录页面
        """
        if not self.page:
            return False

        try:
            # 检测登录页面的特征元素
            login_indicators = [
                'text=登录',
                'button:has-text("完成")',
                'input[type="password"]',
                '.login-type',  # 验证码登录按钮
            ]

            # 检测交易页面的特征元素（如果存在则不是登录页）
            trade_indicators = [
                self.selectors["buy_button"],
                self.selectors["sell_button"],
            ]

            # 如果交易元素存在，说明已登录
            for sel in trade_indicators:
                elem = await self.page.query_selector(sel)
                if elem:
                    return False

            # 检查登录页面特征
            login_score = 0
            for sel in login_indicators:
                try:
                    elem = await self.page.query_selector(sel)
                    if elem:
                        login_score += 1
                except:
                    pass

            # 如果有2个以上登录特征，认为是登录页
            return login_score >= 2

        except Exception as e:
            self.logger.error(f"检测登录页面失败: {e}")
            return False

    async def _auto_login(self) -> bool:
        """
        执行自动登录

        Returns:
            True 如果登录成功
        """
        if not LOGIN_CONFIG.get("enabled", True):
            self.logger.info("自动登录已禁用")
            return False

        phone, password = self._get_login_credentials()

        if not phone or not password:
            self.logger.warning("未配置登录凭证，无法自动登录。请设置环境变量 SMART_TRADING_PHONE 和 SMART_TRADING_PASSWORD")
            return False

        self.logger.info(f"开始自动登录，手机号: {phone[:3]}****{phone[-4:]}")

        try:
            login_timeout = int(LOGIN_CONFIG.get("login_timeout_ms", 30000))
            wait_after_login = float(LOGIN_CONFIG.get("wait_after_login_sec", 5))

            # 等待页面加载完成
            await asyncio.sleep(2)

            # 查找手机号输入框（第一个 text 类型的 input）
            phone_inputs = await self.page.query_selector_all('input.fmui-input-reference')
            phone_input = None
            for inp in phone_inputs:
                inp_type = await inp.get_attribute('type')
                if inp_type == 'text' or inp_type is None:
                    phone_input = inp
                    break

            if not phone_input:
                # 尝试其他选择器
                phone_input = await self.page.query_selector('input[type="text"]')

            if not phone_input:
                self.logger.error("未找到手机号输入框")
                return False

            # 清空并输入手机号
            await phone_input.click()
            await phone_input.fill("")
            await phone_input.fill(phone)
            self.logger.info("已输入手机号")

            # 查找密码输入框
            password_input = await self.page.query_selector('input[type="password"]')
            if not password_input:
                self.logger.error("未找到密码输入框")
                return False

            # 输入密码
            await password_input.click()
            await password_input.fill(password)
            self.logger.info("已输入密码")

            # 查找并点击登录按钮
            login_btn = await self.page.query_selector('button:has-text("完成")')
            if not login_btn:
                login_btn = await self.page.query_selector('button:has-text("登录")')
            if not login_btn:
                login_btn = await self.page.query_selector('button.fmui-button-primary')

            if not login_btn:
                self.logger.error("未找到登录按钮")
                return False

            await login_btn.click()
            self.logger.info("已点击登录按钮，等待登录完成...")

            # 等待登录完成（检测交易按钮出现）
            try:
                await self.page.wait_for_selector(
                    self.selectors["buy_button"],
                    timeout=login_timeout,
                    state="attached"
                )
                self.logger.info("登录成功！已检测到交易页面")
                await asyncio.sleep(wait_after_login)
                return True
            except Exception as e:
                # 检查是否有错误提示
                error_msg = await self.page.query_selector('.fmui-toast, .error-message, .toast-message')
                if error_msg:
                    error_text = await error_msg.inner_text()
                    self.logger.error(f"登录失败，错误信息: {error_text}")
                else:
                    self.logger.error(f"登录超时或失败: {e}")

                # 保存失败截图
                try:
                    from pathlib import Path
                    out_dir = Path(__file__).parent / "trade_data"
                    out_dir.mkdir(exist_ok=True)
                    path = out_dir / f"login_failed_{int(time.time())}.png"
                    await self.page.screenshot(path=str(path), full_page=True)
                    self.logger.info(f"已保存登录失败截图: {path}")
                except:
                    pass

                return False

        except Exception as e:
            self.logger.error(f"自动登录过程出错: {e}")
            return False

    async def wait_for_element(self, selector: str, timeout: int = 10000) -> bool:
        """等待元素出现"""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except:
            return False

    async def screenshot(self, path: str = "screenshot.png"):
        """截图保存"""
        if self.page:
            await self.page.screenshot(path=path)
            self.logger.info(f"截图已保存: {path}")

    # ==================== 交易操作方法 ====================

    async def get_buy_price(self) -> Optional[float]:
        """获取买入价格"""
        try:
            btn = await self.page.query_selector(self.selectors["buy_button"])
            if btn:
                text = await btn.inner_text()
                # 提取价格数字 (格式: "买入 58.131")
                parts = text.split()
                if len(parts) >= 2:
                    return float(parts[1])
        except Exception as e:
            self.logger.error(f"获取买入价格失败: {e}")
        return None

    async def get_sell_price(self) -> Optional[float]:
        """获取卖出价格"""
        try:
            btn = await self.page.query_selector(self.selectors["sell_button"])
            if btn:
                text = await btn.inner_text()
                # 提取价格数字 (格式: "卖出 58.102")
                parts = text.split()
                if len(parts) >= 2:
                    return float(parts[1])
        except Exception as e:
            self.logger.error(f"获取卖出价格失败: {e}")
        return None

    async def get_spread(self) -> Optional[int]:
        """获取点差"""
        try:
            elem = await self.page.query_selector(self.selectors["spread"])
            if elem:
                text = await elem.inner_text()
                return int(text.strip())
        except Exception as e:
            self.logger.error(f"获取点差失败: {e}")
        return None

    async def get_symbol(self) -> Optional[str]:
        """获取当前交易品种"""
        import re
        # 已知的交易品种列表
        known_symbols = ["USOIL", "XAUUSD", "ETHUSD", "BTCUSD", "GBPUSD", "EURUSD"]
        try:
            elem = await self.page.query_selector(self.selectors["symbol"])
            if elem:
                text = await elem.inner_text()
                # 去除所有空白字符（空格、换行、制表符等）
                raw = re.sub(r'\s+', '', text).upper()
                # 检查是否包含已知品种
                for sym in known_symbols:
                    if sym in raw:
                        return sym
                # 如果没找到已知品种，返回清理后的文本
                return raw if raw else None
        except Exception as e:
            self.logger.error(f"获取交易品种失败: {e}")
        return None

    async def switch_symbol(self, symbol: str) -> bool:
        """
        切换交易品种

        Args:
            symbol: 品种代码，如 'XAUUSD', 'USOIL', 'ETHUSD' 等

        Returns:
            是否切换成功
        """
        if not self.is_connected or not self.page:
            self.logger.error("未连接到页面")
            return False

        try:
            # 方法1: 直接点击品种列表中的项目
            selector = f'.custom-list-item[data-item="{symbol}"]'
            item = await self.page.query_selector(selector)

            if item:
                await item.click()
                await asyncio.sleep(1)  # 等待页面更新
                current = await self.get_symbol()
                if current and symbol.upper() in current.upper():
                    self.logger.info(f"成功切换到品种: {symbol}")
                    return True

            # 方法2: 尝试通过搜索框搜索品种
            search_input = await self.page.query_selector('input[placeholder*="搜索"], input[placeholder*="Search"], .symbol-search input')
            if search_input:
                await search_input.fill(symbol)
                await asyncio.sleep(0.5)
                # 点击搜索结果
                result_item = await self.page.query_selector(f'.custom-list-item[data-item="{symbol}"], .search-result-item:has-text("{symbol}")')
                if result_item:
                    await result_item.click()
                    await asyncio.sleep(1)
                    self.logger.info(f"通过搜索切换到品种: {symbol}")
                    return True

            # 方法3: 在页面上查找包含品种名称的可点击元素
            symbol_elem = await self.page.query_selector(f'text="{symbol}"')
            if symbol_elem:
                await symbol_elem.click()
                await asyncio.sleep(1)
                self.logger.info(f"通过文本匹配切换到品种: {symbol}")
                return True

            self.logger.warning(f"未找到品种 {symbol}，可能需要先添加到自选列表")
            return False

        except Exception as e:
            self.logger.error(f"切换品种失败: {e}")
            return False

    async def click_buy(self) -> bool:
        """
        点击买入按钮
        
        Returns:
            是否点击成功
        """
        if not self.is_connected or not self.page:
            return False

        try:
            btn = await self.page.query_selector(self.selectors["buy_button"])
            if btn:
                await btn.click()
                self.logger.info("已点击买入按钮")
                return True
        except Exception as e:
            self.logger.error(f"点击买入按钮失败: {e}")
        return False

    async def click_sell(self) -> bool:
        """
        点击卖出按钮
        
        Returns:
            是否点击成功
        """
        if not self.is_connected or not self.page:
            return False

        try:
            btn = await self.page.query_selector(self.selectors["sell_button"])
            if btn:
                await btn.click()
                self.logger.info("已点击卖出按钮")
                return True
        except Exception as e:
            self.logger.error(f"点击卖出按钮失败: {e}")
        return False

    async def click_buy_from_list(self, symbol: str) -> bool:
        """
        尝试在自选列表中直接点击 Ask 单元格触发买入（避免切换品种）。
        """
        if not self.is_connected or not self.page:
            return False
        selector = f'.custom-list-item[data-item="{symbol}"] .item-ask, .custom-list-item[data-item="{symbol}"] .item-ask span'
        try:
            el = await self.page.query_selector(selector)
            if el:
                await el.click()
                self.logger.info(f"已从列表点击买入(Ask): {symbol}")
                return True
        except Exception as e:
            self.logger.error(f"列表买入点击失败({symbol}): {e}")
        return False

    async def click_sell_from_list(self, symbol: str) -> bool:
        """
        尝试在自选列表中直接点击 Bid 单元格触发卖出（避免切换品种）。
        """
        if not self.is_connected or not self.page:
            return False
        selector = f'.custom-list-item[data-item="{symbol}"] .item-bid, .custom-list-item[data-item="{symbol}"] .item-bid span'
        try:
            el = await self.page.query_selector(selector)
            if el:
                await el.click()
                self.logger.info(f"已从列表点击卖出(Bid): {symbol}")
                return True
        except Exception as e:
            self.logger.error(f"列表卖出点击失败({symbol}): {e}")
        return False

    async def close_position_by_symbol(self, symbol: str) -> bool:
        """
        平掉指定品种的一个持仓（通过点击持仓行上的平仓按钮）

        Args:
            symbol: 品种代码，如 ETHUSD, XAUUSD

        Returns:
            是否成功点击平仓按钮
        """
        if not self.is_connected or not self.page:
            return False

        try:
            await self.ensure_positions_tab()

            # 首选：严格动作链（hover + 横向滚到最右 + 点击行尾X）
            try:
                rows = self.page.locator("tr", has_text=symbol)
                count = await rows.count()

                # Prefer the row that is visually in the bottom positions area (largest y).
                candidates = []
                for idx in range(min(int(count), 20)):
                    row = rows.nth(idx)
                    try:
                        if not await row.is_visible():
                            continue
                        try:
                            txt = (await row.inner_text(timeout=1200)).strip()
                        except Exception:
                            txt = ""
                        # skip header-ish rows
                        if not txt or ("品种" in txt) or ("Symbol" in txt):
                            continue
                        try:
                            box = await row.bounding_box()
                        except Exception:
                            box = None
                        y = float(box.get("y")) if box and isinstance(box, dict) and box.get("y") is not None else -1.0
                        candidates.append((y, idx))
                    except Exception:
                        continue

                candidates.sort(reverse=True)
                for _, idx in candidates[:8]:
                    row = rows.nth(idx)
                    try:
                        for attempt in range(3):
                            reveal = await self._scroll_row_actions_into_view(row)
                            self.logger.debug(f"[平仓] reveal attempt={attempt+1} {reveal}")
                            await asyncio.sleep(0.1)
                            clicked = await self._click_close_x_in_row(row)
                            self.logger.info(f"[平仓] click attempt={attempt+1} {clicked}")
                            if clicked.get("clicked"):
                                await asyncio.sleep(0.3)
                                return True
                    except Exception:
                        continue
            except Exception:
                pass

            # 方法0: JavaScript注入 - 最可靠的方式
            js_code = f'''
            (function() {{
                var symbol = '{symbol}';
                var result = {{}};

                // 查找包含该品种的所有行
                var rows = document.querySelectorAll('tr');
                for (var i = 0; i < rows.length; i++) {{
                    var row = rows[i];
                    if (row.textContent.indexOf(symbol) === -1) continue;

                    // 找到包含symbol的行，尝试找平仓按钮
                    // 1. 找最后一个单元格(通常是操作列)
                    var cells = row.querySelectorAll('td');
                    if (cells.length > 0) {{
                        var lastCell = cells[cells.length - 1];
                        // 找所有可点击元素
                        var clickables = lastCell.querySelectorAll('*');
                        for (var j = 0; j < clickables.length; j++) {{
                            var el = clickables[j];
                            // 检查是否是关闭按钮(x, 平仓等)
                            var text = el.textContent.trim();
                            var cls = el.className || '';
                            if (text === '×' || text === 'x' || text === 'X' ||
                                text.indexOf('平仓') !== -1 || text.indexOf('关闭') !== -1 ||
                                cls.indexOf('close') !== -1 || cls.indexOf('delete') !== -1) {{
                                el.click();
                                return {{success: true, method: 'js_cell_button', text: text}};
                            }}
                        }}
                    }}

                    // 2. 在整行找关闭按钮
                    var allBtns = row.querySelectorAll('button, a, span, i, svg');
                    for (var k = 0; k < allBtns.length; k++) {{
                        var btn = allBtns[k];
                        var text = btn.textContent.trim();
                        var cls = btn.className || '';
                        if (text === '×' || text === 'x' || text === 'X' ||
                            cls.indexOf('close') !== -1 || cls.indexOf('delete') !== -1 ||
                            cls.indexOf('icon-x') !== -1) {{
                            btn.click();
                            return {{success: true, method: 'js_row_button', text: text, class: cls}};
                        }}
                    }}

                    // 3. 尝试右键点击触发上下文菜单
                    var event = new MouseEvent('contextmenu', {{
                        bubbles: true,
                        cancelable: true,
                        view: window,
                        clientX: row.getBoundingClientRect().x + 100,
                        clientY: row.getBoundingClientRect().y + 10
                    }});
                    row.dispatchEvent(event);
                    return {{success: 'context_menu_triggered', row_text: row.textContent.substring(0, 100)}};
                }}

                return {{success: false, error: 'no_matching_row', rows_checked: rows.length}};
            }})();
            '''

            result = await self.page.evaluate(js_code)
            self.logger.info(f"[平仓] JS注入结果: {result}")

            if result and result.get('success') == True:
                self.logger.info(f"[平仓] 成功通过JS点击 {symbol} 的平仓按钮")
                await asyncio.sleep(0.5)
                return True
            elif result and result.get('success') == 'context_menu_triggered':
                # 右键菜单已触发，等待并点击平仓选项
                await asyncio.sleep(0.3)
                menu_js = '''
                (function() {
                    var items = document.querySelectorAll('*');
                    for (var i = 0; i < items.length; i++) {
                        var el = items[i];
                        var text = el.textContent.trim();
                        var style = window.getComputedStyle(el);
                        if ((text === '平仓' || text === 'Close') && style.display !== 'none' && style.visibility !== 'hidden') {
                            el.click();
                            return {clicked: true, text: text};
                        }
                    }
                    return {clicked: false};
                })();
                '''
                menu_result = await self.page.evaluate(menu_js)
                self.logger.info(f"[平仓] 右键菜单点击: {menu_result}")
                if menu_result and menu_result.get('clicked'):
                    await asyncio.sleep(0.3)
                    return True

            # 如果JS方法失败，回退到Playwright选择器方法
            self.logger.info(f"[平仓] JS方法未成功，尝试Playwright选择器")

            # 首先尝试找到包含该品种的持仓行
            # KVB平台的持仓行通常包含品种名称和平仓按钮
            row_selectors = [
                f"tr:has-text('{symbol}')",
                f"[class*='position']:has-text('{symbol}')",
                f"[class*='order']:has-text('{symbol}')",
                f"div:has-text('{symbol}')",
            ]

            for row_selector in row_selectors:
                try:
                    rows = await self.page.query_selector_all(row_selector)
                    if not rows:
                        continue

                    for row in rows:
                        # 在这一行里找平仓按钮
                        # KVB平台：持仓行最右侧有x按钮，或右键菜单有"平仓"选项
                        close_selectors = [
                            # x按钮 - KVB持仓行右侧的关闭按钮
                            "[class*='close-icon']",
                            "[class*='icon-close']",
                            "[class*='icon-x']",
                            "i.close",
                            "i.x",
                            "span.x",
                            "div.x",
                            "svg[class*='close']",
                            # 标准按钮
                            "button:has-text('×')",
                            "button:has-text('X')",
                            "button:has-text('x')",
                            "button:has-text('平仓')",
                            "button:has-text('关闭')",
                            "button:has-text('Close')",
                            "[class*='close']",
                            ".close-btn",
                            ".btn-close",
                            "span:has-text('平仓')",
                            "a:has-text('平仓')",
                        ]

                        for close_sel in close_selectors:
                            try:
                                close_btn = await row.query_selector(close_sel)
                                if close_btn:
                                    await close_btn.click()
                                    self.logger.info(f"[平仓] 成功点击 {symbol} 的平仓按钮")
                                    await asyncio.sleep(0.5)
                                    return True
                            except:
                                pass
                except:
                    pass

            # 方法2：尝试右键点击持仓行，然后点击上下文菜单中的"平仓"
            self.logger.info(f"[平仓] 尝试右键菜单平仓 {symbol}")
            for row_selector in row_selectors:
                try:
                    rows = await self.page.query_selector_all(row_selector)
                    for row in rows:
                        # 右键点击
                        await row.click(button="right")
                        await asyncio.sleep(0.3)

                        # 查找上下文菜单中的平仓选项
                        context_selectors = [
                            "text=平仓",
                            "[class*='menu'] >> text=平仓",
                            "[class*='context'] >> text=平仓",
                            "[class*='dropdown'] >> text=平仓",
                            "li:has-text('平仓')",
                            "div:has-text('平仓'):visible",
                        ]
                        for ctx_sel in context_selectors:
                            try:
                                menu_item = await self.page.query_selector(ctx_sel)
                                if menu_item and await menu_item.is_visible():
                                    await menu_item.click()
                                    self.logger.info(f"[平仓] 通过右键菜单成功点击 {symbol} 的平仓选项")
                                    await asyncio.sleep(0.5)
                                    return True
                            except:
                                pass

                        # 点击空白处关闭菜单
                        await self.page.click("body", position={"x": 10, "y": 10})
                        await asyncio.sleep(0.2)
                except:
                    pass

            self.logger.warning(f"[平仓] 未找到 {symbol} 的平仓按钮")
            return False

        except Exception as e:
            self.logger.error(f"[平仓] {symbol} 平仓失败: {e}")
            return False

    async def close_position_by_ticket(self, ticket: str | int) -> bool:
        """
        Close a specific position by order ticket (e.g. 67159382).

        This is safer than close-by-symbol when the platform has multiple positions per symbol.
        It clicks the trailing X/× on the matching holdings row, and does NOT click the final confirm.
        """
        if not self.is_connected or not self.page:
            return False

        t = str(ticket).strip().lstrip("#")
        if not t:
            return False
        needle = f"#{t}"

        try:
            await self.ensure_positions_tab()

            # Prefer rows inside the FMUI holdings table wrapper.
            rows = self.page.locator(".fmui-table-wrapper tr", has_text=needle)
            try:
                count = await rows.count()
            except Exception:
                count = 0
            if count <= 0:
                # fallback: any <tr> containing the ticket text
                rows = self.page.locator("tr", has_text=needle)
                try:
                    count = await rows.count()
                except Exception:
                    count = 0
            if count <= 0:
                self.logger.warning(f"[平仓] 未找到订单 {needle} 的持仓行")
                return False

            # Pick the visually lowest matching row (y max).
            candidates = []
            for idx in range(min(int(count), 20)):
                row = rows.nth(idx)
                try:
                    if not await row.is_visible():
                        continue
                    try:
                        txt = (await row.inner_text(timeout=1200)).strip()
                    except Exception:
                        txt = ""
                    if not txt or ("品种" in txt) or ("Symbol" in txt):
                        continue
                    try:
                        box = await row.bounding_box()
                    except Exception:
                        box = None
                    y = float(box.get("y")) if box and isinstance(box, dict) and box.get("y") is not None else -1.0
                    candidates.append((y, idx))
                except Exception:
                    continue

            candidates.sort(reverse=True)
            for _, idx in candidates[:5]:
                row = rows.nth(idx)
                try:
                    for attempt in range(3):
                        reveal = await self._scroll_row_actions_into_view(row)
                        self.logger.debug(f"[平仓] ticket={needle} reveal attempt={attempt+1} {reveal}")
                        await asyncio.sleep(0.1)
                        clicked = await self._click_close_x_in_row(row)
                        self.logger.info(f"[平仓] ticket={needle} click attempt={attempt+1} {clicked}")
                        if clicked.get("clicked"):
                            await asyncio.sleep(0.3)
                            return True
                except Exception:
                    continue

            self.logger.warning(f"[平仓] 未能点击订单 {needle} 的行尾X")
            return False

        except Exception as e:
            self.logger.error(f"[平仓] ticket={needle} 平仓失败: {e}")
            return False

    async def close_all_positions(self) -> int:
        """
        尝试平掉所有持仓（通过点击持仓列表中的平仓按钮）

        Returns:
            平仓成功的数量
        """
        if not self.is_connected or not self.page:
            return 0

        closed_count = 0

        # 常见的持仓列表和平仓按钮选择器
        position_selectors = [
            # 尝试多种可能的选择器
            ".position-list .position-item .close-btn",
            ".positions-panel .position-row .btn-close",
            ".trade-positions .position .close-position",
            "[class*='position'] [class*='close']",
            ".order-list .order-item .close-btn",
            "button:has-text('平仓')",
            "button:has-text('关闭')",
            "span:has-text('平仓')",
            ".close-all-btn",
            "[data-action='close']",
        ]

        for selector in position_selectors:
            try:
                buttons = await self.page.query_selector_all(selector)
                if buttons:
                    self.logger.info(f"找到 {len(buttons)} 个平仓按钮: {selector}")
                    for btn in buttons:
                        try:
                            await btn.click()
                            closed_count += 1
                            await asyncio.sleep(0.5)
                            # 如果有确认对话框，尝试点击确认
                            confirm_selectors = [
                                "button:has-text('确认')",
                                "button:has-text('确定')",
                                ".confirm-btn",
                                "[class*='confirm']",
                            ]
                            for cs in confirm_selectors:
                                try:
                                    confirm = await self.page.query_selector(cs)
                                    if confirm:
                                        await confirm.click()
                                        await asyncio.sleep(0.3)
                                        break
                                except:
                                    pass
                        except Exception as e:
                            self.logger.error(f"点击平仓按钮失败: {e}")
                    break
            except Exception as e:
                self.logger.debug(f"选择器 {selector} 无结果: {e}")

        self.logger.info(f"平仓操作完成，共平 {closed_count} 个")
        return closed_count

    async def execute_trade(self, action: str) -> bool:
        """
        执行交易操作
        
        Args:
            action: 'buy' 或 'sell'
            
        Returns:
            是否执行成功
        """
        if action.lower() == 'buy':
            return await self.click_buy()
        elif action.lower() == 'sell':
            return await self.click_sell()
        else:
            self.logger.error(f"未知的交易操作: {action}")
            return False

    async def set_lot_size(self, lots: float) -> bool:
        """
        尝试设置下单手数/数量。

        说明：KVB 页面不同版本的输入框可能不同，此方法尽力而为；
        找不到输入框会返回 False（auto_trader 可配置为继续按页面默认手数下单）。
        """
        import asyncio

        if not self.is_connected or not self.page:
            return False

        text = f"{lots}".rstrip("0").rstrip(".") if lots is not None else ""
        if not text:
            return False

        self.logger.info(f"开始设置手数: {text}")

        async def try_set(locator, desc: str = "") -> bool:
            try:
                count = await asyncio.wait_for(locator.count(), timeout=2.0)
            except Exception:
                return False
            if count == 0:
                return False
            self.logger.debug(f"找到 {count} 个元素: {desc}")
            for idx in range(min(int(count), 3)):  # 减少到3个
                el = locator.nth(idx)
                try:
                    visible = await asyncio.wait_for(el.is_visible(), timeout=1.0)
                    if not visible:
                        continue
                    try:
                        await asyncio.wait_for(el.scroll_into_view_if_needed(), timeout=1.0)
                    except Exception:
                        pass
                    try:
                        await el.click(timeout=500)
                    except Exception:
                        try:
                            await el.click(force=True, timeout=500)
                        except Exception:
                            continue

                    # Prefer fill(), but fall back to keyboard for custom widgets.
                    try:
                        await asyncio.wait_for(el.fill(""), timeout=1.0)
                        await asyncio.wait_for(el.fill(text), timeout=1.0)
                    except Exception:
                        try:
                            await self.page.keyboard.press("Control+A")
                        except Exception:
                            pass
                        await self.page.keyboard.type(text, delay=10)

                    try:
                        await el.press("Enter", timeout=500)
                    except Exception:
                        pass

                    try:
                        value = await asyncio.wait_for(el.evaluate(
                            "(el) => (el.value ?? el.innerText ?? el.textContent ?? '').toString()"
                        ), timeout=1.0)
                    except Exception:
                        value = ""
                    value = (value or "").strip()
                    self.last_lot_value = value or text
                    self.logger.info(f"已尝试设置手数: {text} (读取到: {self.last_lot_value})")
                    return True
                except Exception as e:
                    self.logger.debug(f"设置元素 {idx} 失败: {e}")
                    continue
            return False

        scope_selectors = [
            # Try common “order/ticket” containers first to avoid hitting unrelated inputs.
            ".order-dialog, .order-modal, .trade-modal, .order-ticket, .trade-ticket, .order-panel, .trade-panel, .order-container",
            "body",
        ]
        input_selectors = [
            # Most specific first
            'input[placeholder*="手"]',
            'input[placeholder*="数量"]',
            'input[placeholder*="Lot" i]',
            'input[placeholder*="Size" i]',
            'input[aria-label*="手"]',
            'input[aria-label*="数量"]',
            'input[aria-label*="Lot" i]',
            'input[aria-label*="Size" i]',
            'input[name*="lot" i]',
            'input[id*="lot" i]',
            'input[name*="size" i]',
            'input[id*="size" i]',
            'input[name*="volume" i]',
            'input[id*="volume" i]',
            'input[name*="qty" i]',
            'input[id*="qty" i]',
            'input[type="number"]',
            'input[role="spinbutton"]',
            '[role="spinbutton"]',
            '[contenteditable="true"]',
            "input",
        ]
        label_patterns = [
            r"手数",
            r"数量",
            r"交易量",
            r"\bLot\b",
            r"\bSize\b",
            r"\bVolume\b",
        ]

        # 添加总超时保护
        start_time = asyncio.get_event_loop().time()
        max_time = 10.0  # 最多10秒

        for scope_sel in scope_selectors:
            if asyncio.get_event_loop().time() - start_time > max_time:
                self.logger.warning("设置手数超时")
                break

            scope = self.page.locator(scope_sel).first
            try:
                if scope_sel != "body" and (await asyncio.wait_for(scope.count(), timeout=2.0)) == 0:
                    continue
                self.logger.debug(f"正在搜索范围: {scope_sel[:30]}")
            except Exception:
                continue

            # 先用"标签附近"方式定位：更稳
            for pat in label_patterns:
                if asyncio.get_event_loop().time() - start_time > max_time:
                    break
                try:
                    label = scope.locator(f"text=/{pat}/i").first
                    if (await asyncio.wait_for(label.count(), timeout=1.0)) == 0:
                        continue
                    # input right after the label (best-effort)
                    if await try_set(label.locator("xpath=following::input[1]"), f"label:{pat}/input"):
                        return True
                    if await try_set(label.locator('xpath=following::*[@contenteditable="true"][1]'), f"label:{pat}/editable"):
                        return True
                    if await try_set(label.locator("xpath=following::*[@role='spinbutton'][1]"), f"label:{pat}/spinbutton"):
                        return True
                except Exception:
                    continue

            # 再退化为"常见输入框"扫描（只试前几个最可能的）
            for sel in input_selectors[:8]:  # 只试前8个选择器
                if asyncio.get_event_loop().time() - start_time > max_time:
                    break
                try:
                    if await try_set(scope.locator(sel), f"input:{sel[:20]}"):
                        return True
                except Exception:
                    continue

        self.logger.warning("未找到手数输入框，可能将使用页面默认手数")
        return False


class MockDataFetcher:
    """
    模拟数据获取器
    用于在网页不可用时进行测试
    """

    def __init__(self):
        self._price_history: List[PriceData] = []
        self._base_price = 100.0
        self._generate_initial_data()

    def _generate_initial_data(self):
        """生成初始历史数据"""
        import numpy as np
        np.random.seed(int(time.time()) % 1000)

        for i in range(100):
            self._base_price += np.random.randn() * 0.5

            high = self._base_price + abs(np.random.randn()) * 0.3
            low = self._base_price - abs(np.random.randn()) * 0.3

            self._price_history.append(PriceData(
                current_price=self._base_price,
                high=high,
                low=low,
                open=self._base_price + np.random.randn() * 0.1,
                close=self._base_price,
                timestamp=time.time() - (100 - i) * 60
            ))

    def tick(self) -> PriceData:
        """模拟价格跳动"""
        import numpy as np

        # 随机波动
        change = np.random.randn() * 0.3
        self._base_price += change

        high = self._base_price + abs(np.random.randn()) * 0.2
        low = self._base_price - abs(np.random.randn()) * 0.2

        price_data = PriceData(
            current_price=self._base_price,
            high=high,
            low=low,
            open=self._price_history[-1].close if self._price_history else self._base_price,
            close=self._base_price,
            timestamp=time.time()
        )

        self._price_history.append(price_data)
        if len(self._price_history) > 500:
            self._price_history.pop(0)

        return price_data

    def get_price_arrays(self) -> tuple:
        """获取价格数组"""
        import numpy as np

        high = np.array([p.high for p in self._price_history])
        low = np.array([p.low for p in self._price_history])
        close = np.array([p.close for p in self._price_history])

        return high, low, close

    def get_current_price(self) -> float:
        """获取当前价格"""
        return self._base_price
