"""
交易系统完全镜像仪表板 - 实时同步平台所有状态
支持嵌入KVB网站 + 远程平仓操作
"""

import asyncio
import json
import time
import os
from datetime import datetime
from pathlib import Path
from aiohttp import web
import aiohttp_cors

# 数据目录
DATA_DIR = Path(__file__).parent / "trade_data"
POSITIONS_FILE = DATA_DIR / "positions.json"
PLATFORM_POSITIONS_FILE = DATA_DIR / "platform_positions.json"
SCREENSHOT_FILE = DATA_DIR / "latest_screenshot.png"
COMMAND_FILE = DATA_DIR / "dashboard_commands.json"  # 命令文件


def load_system_positions():
    """加载系统追踪的持仓"""
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}


def load_platform_positions():
    """加载平台检测到的持仓"""
    if PLATFORM_POSITIONS_FILE.exists():
        try:
            with open(PLATFORM_POSITIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return []


def load_all_events(limit=500):
    """加载所有交易事件"""
    today = datetime.now().strftime("%Y%m%d")
    jsonl_file = DATA_DIR / f"trade_events_{today}.jsonl"

    events = []
    if jsonl_file.exists():
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except:
                        pass
        except:
            pass
    return events[-limit:] if len(events) > limit else events


def get_platform_label_count():
    """从最新日志获取平台标签显示的持仓数量"""
    import re
    today = datetime.now().strftime("%Y%m%d")
    jsonl_file = DATA_DIR / f"trade_events_{today}.jsonl"

    try:
        import glob
        log_patterns = [
            DATA_DIR.parent / "trading.log",
            Path("C:/Users/Lenovo/AppData/Local/Temp/claude") / "**/*.output",
        ]

        for pattern in log_patterns:
            if "*" in str(pattern):
                files = glob.glob(str(pattern), recursive=True)
                if files:
                    files.sort(key=os.path.getmtime, reverse=True)
                    for f in files[:3]:
                        try:
                            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                                content = fh.read()
                                matches = re.findall(r'持仓\s*\(?(\d+)\)?', content)
                                if matches:
                                    return int(matches[-1])
                        except:
                            pass
            elif pattern.exists():
                try:
                    with open(pattern, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                        matches = re.findall(r'持仓\s*\(?(\d+)\)?', content)
                        if matches:
                            return int(matches[-1])
                except:
                    pass
    except:
        pass

    return None


def categorize_events(events):
    """分类事件"""
    categorized = {
        "heartbeats": [],
        "opens": [],
        "closes": [],
        "adds": [],
        "signals": [],
        "errors": [],
        "others": [],
    }

    for event in events:
        kind = event.get("kind", "")
        if kind == "heartbeat":
            categorized["heartbeats"].append(event)
        elif kind == "open":
            categorized["opens"].append(event)
        elif kind == "close":
            categorized["closes"].append(event)
        elif kind == "add":
            categorized["adds"].append(event)
        elif kind == "signal":
            categorized["signals"].append(event)
        elif kind == "error":
            categorized["errors"].append(event)
        else:
            categorized["others"].append(event)

    return categorized


def send_command(cmd_type: str, **kwargs):
    """发送命令到auto_trader"""
    command = {
        "type": cmd_type,
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "status": "pending",
        **kwargs
    }

    # 读取现有命令
    commands = []
    if COMMAND_FILE.exists():
        try:
            with open(COMMAND_FILE, "r", encoding="utf-8") as f:
                commands = json.load(f)
        except:
            commands = []

    commands.append(command)

    # 只保留最近10条命令
    commands = commands[-10:]

    with open(COMMAND_FILE, "w", encoding="utf-8") as f:
        json.dump(commands, f, ensure_ascii=False, indent=2)

    return command


def get_commands():
    """获取所有命令"""
    if COMMAND_FILE.exists():
        try:
            with open(COMMAND_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return []


async def get_full_sync_data(request):
    """API: 获取完全同步数据"""
    system_positions = load_system_positions()
    platform_positions = load_platform_positions()
    all_events = load_all_events()
    categorized = categorize_events(all_events)

    platform_by_symbol = {}
    for pos in platform_positions:
        symbol = pos.get("symbol", "UNKNOWN")
        if symbol not in platform_by_symbol:
            platform_by_symbol[symbol] = {"long": [], "short": []}
        direction = pos.get("direction", "unknown")
        if direction in platform_by_symbol[symbol]:
            platform_by_symbol[symbol][direction].append(pos)

    tracked_symbols = set(system_positions.keys())
    orphan_count = len(platform_positions) - len(tracked_symbols)

    has_screenshot = SCREENSHOT_FILE.exists()
    screenshot_mtime = SCREENSHOT_FILE.stat().st_mtime if has_screenshot else None
    platform_label_count = get_platform_label_count()

    # 获取命令状态
    commands = get_commands()

    return web.json_response({
        "timestamp": datetime.now().isoformat(),
        "system_positions": system_positions,
        "platform_positions": platform_positions,
        "platform_count": len(platform_positions),
        "platform_label_count": platform_label_count,
        "system_count": len(system_positions),
        "orphan_count": max(0, orphan_count),
        "detection_mismatch": platform_label_count and platform_label_count != len(platform_positions),
        "platform_by_symbol": {
            k: {"long": len(v["long"]), "short": len(v["short"])}
            for k, v in platform_by_symbol.items()
        },
        "events": {
            "total": len(all_events),
            "opens": categorized["opens"][-50:],
            "closes": categorized["closes"][-50:],
            "adds": categorized["adds"][-50:],
            "signals": categorized["signals"][-100:],
            "errors": categorized["errors"][-20:],
            "heartbeats": categorized["heartbeats"][-10:],
        },
        "statistics": {
            "total_opens": len(categorized["opens"]),
            "total_closes": len(categorized["closes"]),
            "total_adds": len(categorized["adds"]),
            "total_signals": len(categorized["signals"]),
            "total_errors": len(categorized["errors"]),
        },
        "has_screenshot": has_screenshot,
        "screenshot_mtime": screenshot_mtime,
        "commands": commands[-5:],  # 最近5条命令
    })


async def get_screenshot(request):
    """API: 获取最新截图"""
    if SCREENSHOT_FILE.exists():
        return web.FileResponse(SCREENSHOT_FILE)
    return web.Response(status=404, text="No screenshot available")


async def close_position_api(request):
    """API: 发送平仓命令"""
    try:
        data = await request.json()
        symbol = data.get("symbol")
        direction = data.get("direction")
        index = data.get("index", 0)  # 第几个同品种持仓

        if not symbol:
            return web.json_response({"error": "symbol required"}, status=400)

        cmd = send_command(
            "close_position",
            symbol=symbol,
            direction=direction,
            index=index
        )

        return web.json_response({
            "success": True,
            "message": f"平仓命令已发送: {symbol}",
            "command": cmd
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def close_all_api(request):
    """API: 发送全部平仓命令"""
    try:
        cmd = send_command("close_all")
        return web.json_response({
            "success": True,
            "message": "全部平仓命令已发送",
            "command": cmd
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def index(request):
    """主页 - 完全镜像仪表板（嵌入KVB网站）"""
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KVB 交易系统完全镜像</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e4e4e4;
            min-height: 100vh;
        }

        .top-bar {
            background: rgba(0,0,0,0.4);
            padding: 8px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid rgba(0,212,255,0.4);
        }
        .top-bar h1 {
            color: #00d4ff;
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .live-dot {
            width: 10px;
            height: 10px;
            background: #4ade80;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 5px #4ade80; }
            50% { opacity: 0.5; box-shadow: 0 0 15px #4ade80; }
        }
        .top-bar-right {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .status-text { color: #888; font-size: 12px; }
        .top-btn {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            border: none;
            color: white;
            padding: 6px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
        }
        .top-btn:hover { opacity: 0.9; }
        .top-btn.success { background: linear-gradient(135deg, #4ade80, #22c55e); }

        .main-container {
            display: grid;
            grid-template-columns: 280px 1fr 300px;
            gap: 10px;
            padding: 10px;
            height: calc(100vh - 50px);
        }

        /* 左侧栏 */
        .left-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-y: auto;
        }

        /* 中间 - KVB iframe */
        .center-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow: hidden;
        }

        /* 右侧栏 - 日志 */
        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow: hidden;
        }

        .card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            overflow: hidden;
        }
        .card-header {
            background: rgba(0,212,255,0.1);
            padding: 8px 12px;
            font-size: 13px;
            font-weight: bold;
            color: #00d4ff;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-body {
            padding: 10px;
        }
        .card-body.no-padding { padding: 0; }

        /* 统计 */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .stat-item {
            background: rgba(255,255,255,0.02);
            border-radius: 6px;
            padding: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 2px;
        }
        .stat-label {
            font-size: 10px;
            color: #666;
        }
        .stat-item.danger .stat-value { color: #ef4444; }
        .stat-item.warning .stat-value { color: #f59e0b; }
        .stat-item.success .stat-value { color: #4ade80; }
        .stat-item.info .stat-value { color: #00d4ff; }

        /* 持仓列表 - 带操作按钮 */
        .position-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .position-row {
            display: grid;
            grid-template-columns: 65px 35px 40px 50px auto;
            gap: 5px;
            padding: 6px 10px;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            font-size: 11px;
            align-items: center;
        }
        .position-row:hover {
            background: rgba(255,255,255,0.03);
        }
        .position-row .symbol { font-weight: bold; }
        .position-row .long { color: #4ade80; }
        .position-row .short { color: #ef4444; }
        .position-row .price { color: #888; }
        .close-btn {
            background: #ef4444;
            border: none;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
        }
        .close-btn:hover { background: #dc2626; }
        .close-btn:disabled { background: #666; cursor: not-allowed; }

        /* KVB iframe容器 */
        .kvb-container {
            flex: 1;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }
        .kvb-iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .kvb-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(0,0,0,0.8);
            color: #888;
            font-size: 14px;
            flex-direction: column;
            gap: 15px;
        }
        .kvb-overlay a {
            color: #00d4ff;
            text-decoration: none;
            padding: 10px 20px;
            border: 1px solid #00d4ff;
            border-radius: 5px;
        }
        .kvb-overlay a:hover {
            background: rgba(0,212,255,0.2);
        }

        /* 日志 */
        .log-container {
            flex: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .log-tabs {
            display: flex;
            gap: 3px;
            padding: 8px;
            background: rgba(0,0,0,0.2);
            border-bottom: 1px solid rgba(255,255,255,0.05);
            flex-wrap: wrap;
        }
        .log-tab {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 10px;
            cursor: pointer;
            background: rgba(255,255,255,0.05);
            border: none;
            color: #888;
            transition: all 0.2s;
        }
        .log-tab:hover { background: rgba(255,255,255,0.1); }
        .log-tab.active {
            background: #00d4ff;
            color: #000;
        }
        .log-tab .count {
            background: rgba(0,0,0,0.3);
            padding: 1px 5px;
            border-radius: 8px;
            margin-left: 3px;
            font-size: 9px;
        }
        .log-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }
        .log-item {
            padding: 6px 8px;
            margin-bottom: 4px;
            background: rgba(255,255,255,0.02);
            border-radius: 5px;
            font-size: 10px;
            border-left: 3px solid #333;
        }
        .log-item.open { border-left-color: #4ade80; }
        .log-item.close { border-left-color: #ef4444; }
        .log-item.add { border-left-color: #00d4ff; }
        .log-item.signal { border-left-color: #f59e0b; }
        .log-item.error { border-left-color: #dc2626; background: rgba(220,38,38,0.1); }
        .log-time { color: #666; margin-right: 6px; }
        .log-symbol { font-weight: bold; margin-right: 4px; }
        .log-action { margin-right: 4px; }
        .log-action.buy { color: #4ade80; }
        .log-action.sell { color: #ef4444; }
        .log-detail { color: #888; }

        /* 滚动条 */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

        /* 命令状态 */
        .command-status {
            font-size: 10px;
            padding: 3px 8px;
            border-radius: 3px;
            margin-left: 8px;
        }
        .command-status.pending { background: #f59e0b; color: #000; }
        .command-status.success { background: #4ade80; color: #000; }
        .command-status.failed { background: #ef4444; color: white; }

        /* 锁仓警告 */
        .lock-warning {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            border-radius: 5px;
            padding: 8px;
            margin-bottom: 10px;
            font-size: 11px;
            color: #ef4444;
        }
    </style>
</head>
<body>
    <div class="top-bar">
        <h1>
            <span class="live-dot"></span>
            KVB 交易系统完全镜像
        </h1>
        <div class="top-bar-right">
            <div class="status-text" id="update-status">正在连接...</div>
            <button class="top-btn" onclick="closeAllPositions()">一键全平</button>
            <button class="top-btn success" onclick="openKvbWindow()">打开KVB</button>
        </div>
    </div>

    <div class="main-container">
        <!-- 左侧面板 -->
        <div class="left-panel">
            <!-- 统计 -->
            <div class="card">
                <div class="card-header">实时统计</div>
                <div class="card-body">
                    <div class="stats-grid">
                        <div class="stat-item info">
                            <div class="stat-value" id="platform-count">-</div>
                            <div class="stat-label">平台持仓</div>
                        </div>
                        <div class="stat-item success">
                            <div class="stat-value" id="system-count">-</div>
                            <div class="stat-label">系统追踪</div>
                        </div>
                        <div class="stat-item danger" id="orphan-card">
                            <div class="stat-value" id="orphan-count">-</div>
                            <div class="stat-label">孤儿仓位</div>
                        </div>
                        <div class="stat-item warning">
                            <div class="stat-value" id="lock-count">-</div>
                            <div class="stat-label">锁仓品种</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 锁仓警告 -->
            <div id="lock-warning" class="lock-warning" style="display:none;">
                <strong>锁仓警告!</strong> 以下品种同时持有多空仓位：
                <div id="lock-symbols"></div>
            </div>

            <!-- 平台持仓（带平仓按钮）-->
            <div class="card" style="flex:1;overflow:hidden;display:flex;flex-direction:column;">
                <div class="card-header">
                    平台持仓（可操作）
                    <span id="platform-pos-count" style="color:#00d4ff"></span>
                </div>
                <div class="card-body no-padding position-list" id="platform-positions">
                    <div style="padding:15px;text-align:center;color:#666">加载中...</div>
                </div>
            </div>

            <!-- 系统持仓 -->
            <div class="card">
                <div class="card-header">
                    系统追踪
                    <span id="system-pos-count" style="color:#4ade80"></span>
                </div>
                <div class="card-body no-padding position-list" id="system-positions">
                    <div style="padding:15px;text-align:center;color:#666">加载中...</div>
                </div>
            </div>
        </div>

        <!-- 中间面板 - KVB网站 -->
        <div class="center-panel">
            <div class="card kvb-container">
                <div class="kvb-overlay" id="kvb-overlay">
                    <div>由于浏览器安全限制，无法直接嵌入KVB网站</div>
                    <a href="https://mykvb.com" target="_blank">点击在新窗口打开 KVB</a>
                    <div style="color:#666;font-size:12px;margin-top:10px;">
                        提示: 您可以将此镜像窗口与KVB窗口并排放置<br>
                        在镜像中点击"平仓"按钮即可远程操作
                    </div>
                </div>
                <iframe id="kvb-iframe" class="kvb-iframe" style="display:none;"
                    sandbox="allow-same-origin allow-scripts allow-forms"></iframe>
            </div>
        </div>

        <!-- 右侧面板 - 日志 -->
        <div class="right-panel">
            <div class="card log-container" style="flex:1">
                <div class="log-tabs">
                    <button class="log-tab active" data-type="all" onclick="switchLogTab('all')">
                        全部<span class="count" id="count-all">0</span>
                    </button>
                    <button class="log-tab" data-type="opens" onclick="switchLogTab('opens')">
                        开仓<span class="count" id="count-opens">0</span>
                    </button>
                    <button class="log-tab" data-type="closes" onclick="switchLogTab('closes')">
                        平仓<span class="count" id="count-closes">0</span>
                    </button>
                    <button class="log-tab" data-type="signals" onclick="switchLogTab('signals')">
                        信号<span class="count" id="count-signals">0</span>
                    </button>
                    <button class="log-tab" data-type="errors" onclick="switchLogTab('errors')">
                        错误<span class="count" id="count-errors">0</span>
                    </button>
                </div>
                <div class="log-list" id="log-list">
                    <div style="padding:15px;text-align:center;color:#666">加载中...</div>
                </div>
            </div>

            <!-- 命令状态 -->
            <div class="card">
                <div class="card-header">命令状态</div>
                <div class="card-body" id="command-list" style="font-size:11px;max-height:100px;overflow-y:auto;">
                    <div style="color:#666">暂无命令</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentLogType = 'all';
        let allEventsData = {};
        const KVB_URL = 'https://mykvb.com';

        function formatTime(datetime) {
            if (!datetime) return '-';
            const parts = datetime.split('T');
            if (parts.length > 1) {
                return parts[1].substring(0, 8);
            }
            return datetime;
        }

        function formatDateTime(isoStr) {
            if (!isoStr) return '-';
            try {
                const d = new Date(isoStr);
                return d.toLocaleString('zh-CN');
            } catch {
                return isoStr;
            }
        }

        function openKvbWindow() {
            window.open(KVB_URL, 'kvb_trading', 'width=1200,height=800');
        }

        function switchLogTab(type) {
            currentLogType = type;
            document.querySelectorAll('.log-tab').forEach(tab => {
                tab.classList.toggle('active', tab.dataset.type === type);
            });
            renderLogs(allEventsData);
        }

        async function closePosition(symbol, direction, index) {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = '...';

            try {
                const response = await fetch('/api/close', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol, direction, index})
                });
                const data = await response.json();

                if (data.success) {
                    btn.textContent = '已发送';
                    btn.style.background = '#4ade80';
                    setTimeout(() => {
                        btn.textContent = '平仓';
                        btn.style.background = '';
                        btn.disabled = false;
                    }, 3000);
                } else {
                    btn.textContent = '失败';
                    btn.disabled = false;
                }
            } catch (e) {
                btn.textContent = '错误';
                btn.disabled = false;
            }
        }

        async function closeAllPositions() {
            if (!confirm('确定要平掉所有持仓吗？')) return;

            try {
                const response = await fetch('/api/close_all', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                const data = await response.json();
                alert(data.message || '命令已发送');
            } catch (e) {
                alert('发送失败: ' + e.message);
            }
        }

        function renderLogs(events) {
            const container = document.getElementById('log-list');
            let logs = [];

            if (currentLogType === 'all') {
                logs = [
                    ...(events.opens || []).map(e => ({...e, _type: 'open'})),
                    ...(events.closes || []).map(e => ({...e, _type: 'close'})),
                    ...(events.adds || []).map(e => ({...e, _type: 'add'})),
                    ...(events.signals || []).map(e => ({...e, _type: 'signal'})),
                    ...(events.errors || []).map(e => ({...e, _type: 'error'})),
                ].sort((a, b) => (b.ts || 0) - (a.ts || 0));
            } else {
                logs = (events[currentLogType] || []).map(e => ({...e, _type: currentLogType.replace(/s$/, '')}));
                logs.sort((a, b) => (b.ts || 0) - (a.ts || 0));
            }

            if (logs.length === 0) {
                container.innerHTML = '<div style="padding:15px;text-align:center;color:#666">暂无日志</div>';
                return;
            }

            container.innerHTML = logs.slice(0, 80).map(log => {
                const time = formatTime(log.datetime);
                const type = log._type || log.kind || '';
                const symbol = log.symbol || '';
                const direction = log.direction || '';
                const reason = log.reason || log.signal_type || '';
                const price = log.price ? log.price.toFixed(2) : '';
                const lot = log.lot_size || log.lot || '';
                const pnl = log.pnl !== undefined ? (log.pnl >= 0 ? '+' : '') + log.pnl.toFixed(2) : '';

                let detail = '';
                if (type === 'open' || type === 'add') {
                    detail = `${lot}手 @ ${price}`;
                } else if (type === 'close') {
                    detail = `${lot}手 ${pnl ? 'P&L:' + pnl : ''}`;
                } else if (type === 'signal') {
                    detail = reason;
                } else if (type === 'error') {
                    detail = log.error || log.message || '';
                }

                const actionClass = direction === 'long' ? 'buy' : direction === 'short' ? 'sell' : '';
                const actionText = direction === 'long' ? 'BUY' : direction === 'short' ? 'SELL' : type.toUpperCase();

                return `
                    <div class="log-item ${type}">
                        <span class="log-time">${time}</span>
                        <span class="log-symbol">${symbol}</span>
                        <span class="log-action ${actionClass}">${actionText}</span>
                        <span class="log-detail">${detail}</span>
                    </div>
                `;
            }).join('');
        }

        function renderPlatformPositions(positions, systemPositions, platformBySymbol) {
            const container = document.getElementById('platform-positions');
            const trackedSymbols = Object.keys(systemPositions || {});

            if (!positions || positions.length === 0) {
                container.innerHTML = '<div style="padding:15px;text-align:center;color:#666">平台无持仓</div>';
                return;
            }

            // 检测锁仓
            const lockSymbols = [];
            Object.entries(platformBySymbol || {}).forEach(([symbol, data]) => {
                if (data.long > 0 && data.short > 0) {
                    lockSymbols.push(`${symbol}(多${data.long}/空${data.short})`);
                }
            });

            // 更新锁仓警告
            const lockWarning = document.getElementById('lock-warning');
            const lockCountEl = document.getElementById('lock-count');
            if (lockSymbols.length > 0) {
                lockWarning.style.display = 'block';
                document.getElementById('lock-symbols').textContent = lockSymbols.join(', ');
                lockCountEl.textContent = lockSymbols.length;
            } else {
                lockWarning.style.display = 'none';
                lockCountEl.textContent = '0';
            }

            // 按品种分组
            const grouped = {};
            positions.forEach((pos, idx) => {
                const symbol = pos.symbol || 'UNKNOWN';
                if (!grouped[symbol]) grouped[symbol] = [];
                grouped[symbol].push({...pos, _index: idx});
            });

            let html = '';
            Object.entries(grouped).forEach(([symbol, poses]) => {
                const isTracked = trackedSymbols.includes(symbol);
                poses.forEach((pos, i) => {
                    const isOrphan = !isTracked || i > 0;
                    const dirClass = pos.direction === 'long' ? 'long' : 'short';
                    const dirText = pos.direction === 'long' ? '多' : '空';

                    html += `
                        <div class="position-row">
                            <span class="symbol">${symbol}</span>
                            <span class="${dirClass}">${dirText}</span>
                            <span>${pos.lot_size || '-'}</span>
                            <span>${isOrphan ? '<span style="color:#f59e0b">孤儿</span>' : '<span style="color:#4ade80">追踪</span>'}</span>
                            <button class="close-btn" onclick="closePosition('${symbol}', '${pos.direction}', ${pos._index})">平仓</button>
                        </div>
                    `;
                });
            });

            container.innerHTML = html;
        }

        function renderSystemPositions(positions) {
            const container = document.getElementById('system-positions');

            if (!positions || Object.keys(positions).length === 0) {
                container.innerHTML = '<div style="padding:15px;text-align:center;color:#666">无追踪持仓</div>';
                return;
            }

            container.innerHTML = Object.entries(positions).map(([symbol, pos]) => {
                const dirClass = pos.direction === 'long' ? 'long' : 'short';
                const dirText = pos.direction === 'long' ? '多' : '空';
                return `
                    <div class="position-row" style="grid-template-columns: 65px 35px 40px auto;">
                        <span class="symbol">${symbol}</span>
                        <span class="${dirClass}">${dirText}</span>
                        <span>${pos.lot_size || '-'}</span>
                        <span class="price">${pos.entry_price?.toFixed(2) || '-'}</span>
                    </div>
                `;
            }).join('');
        }

        function renderCommands(commands) {
            const container = document.getElementById('command-list');

            if (!commands || commands.length === 0) {
                container.innerHTML = '<div style="color:#666">暂无命令</div>';
                return;
            }

            container.innerHTML = commands.slice().reverse().map(cmd => {
                const time = formatTime(cmd.datetime);
                const statusClass = cmd.status || 'pending';
                return `
                    <div style="margin-bottom:5px;">
                        <span style="color:#666">${time}</span>
                        <span style="margin-left:5px">${cmd.type}</span>
                        ${cmd.symbol ? `<span style="margin-left:5px;font-weight:bold">${cmd.symbol}</span>` : ''}
                        <span class="command-status ${statusClass}">${cmd.status || 'pending'}</span>
                    </div>
                `;
            }).join('');
        }

        async function refreshData() {
            try {
                const response = await fetch('/api/sync');
                const data = await response.json();

                // 统计
                document.getElementById('platform-count').textContent = data.platform_count || 0;
                document.getElementById('system-count').textContent = data.system_count || 0;
                document.getElementById('orphan-count').textContent = data.orphan_count || 0;

                // 日志计数
                document.getElementById('count-all').textContent =
                    (data.statistics?.total_opens || 0) +
                    (data.statistics?.total_closes || 0) +
                    (data.statistics?.total_signals || 0);
                document.getElementById('count-opens').textContent = data.statistics?.total_opens || 0;
                document.getElementById('count-closes').textContent = data.statistics?.total_closes || 0;
                document.getElementById('count-signals').textContent = data.statistics?.total_signals || 0;
                document.getElementById('count-errors').textContent = data.statistics?.total_errors || 0;

                // 持仓计数
                document.getElementById('platform-pos-count').textContent = `(${data.platform_count || 0})`;
                document.getElementById('system-pos-count').textContent = `(${data.system_count || 0})`;

                // 孤儿卡片颜色
                const orphanCard = document.getElementById('orphan-card');
                orphanCard.className = data.orphan_count > 0 ? 'stat-item danger' : 'stat-item success';

                // 更新时间
                document.getElementById('update-status').textContent =
                    '最后更新: ' + formatDateTime(data.timestamp);

                // 渲染
                allEventsData = data.events || {};
                renderLogs(allEventsData);
                renderPlatformPositions(data.platform_positions, data.system_positions, data.platform_by_symbol);
                renderSystemPositions(data.system_positions);
                renderCommands(data.commands);

            } catch (error) {
                console.error('刷新失败:', error);
                document.getElementById('update-status').textContent = '连接失败 - ' + error.message;
            }
        }

        // 尝试加载KVB iframe
        function tryLoadKvb() {
            const iframe = document.getElementById('kvb-iframe');
            const overlay = document.getElementById('kvb-overlay');

            // 大多数交易网站都禁止iframe嵌入，所以我们默认显示overlay
            // 如果用户想尝试，可以取消下面的注释
            // iframe.src = KVB_URL;
            // iframe.onload = () => {
            //     overlay.style.display = 'none';
            //     iframe.style.display = 'block';
            // };
            // iframe.onerror = () => {
            //     overlay.style.display = 'flex';
            //     iframe.style.display = 'none';
            // };
        }

        // 初始化
        tryLoadKvb();
        refreshData();
        setInterval(refreshData, 2000);
    </script>
</body>
</html>"""
    return web.Response(text=html, content_type='text/html')


async def start_dashboard(port=8080):
    """启动仪表板服务器"""
    app = web.Application()

    # 配置CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*"
        )
    })

    # 路由
    app.router.add_get('/', index)
    app.router.add_get('/api/sync', get_full_sync_data)
    app.router.add_get('/api/screenshot', get_screenshot)
    app.router.add_post('/api/close', close_position_api)
    app.router.add_post('/api/close_all', close_all_api)

    # 为所有路由添加CORS
    for route in list(app.router.routes()):
        cors.add(route)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', port)
    await site.start()

    print(f"""
================================================================================
  KVB 交易系统完全镜像仪表板 (带远程操作)
================================================================================

  访问地址: http://127.0.0.1:{port}

  功能:
  ├─ 实时同步平台所有持仓
  ├─ 锁仓检测与警告
  ├─ 远程平仓操作（点击按钮即可平仓）
  ├─ 一键全平功能
  ├─ 所有开仓/平仓/信号日志
  └─ 命令状态追踪

  注意: 平仓命令需要auto_trader正在运行才会执行

  按 Ctrl+C 停止
================================================================================
""")

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(start_dashboard())
    except KeyboardInterrupt:
        print("\n仪表板已停止")
