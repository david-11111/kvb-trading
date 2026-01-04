"""
KVB 智能交易浏览器
内嵌浏览器 + 镜像监控 + 智能交易模块
"""

import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QLineEdit, QMessageBox, QStatusBar, QGroupBox, QGridLayout,
    QSizePolicy
)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineProfile
from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
from PyQt5.QtNetwork import QNetworkProxy

# 代理配置 (根据需要修改)
PROXY_CONFIG = {
    "enabled": True,  # 是否启用代理
    "type": "http",   # http, socks5
    "host": "127.0.0.1",
    "port": 7890,     # 常见代理端口: 7890(Clash), 1080(SS), 10808(V2Ray)
}

# 数据目录
DATA_DIR = Path(__file__).parent / "trade_data"
POSITIONS_FILE = DATA_DIR / "positions.json"
PLATFORM_POSITIONS_FILE = DATA_DIR / "platform_positions.json"
COMMAND_FILE = DATA_DIR / "dashboard_commands.json"


class DataLoader:
    """数据加载器"""

    @staticmethod
    def load_system_positions():
        if POSITIONS_FILE.exists():
            try:
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {}

    @staticmethod
    def load_platform_positions():
        if PLATFORM_POSITIONS_FILE.exists():
            try:
                with open(PLATFORM_POSITIONS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return []

    @staticmethod
    def load_events(limit=100):
        today = datetime.now().strftime("%Y%m%d")
        jsonl_file = DATA_DIR / f"trade_events_{today}.jsonl"
        events = []
        if jsonl_file.exists():
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            events.append(json.loads(line.strip()))
                        except:
                            pass
            except:
                pass
        return events[-limit:]

    @staticmethod
    def send_command(cmd_type: str, **kwargs):
        command = {
            "type": cmd_type,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "status": "pending",
            **kwargs
        }
        commands = []
        if COMMAND_FILE.exists():
            try:
                with open(COMMAND_FILE, "r", encoding="utf-8") as f:
                    commands = json.load(f)
            except:
                commands = []
        commands.append(command)
        commands = commands[-10:]
        with open(COMMAND_FILE, "w", encoding="utf-8") as f:
            json.dump(commands, f, ensure_ascii=False, indent=2)
        return command


class StyledFrame(QFrame):
    """带样式的框架"""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            StyledFrame {
                background-color: #1a1a2e;
                border: 1px solid #2d2d44;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                QLabel {
                    color: #00d4ff;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 5px;
                    background-color: rgba(0, 212, 255, 0.1);
                    border-radius: 4px;
                }
            """)
            layout.addWidget(title_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(5)
        layout.addLayout(self.content_layout)


class StatCard(QWidget):
    """统计卡片"""
    def __init__(self, label, value="0", color="#00d4ff", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(2)

        self.value_label = QLabel(str(value))
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 28px;
                font-weight: bold;
            }}
        """)

        self.name_label = QLabel(label)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 11px;
            }
        """)

        layout.addWidget(self.value_label)
        layout.addWidget(self.name_label)

        self.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.02);
                border-radius: 6px;
            }
        """)

    def set_value(self, value, color=None):
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"""
                QLabel {{
                    color: {color};
                    font-size: 28px;
                    font-weight: bold;
                }}
            """)


class PositionRow(QWidget):
    """持仓行"""
    close_clicked = pyqtSignal(str, str, int)

    def __init__(self, symbol, direction, lot_size, is_orphan=False, index=0, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.direction = direction
        self.index = index

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(10)

        # 品种
        symbol_label = QLabel(symbol)
        symbol_label.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        symbol_label.setFixedWidth(70)
        layout.addWidget(symbol_label)

        # 方向
        dir_text = "多" if direction == "long" else "空"
        dir_color = "#4ade80" if direction == "long" else "#ef4444"
        dir_label = QLabel(dir_text)
        dir_label.setStyleSheet(f"color: {dir_color}; font-size: 12px;")
        dir_label.setFixedWidth(30)
        layout.addWidget(dir_label)

        # 手数
        lot_label = QLabel(str(lot_size))
        lot_label.setStyleSheet("color: #888; font-size: 12px;")
        lot_label.setFixedWidth(40)
        layout.addWidget(lot_label)

        # 状态标签
        if is_orphan:
            status_label = QLabel("孤儿")
            status_label.setStyleSheet("""
                QLabel {
                    color: #000;
                    background-color: #f59e0b;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 10px;
                }
            """)
        else:
            status_label = QLabel("追踪")
            status_label.setStyleSheet("""
                QLabel {
                    color: #000;
                    background-color: #4ade80;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 10px;
                }
            """)
        layout.addWidget(status_label)

        layout.addStretch()

        # 平仓按钮
        close_btn = QPushButton("平仓")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:pressed {
                background-color: #b91c1c;
            }
        """)
        close_btn.clicked.connect(self._on_close)
        layout.addWidget(close_btn)

        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
            QWidget:hover {
                background-color: rgba(255, 255, 255, 0.03);
            }
        """)

    def _on_close(self):
        self.close_clicked.emit(self.symbol, self.direction, self.index)


class LogItem(QWidget):
    """日志项"""
    def __init__(self, event, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        kind = event.get("kind", "")
        symbol = event.get("symbol", "")
        direction = event.get("direction", "")
        dt = event.get("datetime", "")

        # 时间
        time_str = dt.split("T")[1][:8] if "T" in dt else dt
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #666; font-size: 10px;")
        time_label.setFixedWidth(55)
        layout.addWidget(time_label)

        # 品种
        symbol_label = QLabel(symbol)
        symbol_label.setStyleSheet("color: white; font-weight: bold; font-size: 11px;")
        symbol_label.setFixedWidth(60)
        layout.addWidget(symbol_label)

        # 类型/方向
        if direction:
            action_text = "BUY" if direction == "long" else "SELL"
            action_color = "#4ade80" if direction == "long" else "#ef4444"
        else:
            action_text = kind.upper()
            action_color = "#888"

        action_label = QLabel(action_text)
        action_label.setStyleSheet(f"color: {action_color}; font-size: 11px;")
        action_label.setFixedWidth(40)
        layout.addWidget(action_label)

        # 详情
        detail = ""
        if kind == "open" or kind == "add":
            lot = event.get("lot_size", "")
            price = event.get("price", "")
            detail = f"{lot}手 @ {price}"
        elif kind == "close":
            pnl = event.get("pnl", 0)
            detail = f"P&L: {'+' if pnl >= 0 else ''}{pnl:.2f}"
        elif kind == "signal":
            detail = event.get("reason", "")[:30]

        detail_label = QLabel(detail)
        detail_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(detail_label)

        layout.addStretch()

        # 根据类型设置左边框颜色
        border_colors = {
            "open": "#4ade80",
            "close": "#ef4444",
            "add": "#00d4ff",
            "signal": "#f59e0b",
            "error": "#dc2626",
        }
        border_color = border_colors.get(kind, "#333")

        self.setStyleSheet(f"""
            QWidget {{
                background-color: rgba(255, 255, 255, 0.02);
                border-left: 3px solid {border_color};
                border-radius: 4px;
            }}
        """)


class LeftPanel(QWidget):
    """左侧面板 - 统计和持仓"""
    close_position_signal = pyqtSignal(str, str, int)
    close_all_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 5, 10)
        layout.setSpacing(10)

        # 标题
        title = QLabel("智能交易镜像")
        title.setStyleSheet("""
            QLabel {
                color: #00d4ff;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        layout.addWidget(title)

        # 统计卡片
        stats_frame = StyledFrame("实时统计")
        stats_grid = QGridLayout()
        stats_grid.setSpacing(8)

        self.stat_platform = StatCard("平台持仓", "0", "#00d4ff")
        self.stat_system = StatCard("系统追踪", "0", "#4ade80")
        self.stat_orphan = StatCard("孤儿仓位", "0", "#ef4444")
        self.stat_lock = StatCard("锁仓品种", "0", "#f59e0b")

        stats_grid.addWidget(self.stat_platform, 0, 0)
        stats_grid.addWidget(self.stat_system, 0, 1)
        stats_grid.addWidget(self.stat_orphan, 1, 0)
        stats_grid.addWidget(self.stat_lock, 1, 1)

        stats_frame.content_layout.addLayout(stats_grid)
        layout.addWidget(stats_frame)

        # 锁仓警告
        self.lock_warning = QLabel()
        self.lock_warning.setStyleSheet("""
            QLabel {
                color: #ef4444;
                background-color: rgba(239, 68, 68, 0.2);
                border: 1px solid #ef4444;
                border-radius: 5px;
                padding: 8px;
                font-size: 11px;
            }
        """)
        self.lock_warning.setWordWrap(True)
        self.lock_warning.hide()
        layout.addWidget(self.lock_warning)

        # 操作按钮
        btn_layout = QHBoxLayout()

        close_all_btn = QPushButton("一键全平")
        close_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        close_all_btn.clicked.connect(self._on_close_all)
        btn_layout.addWidget(close_all_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff;
                color: black;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00b8e6;
            }
        """)
        btn_layout.addWidget(refresh_btn)

        layout.addLayout(btn_layout)

        # 平台持仓列表
        positions_frame = StyledFrame("平台持仓（可操作）")

        self.positions_scroll = QScrollArea()
        self.positions_scroll.setWidgetResizable(True)
        self.positions_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        self.positions_container = QWidget()
        self.positions_layout = QVBoxLayout(self.positions_container)
        self.positions_layout.setContentsMargins(0, 0, 0, 0)
        self.positions_layout.setSpacing(2)
        self.positions_layout.addStretch()

        self.positions_scroll.setWidget(self.positions_container)
        positions_frame.content_layout.addWidget(self.positions_scroll)

        layout.addWidget(positions_frame, 1)

        self.setStyleSheet("""
            QWidget {
                background-color: #0f0f1a;
            }
        """)

    def _on_close_all(self):
        reply = QMessageBox.question(
            self, "确认", "确定要平掉所有持仓吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.close_all_signal.emit()

    def update_data(self, platform_positions, system_positions):
        # 更新统计
        platform_count = len(platform_positions)
        system_count = len(system_positions)

        # 计算孤儿仓位
        tracked_symbols = set(system_positions.keys())
        orphan_count = max(0, platform_count - len(tracked_symbols))

        # 检测锁仓
        platform_by_symbol = {}
        for pos in platform_positions:
            symbol = pos.get("symbol", "UNKNOWN")
            direction = pos.get("direction", "unknown")
            if symbol not in platform_by_symbol:
                platform_by_symbol[symbol] = {"long": 0, "short": 0}
            if direction in platform_by_symbol[symbol]:
                platform_by_symbol[symbol][direction] += 1

        lock_symbols = []
        for symbol, counts in platform_by_symbol.items():
            if counts["long"] > 0 and counts["short"] > 0:
                lock_symbols.append(f"{symbol}(多{counts['long']}/空{counts['short']})")

        # 更新统计卡片
        self.stat_platform.set_value(platform_count)
        self.stat_system.set_value(system_count)
        self.stat_orphan.set_value(orphan_count, "#ef4444" if orphan_count > 0 else "#4ade80")
        self.stat_lock.set_value(len(lock_symbols), "#ef4444" if lock_symbols else "#4ade80")

        # 更新锁仓警告
        if lock_symbols:
            self.lock_warning.setText(f"锁仓警告！{', '.join(lock_symbols)}")
            self.lock_warning.show()
        else:
            self.lock_warning.hide()

        # 更新持仓列表
        # 清空现有项
        while self.positions_layout.count() > 1:
            item = self.positions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 添加新项
        grouped = {}
        for idx, pos in enumerate(platform_positions):
            symbol = pos.get("symbol", "UNKNOWN")
            if symbol not in grouped:
                grouped[symbol] = []
            grouped[symbol].append((idx, pos))

        for symbol, poses in grouped.items():
            is_tracked = symbol in system_positions
            for i, (idx, pos) in enumerate(poses):
                is_orphan = not is_tracked or i > 0
                row = PositionRow(
                    symbol,
                    pos.get("direction", "unknown"),
                    pos.get("lot_size", 0),
                    is_orphan,
                    idx
                )
                row.close_clicked.connect(self._on_position_close)
                self.positions_layout.insertWidget(self.positions_layout.count() - 1, row)

    def _on_position_close(self, symbol, direction, index):
        self.close_position_signal.emit(symbol, direction, index)


class RightPanel(QWidget):
    """右侧面板 - 日志"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 10, 10, 10)
        layout.setSpacing(10)

        # 日志标题
        title = QLabel("交易日志")
        title.setStyleSheet("""
            QLabel {
                color: #00d4ff;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        layout.addWidget(title)

        # 日志选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2d2d44;
                background-color: #1a1a2e;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #1a1a2e;
                color: #888;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #00d4ff;
                color: black;
            }
        """)

        # 各类型日志滚动区域
        self.log_scrolls = {}
        self.log_containers = {}
        self.log_layouts = {}

        for tab_name in ["全部", "开仓", "平仓", "信号", "错误"]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(5, 5, 5, 5)
            container_layout.setSpacing(3)
            container_layout.addStretch()

            scroll.setWidget(container)
            self.tab_widget.addTab(scroll, tab_name)

            self.log_scrolls[tab_name] = scroll
            self.log_containers[tab_name] = container
            self.log_layouts[tab_name] = container_layout

        layout.addWidget(self.tab_widget, 1)

        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 11px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.status_label)

        self.setStyleSheet("""
            QWidget {
                background-color: #0f0f1a;
            }
        """)

    def update_logs(self, events):
        # 分类事件
        categorized = {
            "全部": [],
            "开仓": [],
            "平仓": [],
            "信号": [],
            "错误": [],
        }

        for event in events:
            kind = event.get("kind", "")
            categorized["全部"].append(event)

            if kind == "open" or kind == "add":
                categorized["开仓"].append(event)
            elif kind == "close":
                categorized["平仓"].append(event)
            elif kind == "signal":
                categorized["信号"].append(event)
            elif kind == "error" or kind == "runtime_error":
                categorized["错误"].append(event)

        # 更新各选项卡
        for tab_name, tab_events in categorized.items():
            layout = self.log_layouts[tab_name]

            # 清空现有项
            while layout.count() > 1:
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            # 添加新项（倒序，最新在前）
            for event in reversed(tab_events[-50:]):
                item = LogItem(event)
                layout.insertWidget(layout.count() - 1, item)

        # 更新选项卡标题中的计数
        self.tab_widget.setTabText(0, f"全部({len(categorized['全部'])})")
        self.tab_widget.setTabText(1, f"开仓({len(categorized['开仓'])})")
        self.tab_widget.setTabText(2, f"平仓({len(categorized['平仓'])})")
        self.tab_widget.setTabText(3, f"信号({len(categorized['信号'])})")
        self.tab_widget.setTabText(4, f"错误({len(categorized['错误'])})")

    def set_status(self, text):
        self.status_label.setText(text)


class TradingBrowser(QMainWindow):
    """主窗口 - 智能交易浏览器"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KVB 智能交易浏览器")
        self.setGeometry(100, 100, 1600, 900)

        # 设置深色主题
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a14;
            }
            QSplitter::handle {
                background-color: #2d2d44;
                width: 2px;
            }
        """)

        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧面板
        self.left_panel = LeftPanel()
        self.left_panel.close_position_signal.connect(self._on_close_position)
        self.left_panel.close_all_signal.connect(self._on_close_all)
        main_layout.addWidget(self.left_panel)

        # 中间浏览器
        browser_container = QWidget()
        browser_layout = QVBoxLayout(browser_container)
        browser_layout.setContentsMargins(5, 10, 5, 10)
        browser_layout.setSpacing(5)

        # 地址栏
        url_bar = QWidget()
        url_layout = QHBoxLayout(url_bar)
        url_layout.setContentsMargins(0, 0, 0, 0)
        url_layout.setSpacing(5)

        self.url_input = QLineEdit("https://mykvb.com")
        self.url_input.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a2e;
                color: white;
                border: 1px solid #2d2d44;
                border-radius: 5px;
                padding: 8px 12px;
                font-size: 13px;
            }
        """)
        self.url_input.returnPressed.connect(self._navigate)
        url_layout.addWidget(self.url_input)

        go_btn = QPushButton("前往")
        go_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff;
                color: black;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00b8e6;
            }
        """)
        go_btn.clicked.connect(self._navigate)
        url_layout.addWidget(go_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ade80;
                color: black;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
        """)
        refresh_btn.clicked.connect(self._refresh_browser)
        url_layout.addWidget(refresh_btn)

        browser_layout.addWidget(url_bar)

        # WebEngine浏览器
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://mykvb.com"))
        self.browser.urlChanged.connect(self._on_url_changed)
        self.browser.loadFinished.connect(self._on_page_loaded)  # 页面加载完成后注入脚本

        browser_layout.addWidget(self.browser, 1)

        main_layout.addWidget(browser_container, 1)

        # 右侧面板
        self.right_panel = RightPanel()
        main_layout.addWidget(self.right_panel)

        # 状态栏
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #0f0f1a;
                color: #666;
            }
        """)
        self.statusBar().showMessage("就绪 - 等待连接...")

        # 定时刷新数据
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_data)
        self.refresh_timer.start(2000)  # 每2秒刷新

        # 初始刷新
        self._refresh_data()

    def _navigate(self):
        url = self.url_input.text()
        if not url.startswith("http"):
            url = "https://" + url
        self.browser.setUrl(QUrl(url))

    def _refresh_browser(self):
        self.browser.reload()

    def _on_url_changed(self, url):
        self.url_input.setText(url.toString())

    def _on_page_loaded(self, ok):
        """页面加载完成后注入辅助脚本"""
        if ok:
            self._inject_helper_script()
            self.statusBar().showMessage("页面已加载，辅助脚本已注入")

    def _refresh_data(self):
        try:
            platform_positions = DataLoader.load_platform_positions()
            system_positions = DataLoader.load_system_positions()
            events = DataLoader.load_events()

            self.left_panel.update_data(platform_positions, system_positions)
            self.right_panel.update_logs(events)

            self.statusBar().showMessage(
                f"最后更新: {datetime.now().strftime('%H:%M:%S')} | "
                f"平台持仓: {len(platform_positions)} | "
                f"系统追踪: {len(system_positions)}"
            )
            self.right_panel.set_status(f"更新于 {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            self.statusBar().showMessage(f"刷新失败: {e}")

    def _on_close_position(self, symbol, direction, index):
        """平仓操作 - 同时通过JS注入和命令文件两种方式"""
        try:
            # 方式1: 发送命令给auto_trader（后台Playwright执行）
            DataLoader.send_command("close_position", symbol=symbol, direction=direction, index=index)

            # 方式2: 通过JavaScript注入直接在嵌入浏览器中操作
            # KVB平台: 持仓行右侧有 x 按钮，或右键菜单有"平仓"选项
            js_code = f'''
            (function() {{
                var symbol = '{symbol}';
                console.log('正在查找持仓: ' + symbol);

                // 方法1: 查找包含品种的行，点击 x 按钮
                var allElements = document.querySelectorAll('*');
                for (var i = 0; i < allElements.length; i++) {{
                    var el = allElements[i];
                    var text = el.innerText || el.textContent || '';

                    // 找到包含品种名的元素
                    if (text.indexOf(symbol) !== -1 && text.length < 500) {{
                        // 在这个元素或其父元素中找 x 按钮
                        var parent = el;
                        for (var p = 0; p < 5; p++) {{
                            if (!parent) break;

                            // 查找 x 按钮 (各种可能的选择器)
                            var xBtn = parent.querySelector('[class*="close"], [class*="x"], .x, .X, svg, [data-icon="close"], [title*="关闭"], [title*="平仓"]');
                            if (!xBtn) {{
                                // 查找包含 x 或 × 文本的元素
                                var spans = parent.querySelectorAll('span, div, button, a, i');
                                for (var s = 0; s < spans.length; s++) {{
                                    var spanText = spans[s].innerText || spans[s].textContent || '';
                                    if (spanText.trim() === 'x' || spanText.trim() === 'X' || spanText.trim() === '×' || spanText.trim() === '✕') {{
                                        xBtn = spans[s];
                                        break;
                                    }}
                                }}
                            }}

                            if (xBtn) {{
                                console.log('找到x按钮，点击中...');
                                xBtn.click();
                                return 'clicked_x';
                            }}
                            parent = parent.parentElement;
                        }}
                    }}
                }}

                // 方法2: 尝试右键点击触发上下文菜单
                var rows = document.querySelectorAll('tr, [class*="position"], [class*="row"], [class*="item"]');
                for (var i = 0; i < rows.length; i++) {{
                    var row = rows[i];
                    if (row.textContent.indexOf(symbol) !== -1) {{
                        // 模拟右键点击
                        var evt = new MouseEvent('contextmenu', {{
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            button: 2
                        }});
                        row.dispatchEvent(evt);
                        console.log('已触发右键菜单: ' + symbol);

                        // 等待菜单出现后点击"平仓"
                        setTimeout(function() {{
                            var menuItems = document.querySelectorAll('[class*="menu"] *, [class*="dropdown"] *, [class*="context"] *');
                            for (var m = 0; m < menuItems.length; m++) {{
                                if (menuItems[m].textContent.indexOf('平仓') !== -1) {{
                                    menuItems[m].click();
                                    console.log('点击了平仓菜单项');
                                }}
                            }}
                        }}, 300);

                        return 'right_clicked';
                    }}
                }}

                return 'not_found';
            }})();
            '''
            self.browser.page().runJavaScript(js_code, self._on_js_result)

            self.statusBar().showMessage(f"平仓命令已发送: {symbol}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"发送失败: {e}")

    def _on_js_result(self, result):
        """JavaScript执行结果回调"""
        if result == 'clicked':
            self.statusBar().showMessage("JS注入: 平仓按钮已点击")
        elif result == 'not_found':
            self.statusBar().showMessage("JS注入: 未找到平仓按钮，使用后台命令")

    def _on_close_all(self):
        """一键全平 - 通过JS注入和命令文件"""
        try:
            # 发送命令给auto_trader
            DataLoader.send_command("close_all")

            # 通过JavaScript尝试点击所有平仓按钮
            js_code = '''
            (function() {
                var count = 0;
                var closeButtons = document.querySelectorAll(
                    'button[class*="close"], .close-btn, [data-action="close"]'
                );
                closeButtons.forEach(function(btn) {
                    if (btn.textContent.indexOf('平仓') !== -1 ||
                        btn.textContent.indexOf('关闭') !== -1 ||
                        btn.textContent.indexOf('Close') !== -1) {
                        btn.click();
                        count++;
                    }
                });
                return count;
            })();
            '''
            self.browser.page().runJavaScript(js_code, lambda r: self.statusBar().showMessage(f"JS注入: 尝试点击了 {r} 个平仓按钮"))

            QMessageBox.information(self, "命令已发送", "全部平仓命令已发送")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"发送失败: {e}")

    def _inject_helper_script(self):
        """注入辅助脚本到网页中，用于增强交互"""
        helper_js = '''
        // KVB交易助手脚本
        window.kvbHelper = {
            // 获取所有持仓
            getPositions: function() {
                var positions = [];
                document.querySelectorAll('tr, [class*="position"]').forEach(function(row) {
                    var text = row.textContent;
                    if (text.match(/ETHUSD|XAUUSD|USOIL/)) {
                        positions.push({
                            text: text.substring(0, 100),
                            element: row
                        });
                    }
                });
                return positions.length;
            },

            // 关闭指定品种
            closeSymbol: function(symbol) {
                var rows = document.querySelectorAll('tr, [class*="position"]');
                for (var i = 0; i < rows.length; i++) {
                    if (rows[i].textContent.indexOf(symbol) !== -1) {
                        var btn = rows[i].querySelector('button, a, span');
                        if (btn && (btn.textContent.indexOf('平仓') !== -1 || btn.textContent.indexOf('Close') !== -1)) {
                            btn.click();
                            return true;
                        }
                    }
                }
                return false;
            }
        };
        console.log('KVB交易助手已加载');
        '''
        self.browser.page().runJavaScript(helper_js)


def main():
    # 确保数据目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 设置代理
    if PROXY_CONFIG.get("enabled", False):
        proxy = QNetworkProxy()
        proxy_type = PROXY_CONFIG.get("type", "http").lower()
        if proxy_type == "socks5":
            proxy.setType(QNetworkProxy.Socks5Proxy)
        else:
            proxy.setType(QNetworkProxy.HttpProxy)
        proxy.setHostName(PROXY_CONFIG.get("host", "127.0.0.1"))
        proxy.setPort(PROXY_CONFIG.get("port", 7890))
        QNetworkProxy.setApplicationProxy(proxy)
        print(f"代理已启用: {proxy_type}://{PROXY_CONFIG.get('host')}:{PROXY_CONFIG.get('port')}")

    # 设置深色调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(15, 15, 26))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(26, 26, 46))
    palette.setColor(QPalette.AlternateBase, QColor(15, 15, 26))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(26, 26, 46))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(0, 212, 255))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = TradingBrowser()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
