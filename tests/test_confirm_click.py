import asyncio
from unittest.mock import patch

import pytest


class _FakeKeyboard:
    def __init__(self, page):
        self._page = page

    async def press(self, key: str):
        if key == "Enter":
            self._page.dialog_open = False


class _FakeLocator:
    def __init__(self, *, page, visible=False, on_click=None, text="", children=None):
        self._page = page
        self._visible = visible
        self._on_click = on_click
        self._text = text
        self._children = children or {}

    @property
    def first(self):
        return self

    def nth(self, _i: int):
        return self

    def locator(self, selector: str):
        return self._children.get(selector, _FakeLocator(page=self._page, visible=False))

    async def count(self):
        return 1 if self._visible else 0

    async def is_visible(self):
        return bool(self._visible)

    async def scroll_into_view_if_needed(self, timeout=0):
        return None

    async def click(self, timeout=0, force=False):
        if callable(self._on_click):
            self._on_click()
        return None

    async def inner_text(self):
        return self._text


class _FakePage:
    def __init__(self, *, dialog_open: bool, dialog_children=None, global_children=None):
        self.dialog_open = dialog_open
        self.keyboard = _FakeKeyboard(self)
        self._dialog_children = dialog_children or {}
        self._global_children = global_children or {}

    def locator(self, selector: str):
        if selector in ("[role='dialog']", ".order-dialog, .order-modal, .trade-modal, .order-ticket, .trade-ticket, .order-panel, .trade-panel"):
            return _FakeLocator(
                page=self,
                visible=self.dialog_open,
                children=self._dialog_children,
            )
        return self._global_children.get(selector, _FakeLocator(page=self, visible=False))


@pytest.mark.asyncio
async def test_click_confirm_prefers_dialog_button_and_closes_dialog():
    from auto_trader import AutoTrader

    def close_dialog():
        page.dialog_open = False

    dialog_scope_children = {
        'button:has-text("市价买入")': _FakeLocator(page=None, visible=True, on_click=close_dialog),
    }

    page = _FakePage(dialog_open=True, dialog_children=dialog_scope_children)
    dialog_scope_children['button:has-text("市价买入")']._page = page

    with patch("auto_trader.DataFetcher"):
        trader = AutoTrader(symbols=["ETHUSD"], auto_trade=False, live_trade=False)
    trader.fetcher.page = page

    ok = await trader._click_confirm("市价买入")
    assert ok is True
    assert page.dialog_open is False


@pytest.mark.asyncio
async def test_click_confirm_falls_back_to_enter_when_no_button_found():
    from auto_trader import AutoTrader

    page = _FakePage(dialog_open=True, dialog_children={})

    with patch("auto_trader.DataFetcher"):
        trader = AutoTrader(symbols=["ETHUSD"], auto_trade=False, live_trade=False)
    trader.fetcher.page = page

    ok = await trader._click_confirm("市价买入")
    assert ok is True
    assert page.dialog_open is False

