"""
Smoke test for closing a position via the trailing “X/×” button on mykvb.com/trade.

Default behavior is SAFE: it only clicks the row-end X to open the close dialog.
It will NOT click the final confirm button unless you pass --confirm.

Usage:
  1) Run start_chrome_debug.bat and login, ensure you can see the holdings table.
  2) In another terminal:
       set SMART_TRADING_CDP_URL=http://127.0.0.1:9222
       python close_x_smoke.py --symbol USOIL
     If the dialog appears and you want to auto-confirm:
       python close_x_smoke.py --symbol USOIL --confirm --verify
"""

import argparse
import asyncio
import os
import time
import logging

from data_fetcher import DataFetcher


async def _click_confirm_best_effort(page) -> bool:
    candidates = [
        "button:has-text('市价')",
        "button:has-text('确认')",
        "button:has-text('确定')",
        "button:has-text('OK')",
        "button:has-text('Confirm')",
        "[role='button']:has-text('市价')",
    ]
    for sel in candidates:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                await el.click()
                return True
        except Exception:
            continue
    return False


async def _wait_position(fetcher: DataFetcher, symbol: str, should_exist: bool, timeout: float) -> bool:
    deadline = time.time() + max(1.0, timeout)
    sym = symbol.upper()
    while time.time() < deadline:
        try:
            pos = await fetcher.get_open_positions()
            exists = any((p.get("symbol", "") or "").upper() == sym for p in (pos or []))
            if exists == should_exist:
                return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="USOIL")
    ap.add_argument("--confirm", action="store_true", help="click final confirm button in dialog")
    ap.add_argument("--verify", action="store_true", help="verify position disappears after confirm")
    ap.add_argument("--timeout", type=float, default=12.0)
    ap.add_argument("--debug", action="store_true", help="enable verbose logs from DataFetcher")
    args = ap.parse_args()

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not (os.environ.get("SMART_TRADING_CDP_URL") or os.environ.get("SMART_TRADING_CDP_URL")):
        print("Tip: set SMART_TRADING_CDP_URL=http://127.0.0.1:9222 to attach to your logged-in browser.")

    fetcher = DataFetcher(headless=False)
    ok = await fetcher.connect()
    if not ok:
        print("connect failed")
        return 2

    try:
        symbol = args.symbol.upper()
        print(f"clicking close X for {symbol} ...")
        clicked = await fetcher.close_position_by_symbol(symbol)
        print(f"close_x_clicked={clicked}")
        if not clicked:
            return 1

        if args.confirm:
            page = fetcher.page
            if not page:
                return 1
            print("trying to confirm ...")
            c = await _click_confirm_best_effort(page)
            print(f"confirm_clicked={c}")
            if args.verify:
                v = await _wait_position(fetcher, symbol, should_exist=False, timeout=args.timeout)
                print(f"verify_closed={v}")

        return 0
    finally:
        await fetcher.disconnect()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
