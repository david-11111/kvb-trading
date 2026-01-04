"""
Batch close positions ticket-by-ticket (safer than close-by-symbol when multiple positions exist).

SAFE DEFAULT:
  - Without --confirm, it only lists the tickets it would close (no clicks).

Prereq:
  - Run start_chrome_debug.bat and login, keep https://mykvb.com/trade open.
  - Set SMART_TRADING_CDP_URL (default http://localhost:9222)

Examples (PowerShell):
  cd "D:\\20240313整理文件\\Desktop\\老王说油推广\\smart_trading"
  $env:SMART_TRADING_CDP_URL="http://localhost:9222"

  # List all open tickets for XAUUSD
  python .\\close_tickets.py --symbol XAUUSD

  # Close at most 1 ticket and verify it disappears (REAL action)
  python .\\close_tickets.py --symbol ETHUSD --max 1 --confirm --verify --timeout 20
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

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


async def _wait_ticket_gone(fetcher: DataFetcher, ticket: str, timeout: float) -> bool:
    deadline = time.time() + max(1.0, timeout)
    t = ticket.lstrip("#")
    while time.time() < deadline:
        try:
            cur = await fetcher.get_open_positions()
            if not any(str(p.get("ticket") or "").lstrip("#") == t for p in (cur or [])):
                return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


def _collect_tickets(positions: list[dict], symbol: str) -> list[str]:
    sym = symbol.upper()
    tickets: list[str] = []
    seen = set()
    for p in positions or []:
        if (p.get("symbol") or "").upper() != sym:
            continue
        t = str(p.get("ticket") or "").lstrip("#").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        tickets.append(t)
    return tickets


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="symbol to close, e.g. ETHUSD/XAUUSD/USOIL")
    ap.add_argument("--max", type=int, default=1, help="max tickets to close this run")
    ap.add_argument("--confirm", action="store_true", help="click final confirm button (REAL close)")
    ap.add_argument("--verify", action="store_true", help="verify each ticket disappears after confirm")
    ap.add_argument("--timeout", type=float, default=20.0, help="verify timeout seconds")
    ap.add_argument("--continue-on-failure", action="store_true", help="keep going if one ticket fails")
    args = ap.parse_args()

    cdp = os.environ.get("SMART_TRADING_CDP_URL", "").strip() or "http://localhost:9222"
    symbol = args.symbol.upper().strip()
    max_n = max(1, int(args.max))

    fetcher = DataFetcher(headless=False)
    ok = await fetcher.connect()
    if not ok:
        print(f"connect failed (CDP={cdp})")
        print("Hint: run start_chrome_debug.bat and login first; use http://localhost:9222 for IPv6 loopback.")
        return 2

    try:
        positions = await fetcher.get_open_positions()
        tickets = _collect_tickets(positions, symbol)
        print(f"open positions: {len(positions or [])}, {symbol} tickets: {len(tickets)}")
        if tickets:
            print("tickets:", ", ".join(tickets))

        if not tickets:
            print("Nothing to do.")
            return 0

        if not args.confirm:
            print("SAFE: no --confirm, no clicks were performed.")
            return 0

        page = fetcher.page
        if not page:
            print("no page object")
            return 2

        data_dir = Path(__file__).parent / "trade_data"
        data_dir.mkdir(exist_ok=True)

        closed = 0
        for t in tickets[:max_n]:
            print(f"\nclosing ticket #{t} ...")
            x_ok = await fetcher.close_position_by_ticket(t)
            print(f"  click_x={x_ok}")
            if not x_ok:
                if not args.continue_on_failure:
                    return 1
                continue

            await asyncio.sleep(0.4)
            c_ok = await _click_confirm_best_effort(page)
            print(f"  confirm_clicked={c_ok}")
            if not c_ok:
                # Likely: market closed / confirm disabled / modal not shown.
                try:
                    shot = data_dir / f"close_ticket_{t}_confirm_failed_{int(time.time())}.png"
                    await page.screenshot(path=str(shot), full_page=False)
                    print(f"  saved screenshot: {shot}")
                except Exception:
                    pass
                if not args.continue_on_failure:
                    return 1
                continue

            if args.verify:
                v = await _wait_ticket_gone(fetcher, t, timeout=float(args.timeout))
                print(f"  verify_gone={v}")
                if not v:
                    try:
                        shot = data_dir / f"close_ticket_{t}_verify_failed_{int(time.time())}.png"
                        await page.screenshot(path=str(shot), full_page=False)
                        print(f"  saved screenshot: {shot}")
                    except Exception:
                        pass
                    if not args.continue_on_failure:
                        return 1
                    continue

            closed += 1
            await asyncio.sleep(0.6)

        print(f"\nDone. closed={closed}/{min(max_n, len(tickets))} (symbol={symbol})")
        return 0
    finally:
        try:
            await fetcher.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

