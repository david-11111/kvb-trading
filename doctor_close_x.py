"""
One-command diagnostics for the “close via trailing X” flow.

It runs step-by-step checks and prints PASS/FAIL with actionable hints.
By default it will NOT click the final confirm button (safe).

Prereq:
  - Open the debug browser via start_chrome_debug.bat and login.
  - Ensure CDP endpoint works (usually http://localhost:9222).

Usage (PowerShell):
  cd "D:\\20240313整理文件\\Desktop\\老王说油推广\\smart_trading"
  $env:SMART_TRADING_CDP_URL="http://localhost:9222"
  python .\\doctor_close_x.py --symbol USOIL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request

from data_fetcher import DataFetcher


def _http_get_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data)
    except Exception:
        return None


def _print(step: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {step}"
    if detail:
        line += f" - {detail}"
    print(line)


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


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="USOIL")
    ap.add_argument("--ticket", default="", help="close a specific ticket, e.g. 67159382 (overrides --symbol)")
    ap.add_argument("--cdp", default="", help="override CDP url, e.g. http://localhost:9222")
    ap.add_argument("--confirm", action="store_true", help="click final confirm button (will actually close the position)")
    ap.add_argument("--verify", action="store_true", help="verify the position disappears after confirm")
    ap.add_argument("--timeout", type=float, default=12.0)
    args = ap.parse_args()

    cdp = args.cdp.strip() or os.environ.get("SMART_TRADING_CDP_URL", "").strip() or "http://localhost:9222"
    symbol = args.symbol.upper().strip()
    ticket = (args.ticket or "").strip().lstrip("#")

    ver = _http_get_json(f"{cdp.rstrip('/')}/json/version")
    _print("CDP endpoint reachable", bool(ver), f"{cdp}/json/version")
    if not ver:
        print("Hint: you must start Chrome with --remote-debugging-port, e.g. run start_chrome_debug.bat.")
        print("Hint: if you see only [::1]:9222 listening, use http://localhost:9222 (not 127.0.0.1).")
        return 2

    fetcher = DataFetcher(headless=False)
    ok = await fetcher.connect()
    _print("Playwright attached over CDP", bool(ok), cdp)
    if not ok:
        print("Hint: close other debug browsers, then re-run start_chrome_debug.bat.")
        return 2

    try:
        page = fetcher.page
        if not page:
            _print("Trade page found", False, "no page object")
            return 2

        url = ""
        try:
            url = page.url or ""
        except Exception:
            url = ""
        _print("Trade tab is open", ("mykvb.com/trade" in url), url or "(unknown)")
        if "mykvb.com/trade" not in (url or ""):
            print("Hint: keep a tab open at https://mykvb.com/trade in the debug browser.")

        # Ensure the holdings/positions tab is visible (best-effort).
        tab_ok = await fetcher.ensure_positions_tab()
        _print("Positions tab clicked (best-effort)", bool(tab_ok), "")

        # Try reading open positions from DOM/text parse.
        pos = await fetcher.get_open_positions()
        _print("Read open positions", bool(pos), f"count={len(pos or [])}")
        if pos:
            sample = ", ".join([p.get("symbol", "?") for p in pos[:6]])
            print(f"  positions sample: {sample}")
            # Also show a few tickets so you can copy/paste for --ticket tests.
            tickets = [str(p.get("ticket") or "").lstrip("#") for p in pos if p.get("ticket")]
            tickets = [t for t in tickets if t]
            if tickets:
                print(f"  tickets sample: {', '.join(tickets[:8])}")

        if ticket:
            ticket_exists = bool(pos) and any(str(p.get("ticket") or "").lstrip("#") == ticket for p in pos)
            if not ticket_exists:
                print(f"Hint: platform positions list does not include ticket #{ticket}. Is it still open?")
                if pos:
                    print("Hint: pick one from 'tickets sample' above and retry with --ticket <id>.")
                return 1

            # Ticket mode: directly test close-by-ticket (no need to pre-locate <tr> here).
            clicked_ok = await fetcher.close_position_by_ticket(ticket)
            _print("Click trailing X (no confirm)", bool(clicked_ok), "method=close_position_by_ticket")
            if not clicked_ok:
                print("Hint: ticket exists but X click failed; keep '持仓' table visible and try again.")
                return 1
            print("OK: X was clicked. You should see the close/confirm dialog on the page now.")
            if args.confirm:
                c = await _click_confirm_best_effort(page)
                _print("Click final confirm", bool(c), "")
                if args.verify:
                    deadline = time.time() + max(1.0, float(args.timeout))
                    gone = False
                    while time.time() < deadline:
                        try:
                            cur = await fetcher.get_open_positions()
                            if not any(str(p.get("ticket") or "").lstrip("#") == ticket for p in (cur or [])):
                                gone = True
                                break
                        except Exception:
                            pass
                        await asyncio.sleep(1.0)
                    _print("Verify position gone", bool(gone), f"timeout={args.timeout}s")
            else:
                print("SAFE: this script did NOT click the final confirm button.")
            return 0
        elif pos and not any((p.get("symbol", "") or "").upper() == symbol for p in pos):
            first = (pos[0].get("symbol") or "").upper() if pos else ""
            print(f"Hint: platform positions list does not include {symbol}. Is there an open position?")
            if first:
                print(f"Hint: try: python .\\doctor_close_x.py --symbol {first}")

        # Locate row(s) for the symbol.
        needle = f"#{ticket}" if ticket else symbol
        rows = page.locator("tr", has_text=needle)
        try:
            count = await rows.count()
        except Exception:
            count = 0
        _print("Locate holdings row by <tr> text", count > 0, f"needle={needle} rows={count}")
        if count == 0:
            print("Hint: the holdings table may be virtualized or not rendered as <tr>.")
            print("Hint: try clicking the '持仓' tab manually and keep it visible.")
            return 1

        # Try action chain on up to a few candidate rows.
        success = False
        for idx in range(min(int(count), 5)):
            row = rows.nth(idx)
            try:
                txt = (await row.inner_text(timeout=1200)).strip()
            except Exception:
                txt = ""
            snippet = (txt.replace("\n", " ")[:120] + ("…" if len(txt) > 120 else "")) if txt else "(no text)"
            try:
                box = await row.bounding_box()
            except Exception:
                box = None
            print(f"  candidate row[{idx}] y={(box or {}).get('y')} text={snippet}")

            # Reveal right-side actions (hover + horizontal scroll).
            reveal = await fetcher._scroll_row_actions_into_view(row)
            _print("Reveal row right actions", bool(reveal.get("ok")), f"hovered={reveal.get('hovered')} h_scrolled={reveal.get('h_scrolled')}")
            if reveal.get("container"):
                try:
                    print(f"  scroll container: {reveal.get('container')}")
                except Exception:
                    pass

            # Click X (no confirm).
            if ticket:
                clicked_ok = await fetcher.close_position_by_ticket(ticket)
                clicked = {"clicked": bool(clicked_ok), "method": "close_position_by_ticket", "ticket": ticket}
                _print("Click trailing X (no confirm)", bool(clicked_ok), "method=close_position_by_ticket")
                print(f"  click detail: {clicked}")
                if clicked_ok:
                    success = True
                    break
            else:
                clicked = await fetcher._click_close_x_in_row(row)
                _print("Click trailing X (no confirm)", bool(clicked.get("clicked")), f"method={clicked.get('method') or clicked.get('reason')}")
                if clicked.get("clicked"):
                    print(f"  click detail: {clicked}")
                    success = True
                    break
                else:
                    print(f"  click detail: {clicked}")

        if not success:
            print("Hint: if X is hidden behind the watchlist panel, the horizontal scroll container detection must be tightened.")
            print("Hint: send me the 'scroll container' + 'click detail' output above, I will tune the scroller.")
            return 1

        print("OK: X was clicked. You should see the close/confirm dialog on the page now.")
        if args.confirm:
            c = await _click_confirm_best_effort(page)
            _print("Click final confirm", bool(c), "")
            if args.verify:
                v = await _wait_position(fetcher, symbol, should_exist=False, timeout=float(args.timeout))
                _print("Verify position gone", bool(v), f"timeout={args.timeout}s")
        else:
            print("SAFE: this script did NOT click the final confirm button.")
        return 0
    finally:
        try:
            await fetcher.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    # Suppress noisy background TargetClosedError warnings when detaching from CDP.
    try:
        loop = asyncio.get_event_loop()
        old_handler = loop.get_exception_handler()

        def _handler(loop, context):
            msg = str(context.get("exception") or context.get("message") or "")
            if "TargetClosedError" in msg:
                return
            if old_handler:
                return old_handler(loop, context)
            loop.default_exception_handler(context)

        loop.set_exception_handler(_handler)
    except Exception:
        pass
    raise SystemExit(asyncio.run(main()))
