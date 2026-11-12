"""
ICC swing/day trading bot  (Alpaca + Finnhub + OpenAI, run by GitHub Actions)

Per run (every ~15 min during market hours):
  1. Safety: market open? not too close to open/close? paper-mode guard.
  2. Finnhub day filter: skip early-close days, high-impact US macro days, and
     symbols with earnings yesterday/today/tomorrow.
  3. For each watchlist symbol: pull 1H + 15m bars (Alpaca) and run the rule-based
     ICC detector (icc.py).
  4. OpenAI reviews each rule-based setup (+ Finnhub headlines) and can only APPROVE or VETO.
     It never changes entry/stop/target. If the API fails, the trade is skipped.
  5. Alpaca bracket order (entry + stop + target), risk-based position size.

Usage:  python bot.py [--dry-run]
"""
import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from openai import OpenAI
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest

import icc

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
HERE = Path(__file__).parent
LOG_DIR = HERE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("icc-bot")


# ----------------------------------------------------------------------------- config
def load_config() -> dict:
    return json.loads((HERE / "config.json").read_text())


def need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        log.error("Missing required environment variable: %s", name)
        sys.exit(1)
    return v


def record(event: dict):
    event["ts"] = dt.datetime.now(UTC).isoformat()
    with open(LOG_DIR / "decisions.jsonl", "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


# ----------------------------------------------------------------------------- finnhub
def fh(path: str, **params):
    r = requests.get(
        f"https://finnhub.io/api/v1/{path}",
        params={**params, "token": need("FINNHUB_API_KEY")},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def day_check(cfg: dict, today: dt.date) -> tuple[bool, str]:
    """Return (ok_to_trade, reason). Fails closed if Finnhub holiday data is unreachable."""
    # 1) market holidays / early closes
    hol = fh("stock/market-holiday", exchange="US").get("data", [])
    for h in hol:
        if h.get("atDate") == today.isoformat():
            hours = h.get("tradingHour") or ""
            return False, f"market holiday / early close today: {h.get('eventName')} {hours}"

    # 2) high-impact US macro events (economic calendar is a premium Finnhub endpoint;
    #    fall back to the dates listed in config.json if the plan does not include it)
    d = today.isoformat()
    try:
        ev = fh("calendar/economic").get("economicCalendar", [])
        hits = [e for e in ev
                if str(e.get("time", ""))[:10] == d
                and e.get("country") == "US"
                and str(e.get("impact", "")).lower() == "high"]
        if hits:
            return False, "high-impact US event: " + ", ".join(e.get("event", "?") for e in hits[:3])
    except requests.HTTPError as e:
        log.warning("Finnhub economic calendar unavailable (%s); using config blackout_dates", e)
        if d in cfg["filters"]["blackout_dates"]:
            return False, "date is in config blackout_dates (FOMC/CPI/NFP/etc.)"
    return True, "ok"


def earnings_blocked(symbols: list[str], today: dt.date, window_days: int) -> set[str]:
    lo, hi = today - dt.timedelta(days=window_days), today + dt.timedelta(days=window_days)
    cal = fh("calendar/earnings", **{"from": lo.isoformat(), "to": hi.isoformat()}).get("earningsCalendar", [])
    return {c["symbol"] for c in cal if c.get("symbol") in symbols}


def headlines(symbol: str, today: dt.date, n: int = 5) -> list[str]:
    try:
        items = fh("company-news", symbol=symbol,
                   **{"from": (today - dt.timedelta(days=2)).isoformat(), "to": today.isoformat()})
        return [i["headline"] for i in items[:n]]
    except Exception as e:  # news is context only; never block on it
        log.warning("news fetch failed for %s: %s", symbol, e)
        return []


# ----------------------------------------------------------------------------- alpaca data
def get_bars(data: StockHistoricalDataClient, symbol: str, tf: TimeFrame, days: int, bar_len: pd.Timedelta):
    now = dt.datetime.now(UTC)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=now - dt.timedelta(days=days),
                           end=now, feed=DataFeed.IEX)
    df = data.get_stock_bars(req).df
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df = df[["open", "high", "low", "close"]]
    return df[df.index + bar_len <= pd.Timestamp(now)]  # completed bars only


# ----------------------------------------------------------------------------- OpenAI gate
SYSTEM = """You are a risk reviewer for a rule-based ICC (Indication, Correction, Continuation)
trading bot. The bot has ALREADY found a setup using fixed rules. You may only APPROVE or VETO it.
Never propose different prices. Veto if: the headlines suggest a binary/event risk (earnings,
guidance, lawsuit, halt, M&A, FDA, offering), the setup looks like a low-quality/choppy range, or the
risk/reward is not credible. Approve if nothing argues against it. Reply with JSON only:
{"approve": true|false, "confidence": 0-100, "reason": "<one sentence>"}"""


def ai_review(client: OpenAI, model: str, setup: icc.Setup, recent: pd.DataFrame, news: list[str]) -> dict:
    payload = {
        "setup": setup.to_dict(),
        "last_12_15m_bars": recent.tail(12).round(2).reset_index().astype(str).to_dict("records"),
        "headlines": news,
    }
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": json.dumps(payload)}],
        timeout=30,
    )
    return json.loads(resp.choices[0].message.content)


# ----------------------------------------------------------------------------- orders
def position_size(equity: float, entry: float, stop: float, risk_pct: float, max_pos_pct: float) -> int:
    per_share = abs(entry - stop)
    if per_share <= 0:
        return 0
    by_risk = math.floor(equity * risk_pct / per_share)
    by_cap = math.floor(equity * max_pos_pct / entry)
    return max(0, min(by_risk, by_cap))


def todays_bot_orders(trading: TradingClient) -> int:
    start = dt.datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    orders = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=start, limit=200))
    return sum(1 for o in orders if (o.client_order_id or "").startswith("icc-"))


def submit(trading: TradingClient, s: icc.Setup, qty: int, today: dt.date):
    side = OrderSide.BUY if s.side == "long" else OrderSide.SELL
    order = MarketOrderRequest(
        symbol=s.symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(s.target, 2)),
        stop_loss=StopLossRequest(stop_price=round(s.stop, 2)),
        client_order_id=f"icc-{s.symbol}-{today.isoformat()}-{s.side}",  # duplicate-proof per day
    )
    return trading.submit_order(order)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="analyse and log, but place no orders")
    args = ap.parse_args()

    cfg = load_config()
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    if not paper and os.getenv("LIVE_CONFIRM") != "I_UNDERSTAND_THE_RISKS":
        log.error("Live trading requires LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS. Refusing to run.")
        sys.exit(1)

    trading = TradingClient(need("ALPACA_API_KEY"), need("ALPACA_SECRET_KEY"), paper=paper)
    data = StockHistoricalDataClient(need("ALPACA_API_KEY"), need("ALPACA_SECRET_KEY"))
    ai = OpenAI(api_key=need("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    log.info("mode=%s dry_run=%s model=%s", "PAPER" if paper else "LIVE", args.dry_run, model)

    # --- market-time guards
    clock = trading.get_clock()
    if not clock.is_open:
        log.info("Market closed. Nothing to do.")
        return
    now_et = dt.datetime.now(ET)
    mins_since_open = (now_et - now_et.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60
    mins_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60
    if mins_since_open < cfg["filters"]["skip_first_minutes"]:
        log.info("Within first %s min of the open; skipping.", cfg["filters"]["skip_first_minutes"])
        return
    if mins_to_close < cfg["filters"]["no_new_trades_last_minutes"]:
        log.info("Within last %s min of the session; no new entries.", cfg["filters"]["no_new_trades_last_minutes"])
        return

    # --- Finnhub day filter (fails closed)
    today = now_et.date()
    try:
        ok, why = day_check(cfg, today)
        blocked = earnings_blocked(cfg["watchlist"], today, cfg["filters"]["earnings_window_days"])
    except Exception as e:
        log.error("Finnhub check failed (%s). Failing closed: no trades this run.", e)
        record({"event": "finnhub_failure", "error": str(e)})
        return
    if not ok:
        log.info("Skipping day: %s", why)
        record({"event": "day_blocked", "reason": why})
        return
    if blocked:
        log.info("Earnings filter excludes: %s", sorted(blocked))

    # --- account state
    acct = trading.get_account()
    equity = float(acct.equity)
    held = {p.symbol for p in trading.get_all_positions()}
    working = {o.symbol for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200))}
    busy = held | working
    slots = cfg["risk"]["max_open_positions"] - len(held)
    trades_left = cfg["risk"]["max_trades_per_day"] - todays_bot_orders(trading)
    log.info("equity=%.2f open_positions=%d slots=%d trades_left_today=%d", equity, len(held), slots, trades_left)

    # --- scan
    for sym in cfg["watchlist"]:
        if slots <= 0 or trades_left <= 0:
            break
        if sym in blocked or sym in busy:
            continue
        try:
            h1 = get_bars(data, sym, TimeFrame(1, TimeFrameUnit.Hour), 20, pd.Timedelta(hours=1))
            m15 = get_bars(data, sym, TimeFrame(15, TimeFrameUnit.Minute), 6, pd.Timedelta(minutes=15))
            if len(h1) < 30 or len(m15) < 10:
                continue
            setup = icc.detect(sym, h1, m15, cfg["strategy"])
            if not setup:
                continue
            # only act on a *fresh* continuation trigger (last completed 15m bar)
            if (pd.Timestamp.now(tz="UTC") - (m15.index[-1] + pd.Timedelta(minutes=15))) > pd.Timedelta(minutes=20):
                continue
            log.info("SETUP %s", setup.to_dict())

            if setup.side == "short":
                if not cfg["strategy"]["allow_shorts"] or not trading.get_asset(sym).shortable:
                    continue

            review = ai_review(ai, model, setup, m15, headlines(sym, today))
            approved = bool(review.get("approve")) and int(review.get("confidence", 0)) >= cfg["ai"]["min_confidence"]
            record({"event": "setup", "setup": setup.to_dict(), "ai": review, "approved": approved})
            if not approved:
                log.info("AI vetoed %s: %s", sym, review.get("reason"))
                continue

            qty = position_size(equity, setup.entry, setup.stop,
                                cfg["risk"]["risk_per_trade_pct"], cfg["risk"]["max_position_pct"])
            if qty < 1:
                log.info("Position size < 1 share for %s; skipping.", sym)
                continue

            if args.dry_run:
                log.info("[DRY RUN] would %s %d %s  stop=%.2f target=%.2f", setup.side, qty, sym, setup.stop, setup.target)
                record({"event": "dry_run_order", "symbol": sym, "qty": qty, "setup": setup.to_dict()})
            else:
                o = submit(trading, setup, qty, today)
                log.info("ORDER SUBMITTED %s id=%s", sym, o.id)
                record({"event": "order", "symbol": sym, "qty": qty, "order_id": str(o.id), "setup": setup.to_dict()})
            slots -= 1
            trades_left -= 1
        except Exception as e:
            log.exception("Error processing %s: %s", sym, e)
            record({"event": "error", "symbol": sym, "error": str(e)})


if __name__ == "__main__":
    main()
