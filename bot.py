"""
ICC swing/day trading bot  (Alpaca + FMP + OpenAI, run by GitHub Actions)

Per run (every ~15 min during market hours):
  1. Safety: market open? not too close to open/close? paper-mode guard.
  2. Day filter: Alpaca calendar (early closes), your blackout dates (FOMC/CPI/jobs), and the
     FMP free-plan earnings calendar (earnings yesterday/today/tomorrow).
  3. For each watchlist symbol: pull 1H + 15m bars (Alpaca) and run the rule-based
     ICC detector (icc.py).
  4. OpenAI reviews each rule-based setup (+ Alpaca news headlines) and can only APPROVE or VETO.
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
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest

import icc
from diary import Diary

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


# ----------------------------------------------------------------------------- day filter (Alpaca calendar + FMP)
def fmp(path: str, **params):
    """Call Financial Modeling Prep's /stable API. Raises on HTTP errors and on 200-with-error bodies."""
    r = requests.get(
        f"https://financialmodelingprep.com/stable/{path}",
        params={**params, "apikey": need("FMP_API_KEY")},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and ("Error Message" in data or "error" in data):
        raise RuntimeError(str(data.get("Error Message") or data.get("error"))[:200])
    return data


def session_check(trading: TradingClient, today: dt.date) -> tuple[bool, str]:
    """Holiday / early-close check from Alpaca's own market calendar (free, no extra key)."""
    cal = trading.get_calendar(GetCalendarRequest(start=today, end=today))
    if not cal or str(cal[0].date)[:10] != today.isoformat():
        return False, "not a trading day"
    close = cal[0].close
    close_t = close.time() if isinstance(close, dt.datetime) else close
    if close_t < dt.time(16, 0):
        return False, f"early close today ({close_t})"
    return True, "ok"


def day_check(cfg: dict, today: dt.date, trading: TradingClient) -> tuple[bool, str]:
    """Return (ok_to_trade, reason). Fails closed if the session check cannot run."""
    ok, why = session_check(trading, today)
    if not ok:
        return False, why

    # High-impact US macro days (FOMC / CPI / jobs report ...).
    # FMP's free plan does not clearly include the economic calendar, so by default the bot uses
    # the dates you list in config.json -> filters.blackout_dates. Set
    # filters.use_fmp_economic_calendar to true only if your FMP plan includes that endpoint.
    d = today.isoformat()
    if cfg["filters"].get("use_fmp_economic_calendar", False):
        try:
            events = fmp("economic-calendar", **{"from": d, "to": d})
            hits = [e for e in events
                    if str(e.get("date", ""))[:10] == d
                    and e.get("country") == "US"
                    and str(e.get("impact", "")).lower() == "high"]
            if hits:
                return False, "high-impact US event: " + ", ".join(e.get("event", "?") for e in hits[:3])
        except (requests.HTTPError, RuntimeError) as e:
            log.warning("FMP economic calendar unavailable (%s); using config blackout_dates", e)
    if d in cfg["filters"]["blackout_dates"]:
        return False, "date is in config blackout_dates (FOMC/CPI/NFP/etc.)"
    return True, "ok"


def earnings_blocked(symbols: list[str], today: dt.date, window_days: int,
                     manual: dict | None = None) -> set[str]:
    """Symbols with earnings within +/- window_days.
    FMP's free plan only covers a limited list of symbols, so `manual` (config.json ->
    filters.manual_earnings_dates, e.g. {"BAC": ["2027-01-14"]}) is merged in as a backstop."""
    lo, hi = today - dt.timedelta(days=window_days), today + dt.timedelta(days=window_days)
    cal = fmp("earnings-calendar", **{"from": lo.isoformat(), "to": hi.isoformat()})
    out = {c["symbol"] for c in cal if c.get("symbol") in symbols}
    for sym, dates in (manual or {}).items():
        if sym in symbols and any(lo.isoformat() <= d <= hi.isoformat() for d in dates):
            out.add(sym)
    return out


def headlines(news_client: NewsClient, symbol: str, n: int = 5) -> list[str]:
    """Recent headlines (Alpaca news API, included with your Alpaca keys) as context for the AI
    review. Best effort: never blocks a trade."""
    try:
        ns = news_client.get_news(NewsRequest(symbols=symbol, limit=n))
        return [x.headline for x in ns.data.get("news", [])[:n]]
    except Exception as e:
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
def end_of_day_snapshot(trading: TradingClient, diary: Diary):
    start = dt.datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=start, limit=200)):
        if (o.client_order_id or "").startswith("icc-") and o.filled_qty and float(o.filled_qty) > 0:
            diary.eod.append(f"Filled today: {o.side.value} {o.filled_qty} {o.symbol} @ {o.filled_avg_price}")
    for p in trading.get_all_positions():
        diary.eod.append(f"Open: {p.qty} {p.symbol}, entry {float(p.avg_entry_price):.2f}, "
                         f"unrealized P/L ${float(p.unrealized_pl):,.2f}")
    if not diary.eod:
        diary.eod.append("No fills and no open positions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="analyse and log, but place no orders")
    args = ap.parse_args()
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    diary = Diary(paper, args.dry_run)
    try:
        run(args, paper, diary)
    except Exception as e:
        diary.result(f"Run crashed: {type(e).__name__}: {str(e)[:200]}")
        raise
    finally:
        diary.flush()


def run(args, paper: bool, diary: Diary):
    cfg = load_config()
    if not paper and os.getenv("LIVE_CONFIRM") != "I_UNDERSTAND_THE_RISKS":
        log.error("Live trading requires LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS. Refusing to run.")
        diary.silent = True
        sys.exit(1)

    trading = TradingClient(need("ALPACA_API_KEY"), need("ALPACA_SECRET_KEY"), paper=paper)
    data = StockHistoricalDataClient(need("ALPACA_API_KEY"), need("ALPACA_SECRET_KEY"))
    news = NewsClient(need("ALPACA_API_KEY"), need("ALPACA_SECRET_KEY"))
    ai = OpenAI(api_key=need("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    log.info("mode=%s dry_run=%s model=%s", "PAPER" if paper else "LIVE", args.dry_run, model)

    # --- market-time guards
    clock = trading.get_clock()
    if not clock.is_open:
        log.info("Market closed. Nothing to do.")
        diary.silent = True          # no diary noise outside market hours
        return
    now_et = dt.datetime.now(ET)
    mins_since_open = (now_et - now_et.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60
    mins_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60
    if mins_to_close < 20:
        end_of_day_snapshot(trading, diary)
    if mins_since_open < cfg["filters"]["skip_first_minutes"]:
        msg = f"Within the first {cfg['filters']['skip_first_minutes']} min of the open; waiting for it to settle."
        log.info(msg)
        diary.result("Skipped: " + msg)
        return
    if mins_to_close < cfg["filters"]["no_new_trades_last_minutes"]:
        msg = f"Within the last {cfg['filters']['no_new_trades_last_minutes']} min of the session; no new entries."
        log.info(msg)
        diary.result("Skipped: " + msg)
        return

    # --- day filter (fails closed)
    today = now_et.date()
    try:
        ok, why = day_check(cfg, today, trading)
        blocked = earnings_blocked(cfg["watchlist"], today, cfg["filters"]["earnings_window_days"],
                                   cfg["filters"].get("manual_earnings_dates"))
    except Exception as e:
        log.error("Day-filter check failed (%s). Failing closed: no trades this run.", e)
        record({"event": "day_filter_failure", "error": str(e)})
        diary.result(f"Skipped: day-filter data failed ({type(e).__name__}); failing closed, no trades.")
        diary.fact(f"Error: {str(e)[:200]}")
        return
    if not ok:
        log.info("Skipping day: %s", why)
        record({"event": "day_blocked", "reason": why})
        diary.result(f"Skipped the whole day: {why}")
        return
    diary.fact("Day filter: clear (open session, not a blackout date)")
    if blocked:
        log.info("Earnings filter excludes: %s", sorted(blocked))
        diary.fact(f"Earnings filter excluded: {', '.join(sorted(blocked))}")

    # --- account state
    acct = trading.get_account()
    equity = float(acct.equity)
    held = {p.symbol for p in trading.get_all_positions()}
    working = {o.symbol for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200))}
    busy = held | working
    slots = cfg["risk"]["max_open_positions"] - len(held)
    trades_left = cfg["risk"]["max_trades_per_day"] - todays_bot_orders(trading)
    log.info("equity=%.2f open_positions=%d slots=%d trades_left_today=%d", equity, len(held), slots, trades_left)
    diary.fact(f"Account: equity ${equity:,.2f}, {len(held)} open position(s), "
               f"{max(slots, 0)} slot(s) and {max(trades_left, 0)} trade(s) left today")
    if slots <= 0 or trades_left <= 0:
        diary.result("No new trades: " + ("position limit reached" if slots <= 0 else "daily trade limit reached"))

    # --- scan
    for sym in cfg["watchlist"]:
        if slots <= 0 or trades_left <= 0:
            break
        if sym in blocked:
            diary.note(sym, "skipped, earnings window")
            continue
        if sym in busy:
            diary.note(sym, "skipped, already holding it or has an open order")
            continue
        try:
            h1 = get_bars(data, sym, TimeFrame(1, TimeFrameUnit.Hour), 20, pd.Timedelta(hours=1))
            m15 = get_bars(data, sym, TimeFrame(15, TimeFrameUnit.Minute), 6, pd.Timedelta(minutes=15))
            if len(h1) < 30 or len(m15) < 10:
                diary.note(sym, f"skipped, not enough data from Alpaca ({len(h1)} 1H / {len(m15)} 15m bars)")
                continue
            why_not: dict = {}
            setup = icc.detect(sym, h1, m15, cfg["strategy"], why_not)
            if not setup:
                if why_not.get("stage", 0) >= 1:
                    diary.note(sym, f"no entry: {why_not['msg']}")
                else:
                    diary.quiet(sym)
                continue
            # only act on a *fresh* continuation trigger (last completed 15m bar)
            if (pd.Timestamp.now(tz="UTC") - (m15.index[-1] + pd.Timedelta(minutes=15))) > pd.Timedelta(minutes=20):
                diary.note(sym, "setup found but stale (trigger candle is more than 20 min old); not chasing")
                continue
            log.info("SETUP %s", setup.to_dict())
            desc = (f"entry ~{setup.entry:.2f}, stop {setup.stop:.2f}, "
                    f"target {setup.target:.2f}, R:R {setup.rr:.1f}, {setup.touches} touches")

            if setup.side == "short":
                if not cfg["strategy"]["allow_shorts"] or not trading.get_asset(sym).shortable:
                    diary.note(sym, f"{desc}; skipped, shorting not allowed or asset not shortable")
                    continue

            review = ai_review(ai, model, setup, m15, headlines(news, sym))
            approved = bool(review.get("approve")) and int(review.get("confidence", 0)) >= cfg["ai"]["min_confidence"]
            record({"event": "setup", "setup": setup.to_dict(), "ai": review, "approved": approved})
            ai_txt = f"AI {'approved' if approved else 'vetoed'} ({review.get('confidence')}%): {review.get('reason')}"
            if not approved:
                log.info("AI vetoed %s: %s", sym, review.get("reason"))
                diary.note(sym, f"{desc}. {ai_txt}")
                continue

            qty = position_size(equity, setup.entry, setup.stop,
                                cfg["risk"]["risk_per_trade_pct"], cfg["risk"]["max_position_pct"])
            if qty < 1:
                log.info("Position size < 1 share for %s; skipping.", sym)
                diary.note(sym, f"{desc}. {ai_txt}. Skipped: position size under 1 share at current risk limits")
                continue

            if args.dry_run:
                log.info("[DRY RUN] would %s %d %s  stop=%.2f target=%.2f", setup.side, qty, sym, setup.stop, setup.target)
                record({"event": "dry_run_order", "symbol": sym, "qty": qty, "setup": setup.to_dict()})
                diary.trade(f"**{sym}**: WOULD {setup.side.upper()} {qty} sh. {desc}. {ai_txt}")
            else:
                o = submit(trading, setup, qty, today)
                log.info("ORDER SUBMITTED %s id=%s", sym, o.id)
                record({"event": "order", "symbol": sym, "qty": qty, "order_id": str(o.id), "setup": setup.to_dict()})
                diary.trade(f"**{sym}**: {setup.side.upper()} {qty} sh. {desc}. {ai_txt}")
            slots -= 1
            trades_left -= 1
        except Exception as e:
            log.exception("Error processing %s: %s", sym, e)
            record({"event": "error", "symbol": sym, "error": str(e)})
            diary.note(sym, f"error while processing: {str(e)[:150]}")


if __name__ == "__main__":
    main()
