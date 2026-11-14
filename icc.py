"""
ICC (Indication -> Correction -> Continuation) setup detector.

Pure pandas/numpy: no broker or network code, so it can be tested offline.

Logic (long side; shorts are detected by mirroring the price series):
  1. INDICATION  : a 1H candle CLOSES above a prior confirmed swing high.
                   That swing high becomes the key level. The impulse leg must
                   be at least `impulse_atr` x ATR tall.
  2. CORRECTION  : price pulls back toward the key level without closing
                   below the zone and without taking out the impulse low.
  3. CONTINUATION: on 15m, price touches the key zone `min_touches` times
                   (bounces in between), then a 15m candle closes above the
                   post-touch high -> entry signal.
Stop = below correction low. Target = indication peak (previous liquidity).
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

DEFAULTS = {
    "swing_n": 2,            # bars each side to confirm a swing point
    "lookback_bars": 40,     # how far back (1H bars) a key level may be
    "impulse_atr": 1.5,      # min indication leg height in ATRs
    "zone_atr": 0.5,         # half-width of the key-level zone in ATRs
    "bounce_atr": 0.3,       # bounce needed between touches, in ATRs
    "min_retrace": 0.30,     # correction depth as fraction of indication leg
    "max_retrace": 0.85,
    "min_touches": 2,        # "2nd disrespect" rule; set 1 for more signals
    "stop_atr": 0.25,        # buffer beyond correction extreme
    "min_rr": 1.5,
    "max_risk_atr": 2.5,     # skip if stop is wider than this many ATRs
    "require_mss": False,    # True = only trend-reversal (MSS) indications
    "allow_shorts": True,
}


@dataclass
class Setup:
    symbol: str
    side: str            # "long" or "short"
    level: float         # key support/resistance level
    peak: float          # indication extreme
    leg_extreme: float   # origin of indication leg
    corr_extreme: float  # correction low (long) / high (short)
    entry: float
    stop: float
    target: float
    touches: int
    retrace: float
    atr: float
    rr: float

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def atr_last(df: pd.DataFrame, n: int = 14) -> float:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def swing_points(df: pd.DataFrame, n: int):
    """Indices of confirmed swing highs / lows (fractals)."""
    H, L = df["high"].values, df["low"].values
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if H[i] >= H[i - n:i + n + 1].max() and H[i] > H[i - n:i].max():
            highs.append(i)
        if L[i] <= L[i - n:i + n + 1].min() and L[i] < L[i - n:i].min():
            lows.append(i)
    return highs, lows


def count_touches(m: pd.DataFrame, zone_hi: float, bounce: float):
    """Count distinct touches of the zone; a new touch needs a real bounce first."""
    touches, last_idx, armed, run_max = 0, -1, True, -np.inf
    for i, (lo, hi) in enumerate(zip(m["low"].values, m["high"].values)):
        if lo <= zone_hi:
            if armed:
                touches += 1
                armed = False
            last_idx, run_max = i, -np.inf
        else:
            run_max = max(run_max, hi)
            if run_max >= zone_hi + bounce:
                armed = True
    return touches, last_idx


def _find_long(h1: pd.DataFrame, m15: pd.DataFrame, cfg: dict, why: dict | None = None):
    """Long-side search. If `why` is given it is filled with the furthest stage reached and a
    plain-English reason, so the bot's diary can explain a no-trade."""
    def mark(stage: int, msg: str):
        if why is not None and stage > why.get("stage", -1):
            why["stage"], why["msg"] = stage, msg

    n = cfg["swing_n"]
    if len(h1) < 30 or len(m15) < 10:
        mark(0, "not enough bars yet")
        return None
    atr = atr_last(h1)
    if not np.isfinite(atr) or atr <= 0:
        mark(0, "ATR unavailable")
        return None
    zone = cfg["zone_atr"] * atr
    sh, _ = swing_points(h1, n)
    C, H, L = h1["close"].values, h1["high"].values, h1["low"].values
    mark(0, "no recent breakout above a swing high (no indication)")

    for pos in range(len(sh) - 1, -1, -1):
        s = sh[pos]
        if s < len(h1) - cfg["lookback_bars"]:
            break
        level = H[s]
        k = next((i for i in range(s + 1, len(h1)) if C[i] > level), None)
        if k is None:
            continue
        if cfg["require_mss"] and pos > 0 and not (level < H[sh[pos - 1]]):
            mark(0, "breakout was not a trend reversal (MSS required)")
            continue
        peak_i = k + int(np.argmax(H[k:]))
        if peak_i >= len(h1) - 1:
            mark(1, "fresh breakout; correction has not started yet")
            continue
        peak = float(H[peak_i])
        leg_low = float(L[s:k + 1].min())
        if peak - leg_low < cfg["impulse_atr"] * atr:
            mark(1, f"breakout too weak (under {cfg['impulse_atr']} ATR)")
            continue

        post = h1.iloc[peak_i + 1:]
        corr_low = float(post["low"].min())
        if corr_low <= leg_low or (post["close"] < level - zone).any():
            mark(2, "pullback broke back through the key level; setup invalid")
            continue
        retrace = (peak - corr_low) / (peak - leg_low)
        if retrace < cfg["min_retrace"]:
            mark(2, f"pullback only {retrace:.0%} deep so far (needs {cfg['min_retrace']:.0%}+)")
            continue
        if retrace > cfg["max_retrace"]:
            mark(2, f"pullback {retrace:.0%} deep, too deep (max {cfg['max_retrace']:.0%})")
            continue

        t_start = h1.index[peak_i] + pd.Timedelta(hours=1)
        m = m15[m15.index >= t_start]
        if len(m) < 6:
            mark(3, "correction under way; waiting for more 15m bars")
            continue
        if (m["close"] < level - zone).any():
            mark(2, "15m closed through the key zone; setup invalid")
            continue
        zone_hi = level + zone
        touches, last_idx = count_touches(m, zone_hi, cfg["bounce_atr"] * atr)
        if touches < cfg["min_touches"] or last_idx < 0:
            mark(4, f"{touches}/{cfg['min_touches']} touches of the key zone; waiting")
            continue

        after = m.iloc[last_idx + 1:]
        if len(after) < 3:
            mark(5, "touched the zone; waiting for the bounce to develop")
            continue
        trigger = float(after["high"].iloc[:-1].max())
        last = after.iloc[-1]
        if not last["close"] > trigger:
            mark(5, "zone touched enough; waiting for a 15m close above the trigger high")
            continue

        entry = float(last["close"])
        corr_ext = min(corr_low, float(m["low"].min()))
        stop = corr_ext - cfg["stop_atr"] * atr
        risk, reward = entry - stop, peak - entry
        if risk <= 0 or risk > cfg["max_risk_atr"] * atr:
            mark(6, f"trigger fired but stop too wide ({risk / atr:.1f} ATR, max {cfg['max_risk_atr']})")
            continue
        rr = reward / risk
        if rr < cfg["min_rr"]:
            mark(6, f"trigger fired but reward/risk {rr:.1f} is under {cfg['min_rr']}")
            continue
        return Setup("", "long", float(level), peak, leg_low, corr_ext, entry, stop, peak,
                     touches, float(retrace), float(atr), float(rr))
    return None


def _invert(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["open"], out["close"] = -df["open"], -df["close"]
    out["high"], out["low"] = -df["low"], -df["high"]
    return out


def detect(symbol: str, h1: pd.DataFrame, m15: pd.DataFrame, cfg: dict | None = None,
           why: dict | None = None):
    """Return a Setup (long or short) or None. Data must be completed bars only.
    Pass a dict as `why` to receive {"stage", "msg", "side"}: how far the best candidate got."""
    cfg = {**DEFAULTS, **(cfg or {})}
    wl = {} if why is not None else None
    s = _find_long(h1, m15, cfg, wl)
    if s:
        s.symbol = symbol
        return s
    ws = None
    if cfg["allow_shorts"]:
        ws = {} if why is not None else None
        s = _find_long(_invert(h1), _invert(m15), cfg, ws)
        if s:
            s.symbol, s.side = symbol, "short"
            for f in ("level", "peak", "leg_extreme", "corr_extreme", "entry", "stop", "target"):
                setattr(s, f, -getattr(s, f))
            return s
    if why is not None:
        best, side = (wl or {}), "long"
        if ws and ws.get("stage", -1) > best.get("stage", -1):
            best, side = ws, "short"
        why.update(stage=best.get("stage", 0), msg=best.get("msg", "no setup"), side=side)
    return None
