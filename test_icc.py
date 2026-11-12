"""Offline sanity tests for icc.py using synthetic price paths. Run: python test_icc.py"""
import numpy as np
import pandas as pd

from icc import detect


def build(waypoints, seed=1):
    """Piecewise-linear 15m close path through (n_bars, target_price) waypoints."""
    rng = np.random.default_rng(seed)
    closes, price = [], waypoints[0][1]
    for n, target in waypoints[1:]:
        closes.extend(np.linspace(price, target, n, endpoint=False))
        price = target
    closes.append(price)
    c = np.array(closes) + rng.normal(0, 0.03, len(closes))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) + 0.08
    l = np.minimum(o, c) - 0.08
    idx = pd.date_range("2026-01-05 00:00", periods=len(c), freq="15min", tz="UTC")
    m15 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)
    h1 = m15.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    # drop the final (possibly partial) hour from 1H so only complete bars are used
    last_full = m15.index[-1].floor("1h")
    h1 = h1[h1.index < last_full]
    return h1, m15


def range_phase(bars=240):
    pts, price = [(0, 100.0)], 100.0
    for i in range(bars // 8):
        pts.append((4, 101.8 if i % 2 == 0 else 100.2))
    return pts


def long_setup():
    wp = range_phase()
    wp += [
        (12, 110.0),    # indication leg
        (12, 102.4),    # correction down to key zone (touch 1)
        (6, 103.6),     # bounce
        (6, 102.35),    # touch 2
        (4, 103.2),     # bounce
        (3, 102.9),     # small dip
        (3, 103.9),     # continuation break
    ]
    return wp


def test_long():
    h1, m15 = build(long_setup())
    s = detect("TEST", h1, m15)
    assert s is not None, "expected a long setup"
    assert s.side == "long", s
    assert s.stop < s.entry < s.target, s
    print("long OK   ", s.to_dict())


def test_short_mirror():
    wp = [(n, 200 - p) for n, p in long_setup()]
    h1, m15 = build(wp)
    s = detect("TEST", h1, m15)
    assert s is not None, "expected a short setup"
    assert s.side == "short", s
    assert s.target < s.entry < s.stop, s
    print("short OK  ", s.to_dict())


def test_no_trade_in_range():
    h1, m15 = build(range_phase(300))
    assert detect("TEST", h1, m15) is None
    print("range OK  (no trade in consolidation)")


def test_requires_second_touch():
    # Only one touch of the zone -> should not signal with min_touches=2
    wp = range_phase() + [(12, 110.0), (12, 102.4), (6, 103.6), (4, 103.2), (3, 102.9), (3, 103.9)]
    h1, m15 = build(wp)
    assert detect("TEST", h1, m15, {"min_touches": 2}) is None
    print("touch OK  (single touch rejected)")


if __name__ == "__main__":
    test_long()
    test_short_mirror()
    test_no_trade_in_range()
    test_requires_second_touch()
    print("all tests passed")
