"""Tests for the indicator precompute cache (engine #5 of the offline-evaluation roadmap).

Constructed bars only. The load-bearing property is EQUIVALENCE: a cached series must be identical to
the direct ``src.indicators`` call — the cache may change *when* work happens, never *what the answer
is*. Also pins that a repeat request is a genuine cache hit (no recompute) and that the channel series
are look-ahead-safe.
"""

from datetime import datetime, timedelta, timezone

import random

import pytest

from src.data_loader import OHLCVBar
from src.indicators import (
    average_true_range,
    exponential_moving_average,
    relative_strength_index,
    simple_moving_average,
)
from src.indicator_cache import IndicatorCache, donchian_high, donchian_low


def _bars(n: int = 200, seed: int = 5) -> list[OHLCVBar]:
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rng = random.Random(seed)
    out: list[OHLCVBar] = []
    p = 100.0
    for i in range(n):
        p2 = p * (1 + rng.gauss(0, 0.01))
        hi = max(p, p2) * (1 + abs(rng.gauss(0, 0.002)))
        lo = min(p, p2) * (1 - abs(rng.gauss(0, 0.002)))
        out.append(OHLCVBar(timestamp=t0 + timedelta(hours=i), open=p, high=hi, low=lo,
                            close=p2, volume=100.0))
        p = p2
    return out


# --- equivalence to the tested direct computation -----------------------------


def test_cache_matches_direct_indicators():
    bars = _bars()
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    c = IndicatorCache(bars)
    assert c.sma(20) == simple_moving_average(closes, 20)
    assert c.ema(12) == exponential_moving_average(closes, 12)
    assert c.rsi(2) == relative_strength_index(closes, 2)
    assert c.rsi(14) == relative_strength_index(closes, 14)
    assert c.atr(14) == average_true_range(highs, lows, closes, 14)


# --- memoisation: computed once, reused thereafter ----------------------------


def test_repeat_request_is_a_cache_hit():
    c = IndicatorCache(_bars())
    first = c.rsi(2)
    assert c.computations == 1
    second = c.rsi(2)
    assert c.computations == 1  # no recompute
    assert first is second  # the identical stored object is returned


def test_distinct_params_each_compute_once():
    c = IndicatorCache(_bars())
    c.sma(10)
    c.sma(20)
    c.sma(10)  # hit
    c.ema(10)
    assert c.computations == 3  # sma10, sma20, ema10


# --- donchian channels: correctness + look-ahead safety -----------------------


def test_donchian_high_low_values():
    highs = [1.0, 3.0, 2.0, 5.0, 4.0]
    lows = [1.0, 0.5, 0.8, 0.2, 0.9]
    assert donchian_high(highs, 3) == [None, None, 3.0, 5.0, 5.0]
    assert donchian_low(lows, 3) == [None, None, 0.5, 0.2, 0.2]


def test_donchian_window_ends_at_index_no_lookahead():
    # index i must not depend on any bar after i: a spike at the last bar cannot change earlier values.
    highs = [1.0, 2.0, 3.0, 4.0, 99.0]
    got = donchian_high(highs, 2)
    assert got[:4] == [None, 2.0, 3.0, 4.0]
    assert got[4] == 99.0


def test_donchian_via_cache_matches_free_function():
    bars = _bars()
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    c = IndicatorCache(bars)
    assert c.donchian_high(20) == donchian_high(highs, 20)
    assert c.donchian_low(20) == donchian_low(lows, 20)


def test_donchian_rejects_bad_period():
    with pytest.raises(ValueError):
        donchian_high([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        donchian_low([1.0, 2.0], -1)
