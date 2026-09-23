"""Indicator precompute cache (engine #5 of the offline-evaluation roadmap).

A single evaluation runs the same detector across many parameter passes, and each pass otherwise
recomputes RSI/EMA/SMA/ATR/channels from scratch — O(passes x n) wasted work. This cache computes each
(indicator, period) series ONCE per dataset and hands back the memoised result on every later call.

The one rule that makes it safe: **it never computes anything itself.** Every series is produced by the
already-tested functions in ``src.indicators`` (and the same look-ahead-safe channel convention used by
the detectors), so a cached value is byte-identical to the direct call. The cache changes *when* work
happens, never *what the answer is* — cheapness from automation, never from cutting corners.

Deterministic, stdlib + repo indicators only. Research/offline: it opens no data, assigns no role, and
authorizes nothing. Built and tested on constructed bars (Tier 1).
"""

from __future__ import annotations

from src.data_loader import OHLCVBar
from src.indicators import (
    IndicatorSeries,
    average_true_range,
    exponential_moving_average,
    relative_strength_index,
    simple_moving_average,
)


def donchian_high(highs: list[float], period: int) -> IndicatorSeries:
    """Highest high of the ``period`` bars ENDING at (and including) each index.

    ``result[i]`` = max(highs[i-period+1 .. i]); None before the first full window. Look-ahead-safe:
    it uses only bars at or before ``i``. A detector wanting the *prior* channel reads ``result[i-1]``.
    """
    if period <= 0:
        raise ValueError("period must be greater than zero")
    out: IndicatorSeries = [None] * len(highs)
    for i in range(len(highs)):
        if i + 1 >= period:
            out[i] = max(highs[i - period + 1: i + 1])
    return out


def donchian_low(lows: list[float], period: int) -> IndicatorSeries:
    """Lowest low of the ``period`` bars ending at each index; None before the first full window."""
    if period <= 0:
        raise ValueError("period must be greater than zero")
    out: IndicatorSeries = [None] * len(lows)
    for i in range(len(lows)):
        if i + 1 >= period:
            out[i] = min(lows[i - period + 1: i + 1])
    return out


class IndicatorCache:
    """Per-dataset memoised indicator series over one immutable bar list.

    Each ``sma``/``ema``/``rsi``/``atr``/``donchian_*`` call computes on first use and returns the
    stored list thereafter. ``computations`` counts how many series were actually computed, so a test
    can prove a repeated request was a cache hit (no recompute)."""

    def __init__(self, bars: list[OHLCVBar]) -> None:
        self._bars = bars
        self._closes = [b.close for b in bars]
        self._highs = [b.high for b in bars]
        self._lows = [b.low for b in bars]
        self._cache: dict[tuple[str, int], IndicatorSeries] = {}
        self.computations = 0

    def _memo(self, key: tuple[str, int], compute) -> IndicatorSeries:
        if key not in self._cache:
            self._cache[key] = compute()
            self.computations += 1
        return self._cache[key]

    def sma(self, period: int) -> IndicatorSeries:
        return self._memo(("sma", period), lambda: simple_moving_average(self._closes, period))

    def ema(self, period: int) -> IndicatorSeries:
        return self._memo(("ema", period), lambda: exponential_moving_average(self._closes, period))

    def rsi(self, period: int = 14) -> IndicatorSeries:
        return self._memo(("rsi", period), lambda: relative_strength_index(self._closes, period))

    def atr(self, period: int = 14) -> IndicatorSeries:
        return self._memo(
            ("atr", period),
            lambda: average_true_range(self._highs, self._lows, self._closes, period),
        )

    def donchian_high(self, period: int) -> IndicatorSeries:
        return self._memo(("dch_high", period), lambda: donchian_high(self._highs, period))

    def donchian_low(self, period: int) -> IndicatorSeries:
        return self._memo(("dch_low", period), lambda: donchian_low(self._lows, period))
