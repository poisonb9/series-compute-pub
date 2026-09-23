"""Basic technical indicators for research and simulation.

The functions in this module do not connect to exchanges, do not request
credentials, and do not execute orders. They are intentionally small and
explicit so indicator behavior can be audited in tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Number = int | float
IndicatorSeries = list[float | None]


def _validate_period(period: int) -> None:
    if not isinstance(period, int) or isinstance(period, bool):
        raise TypeError("period must be an integer")
    if period <= 0:
        raise ValueError("period must be greater than zero")


def _to_float_series(values: Sequence[Number], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")

    result: list[float] = []
    for index, value in enumerate(values):
        if value is None or isinstance(value, bool):
            raise ValueError(f"{name}[{index}] must be a finite number")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be a finite number") from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"{name}[{index}] must be a finite number")
        result.append(numeric_value)
    return result


def simple_moving_average(values: Sequence[Number], period: int) -> IndicatorSeries:
    """Return a simple moving average with None before the first full window."""

    _validate_period(period)
    data = _to_float_series(values, "values")
    averages: IndicatorSeries = []
    window_sum = 0.0

    for index, value in enumerate(data):
        window_sum += value
        if index >= period:
            window_sum -= data[index - period]
        if index + 1 < period:
            averages.append(None)
        else:
            averages.append(window_sum / period)

    return averages


def exponential_moving_average(values: Sequence[Number], period: int) -> IndicatorSeries:
    """Return an EMA seeded by the SMA of the first complete window."""

    _validate_period(period)
    data = _to_float_series(values, "values")
    if not data:
        return []

    averages: IndicatorSeries = [None] * len(data)
    if len(data) < period:
        return averages

    multiplier = 2 / (period + 1)
    ema = sum(data[:period]) / period
    averages[period - 1] = ema

    for index in range(period, len(data)):
        ema = (data[index] - ema) * multiplier + ema
        averages[index] = ema

    return averages


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    relative_strength = avg_gain / avg_loss
    return 100 - (100 / (1 + relative_strength))


def relative_strength_index(values: Sequence[Number], period: int = 14) -> IndicatorSeries:
    """Return Wilder's RSI with None until enough price changes exist."""

    _validate_period(period)
    data = _to_float_series(values, "values")
    rsi: IndicatorSeries = [None] * len(data)

    if len(data) <= period:
        return rsi

    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, period + 1):
        change = data[index] - data[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rsi[period] = _rsi_from_averages(avg_gain, avg_loss)

    for index in range(period + 1, len(data)):
        change = data[index] - data[index - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        rsi[index] = _rsi_from_averages(avg_gain, avg_loss)

    return rsi


def average_true_range(
    highs: Sequence[Number],
    lows: Sequence[Number],
    closes: Sequence[Number],
    period: int = 14,
) -> IndicatorSeries:
    """Return Wilder's Average True Range."""

    _validate_period(period)
    high_values = _to_float_series(highs, "highs")
    low_values = _to_float_series(lows, "lows")
    close_values = _to_float_series(closes, "closes")

    if not (len(high_values) == len(low_values) == len(close_values)):
        raise ValueError("highs, lows, and closes must have the same length")

    true_ranges: list[float] = []
    for index, high in enumerate(high_values):
        low = low_values[index]
        if high < low:
            raise ValueError(f"highs[{index}] must be greater than or equal to lows[{index}]")
        if index == 0:
            true_range = high - low
        else:
            previous_close = close_values[index - 1]
            true_range = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        true_ranges.append(true_range)

    atr: IndicatorSeries = [None] * len(true_ranges)
    if len(true_ranges) < period:
        return atr

    current_atr = sum(true_ranges[:period]) / period
    atr[period - 1] = current_atr

    for index in range(period, len(true_ranges)):
        current_atr = ((current_atr * (period - 1)) + true_ranges[index]) / period
        atr[index] = current_atr

    return atr


def on_balance_volume(closes: Sequence[Number], volumes: Sequence[Number]) -> list[float]:
    """Return On-Balance Volume using the first point as zero."""

    close_values = _to_float_series(closes, "closes")
    volume_values = _to_float_series(volumes, "volumes")

    if len(close_values) != len(volume_values):
        raise ValueError("closes and volumes must have the same length")
    for index, volume in enumerate(volume_values):
        if volume < 0:
            raise ValueError(f"volumes[{index}] must be greater than or equal to zero")

    if not close_values:
        return []

    obv = [0.0]
    current_obv = 0.0
    for index in range(1, len(close_values)):
        if close_values[index] > close_values[index - 1]:
            current_obv += volume_values[index]
        elif close_values[index] < close_values[index - 1]:
            current_obv -= volume_values[index]
        obv.append(current_obv)

    return obv
