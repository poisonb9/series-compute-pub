from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.data_loader import OHLCVBar


class SyntheticRegime(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    MIXED = "mixed"


@dataclass(frozen=True)
class SyntheticDataConfig:
    name: str
    regime: SyntheticRegime
    bars_count: int = 120
    start_price: float = 100.0
    start_timestamp: datetime | None = None
    timeframe_minutes: int = 60
    volume: float = 1000.0


def generate_synthetic_ohlcv(config: SyntheticDataConfig) -> list[OHLCVBar]:
    _validate_config(config)

    start_timestamp = config.start_timestamp or datetime(2026, 1, 1)
    closes = _generate_closes(config)
    bars: list[OHLCVBar] = []
    previous_close = config.start_price

    for index, close in enumerate(closes):
        open_price = previous_close if index else config.start_price
        wick = _wick_size(config.regime, index)
        high = max(open_price, close) + wick
        low = max(0.01, min(open_price, close) - wick)

        bars.append(
            OHLCVBar(
                timestamp=start_timestamp
                + timedelta(minutes=config.timeframe_minutes * index),
                open=round(open_price, 6),
                high=round(high, 6),
                low=round(low, 6),
                close=round(close, 6),
                volume=round(config.volume + (index % 7) * 3.0, 6),
            )
        )
        previous_close = close

    return bars


def generate_default_regime_datasets(
    bars_count: int = 120,
    start_price: float = 100.0,
) -> dict[str, list[OHLCVBar]]:
    datasets: dict[str, list[OHLCVBar]] = {}
    for regime in SyntheticRegime:
        config = SyntheticDataConfig(
            name=regime.value,
            regime=regime,
            bars_count=bars_count,
            start_price=start_price,
        )
        datasets[regime.value] = generate_synthetic_ohlcv(config)
    return datasets


def _validate_config(config: SyntheticDataConfig) -> None:
    if not config.name.strip():
        raise ValueError("name cannot be empty.")
    if not isinstance(config.regime, SyntheticRegime):
        raise ValueError("regime must be a SyntheticRegime.")
    if config.bars_count <= 0:
        raise ValueError("bars_count must be greater than zero.")
    if config.start_price <= 0:
        raise ValueError("start_price must be greater than zero.")
    if config.timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be greater than zero.")
    if config.volume <= 0:
        raise ValueError("volume must be greater than zero.")


def _generate_closes(config: SyntheticDataConfig) -> list[float]:
    closes: list[float] = []
    price = config.start_price

    for index in range(config.bars_count):
        if config.regime is SyntheticRegime.TREND_UP:
            price = config.start_price + (index * 0.32) + _small_wave(index)
        elif config.regime is SyntheticRegime.TREND_DOWN:
            price = config.start_price - (index * 0.26) + _small_wave(index)
        elif config.regime is SyntheticRegime.SIDEWAYS:
            price = config.start_price + _sideways_wave(index)
        elif config.regime is SyntheticRegime.VOLATILE:
            price = config.start_price + _volatile_wave(index)
        elif config.regime is SyntheticRegime.MIXED:
            price = _mixed_close(config.start_price, index, config.bars_count)

        closes.append(max(0.01, price))

    return closes


def _small_wave(index: int) -> float:
    return ((index % 5) - 2) * 0.08


def _sideways_wave(index: int) -> float:
    pattern = (0.0, 1.2, -0.9, 0.7, -1.1, 0.4)
    return pattern[index % len(pattern)]


def _volatile_wave(index: int) -> float:
    direction = 1 if index % 2 == 0 else -1
    amplitude = 2.2 + (index % 5) * 0.55
    drift = ((index % 9) - 4) * 0.12
    return (direction * amplitude) + drift


def _mixed_close(start_price: float, index: int, bars_count: int) -> float:
    first_cut = bars_count // 3
    second_cut = (bars_count * 2) // 3

    if index < first_cut:
        return start_price + (index * 0.34) + _small_wave(index)
    if index < second_cut:
        anchor = start_price + (first_cut * 0.34)
        return anchor + _sideways_wave(index)

    anchor = start_price + (first_cut * 0.34)
    return anchor - ((index - second_cut) * 0.42) + _small_wave(index)


def _wick_size(regime: SyntheticRegime, index: int) -> float:
    if regime is SyntheticRegime.VOLATILE:
        return 1.0 + (index % 3) * 0.25
    if regime is SyntheticRegime.SIDEWAYS:
        return 0.35
    return 0.55
