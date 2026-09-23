"""Tests for the measured per-hour spread table.

No data opened: the table is embedded. These pins guard the two things that can rot
silently — the table drifting out of shape, and the measured values drifting away from
the laudo they came from (`docs/MEDICAO_6_TABELA_DE_CUSTO_HORARIO_20260923.md`).
"""

import pytest

from src.execution_realism import RealismConfig, round_turn_cost_bps
from src.measured_spread import (
    LEGACY_FLAT_SPREAD_BPS,
    MEAN_SPREAD_BPS,
    config_for,
    costliest_hour,
    spread_bps,
)


SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY")


def test_every_symbol_has_all_24_hours() -> None:
    for symbol in SYMBOLS:
        assert len(MEAN_SPREAD_BPS[symbol]) == 24, symbol


def test_every_value_is_a_positive_finite_spread() -> None:
    for symbol in SYMBOLS:
        for hour, value in enumerate(MEAN_SPREAD_BPS[symbol]):
            assert 0.0 < value < 100.0, (symbol, hour, value)


@pytest.mark.parametrize(
    "symbol,hour,expected",
    [
        # Values straight from Measurement 6. If the table is regenerated and these move,
        # the laudo is stale and must be reissued with a new date - not quietly updated.
        ("EURUSD", 13, 0.2548),
        ("EURUSD", 21, 1.2738),
        ("GBPUSD", 9, 0.5557),
        ("GBPUSD", 21, 2.5421),
        ("USDJPY", 21, 2.0751),
    ],
)
def test_values_match_the_laudo(symbol: str, hour: int, expected: float) -> None:
    assert spread_bps(symbol, hour) == pytest.approx(expected)


def test_rollover_is_the_costliest_hour_for_every_pair() -> None:
    """21:00 UTC is the daily rollover, and it is the widest hour in all three pairs."""
    for symbol in SYMBOLS:
        assert costliest_hour(symbol) == 21, symbol


def test_the_flat_default_overstated_cost_on_the_typical_hour() -> None:
    """The measured 13:00 spread is several times smaller than the flat 2.0 bps."""
    for symbol in SYMBOLS:
        assert spread_bps(symbol, 13) < LEGACY_FLAT_SPREAD_BPS / 2.0, symbol


def test_but_the_flat_default_UNDERSTATED_cost_at_rollover() -> None:
    """The finding that inverts the headline: at 21:00 UTC the measured mean exceeds the
    flat default for GBPUSD and USDJPY. A table keyed only on the median would hide this."""
    for symbol in ("GBPUSD", "USDJPY"):
        assert spread_bps(symbol, 21) > LEGACY_FLAT_SPREAD_BPS, symbol


def test_config_for_carries_the_measured_spread_and_keeps_slippage() -> None:
    cfg = config_for("GBPUSD", 21, slippage_bps=1.5)
    assert isinstance(cfg, RealismConfig)
    assert cfg.spread_bps == pytest.approx(2.5421)
    # slippage is not observable in quote data, so it must pass through untouched
    assert cfg.slippage_bps == pytest.approx(1.5)


def test_round_turn_cost_tracks_the_hour() -> None:
    """The whole point of the table: the same pair costs more to round-trip at rollover."""
    quiet = round_turn_cost_bps(config_for("GBPUSD", 9))
    rollover = round_turn_cost_bps(config_for("GBPUSD", 21))
    assert rollover > quiet


def test_unmeasured_symbol_raises_instead_of_falling_back() -> None:
    """A wrong-but-plausible cost is worse than a loud failure."""
    with pytest.raises(KeyError) as excinfo:
        spread_bps("XAUUSD", 12)
    assert "XAUUSD" in str(excinfo.value)


@pytest.mark.parametrize("bad_hour", [-1, 24, 99, 1.5, "13", True, None])
def test_bad_hour_raises(bad_hour: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        spread_bps("EURUSD", bad_hour)  # type: ignore[arg-type]
