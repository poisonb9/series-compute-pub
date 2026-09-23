"""Tests for the execution-realism / latency layer.

Constructed bars only; no data opened, no order placed. Pins that fills are always adverse, that a
level only fills when touched, and that intrabar stop/target ambiguity resolves to the stop.
"""

from datetime import datetime

import pytest

from src.data_loader import OHLCVBar
from src.execution_realism import (
    LONG,
    SHORT,
    RealismConfig,
    adverse_fill,
    level_is_touched,
    realistic_entry_fill,
    resolve_long_exit,
    round_turn_cost_bps,
)


CFG = RealismConfig(spread_bps=2.0, slippage_bps=1.0)


def _bar(high, low, close=None, open_=None) -> OHLCVBar:
    return OHLCVBar(
        timestamp=datetime(2026, 1, 1), open=open_ if open_ is not None else close or high,
        high=high, low=low, close=close if close is not None else (high + low) / 2, volume=100.0,
    )


# --- adverse fills ------------------------------------------------------------


def test_buy_fills_higher_sell_fills_lower():
    assert adverse_fill(100.0, is_buy=True, cfg=CFG) > 100.0
    assert adverse_fill(100.0, is_buy=False, cfg=CFG) < 100.0


def test_fill_cost_is_half_spread_plus_slippage():
    # (2/2 + 1) bps = 2 bps = 0.0002
    assert adverse_fill(100.0, is_buy=True, cfg=CFG) == pytest.approx(100.0 * 1.0002)
    assert adverse_fill(100.0, is_buy=False, cfg=CFG) == pytest.approx(100.0 * 0.9998)


def test_zero_cost_config_is_lossless():
    z = RealismConfig(spread_bps=0.0, slippage_bps=0.0)
    assert adverse_fill(100.0, is_buy=True, cfg=z) == pytest.approx(100.0)


def test_reference_price_must_be_positive():
    with pytest.raises(ValueError):
        adverse_fill(0.0, is_buy=True, cfg=CFG)


def test_config_rejects_negative():
    with pytest.raises(ValueError):
        RealismConfig(spread_bps=-1.0)
    with pytest.raises(ValueError):
        RealismConfig(slippage_bps=-1.0)


# --- price touch --------------------------------------------------------------


def test_level_touched_only_within_range():
    b = _bar(high=101.0, low=99.0)
    assert level_is_touched(b, 100.0)
    assert level_is_touched(b, 99.0)
    assert level_is_touched(b, 101.0)
    assert not level_is_touched(b, 98.9)
    assert not level_is_touched(b, 101.1)


# --- intrabar exit resolution -------------------------------------------------


def test_only_stop_touched():
    b = _bar(high=100.5, low=98.0)
    outcome, fill = resolve_long_exit(b, stop=99.0, target=102.0, cfg=CFG)
    assert outcome == "stop"
    assert fill < 99.0  # sell fills lower


def test_only_target_touched():
    b = _bar(high=103.0, low=100.5)
    outcome, fill = resolve_long_exit(b, stop=99.0, target=102.0, cfg=CFG)
    assert outcome == "target"
    assert fill < 102.0


def test_neither_touched():
    b = _bar(high=101.0, low=100.0)
    outcome, fill = resolve_long_exit(b, stop=99.0, target=102.0, cfg=CFG)
    assert outcome == "none"
    assert fill is None


def test_ambiguous_bar_resolves_to_stop_by_default():
    # Both stop (99) and target (102) inside [98, 103] -> adverse-first -> stop.
    b = _bar(high=103.0, low=98.0)
    outcome, _ = resolve_long_exit(b, stop=99.0, target=102.0, cfg=CFG)
    assert outcome == "stop"


def test_ambiguous_bar_optimistic_config_resolves_to_target():
    b = _bar(high=103.0, low=98.0)
    cfg = RealismConfig(spread_bps=2.0, slippage_bps=1.0, assume_adverse_first=False)
    outcome, _ = resolve_long_exit(b, stop=99.0, target=102.0, cfg=cfg)
    assert outcome == "target"


def test_no_target_only_stop_considered():
    b = _bar(high=105.0, low=98.0)
    outcome, _ = resolve_long_exit(b, stop=99.0, target=None, cfg=CFG)
    assert outcome == "stop"


# --- entry fill & round-turn cost ---------------------------------------------


def test_entry_fill_side_direction():
    assert realistic_entry_fill(100.0, LONG, CFG) > 100.0
    assert realistic_entry_fill(100.0, SHORT, CFG) < 100.0


def test_entry_rejects_bad_side():
    with pytest.raises(ValueError):
        realistic_entry_fill(100.0, "sideways", CFG)


def test_round_turn_cost():
    assert round_turn_cost_bps(CFG) == pytest.approx(2.0 + 2 * 1.0)
