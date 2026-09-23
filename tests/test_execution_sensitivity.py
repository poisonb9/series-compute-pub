"""Tests for execution-model sensitivity (same-bar-close vs next-bar-open fills).

Constructed bars only. Pins: the two conventions replay the same trades, a market that keeps rising after
each signal makes same-bar-close optimistic (positive optimism), a flat market makes them equal, costs hit
both sides equally, and the run is deterministic. No real data, no backtester run, no Tier flag.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.backtester import Signal, SignalType
from src.data_loader import OHLCVBar
from src.execution_sensitivity import ExecutionSensitivityResult, execution_sensitivity


def _bar(i, o, h, l, c):
    return OHLCVBar(timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i),
                    open=o, high=h, low=l, close=c, volume=100.0)


def _flat_signal(entry_at, exit_at):
    def fn(bars, i):
        if i == entry_at:
            return Signal(signal_type=SignalType.BUY)
        if i == exit_at:
            return Signal(signal_type=SignalType.EXIT)
        return Signal(signal_type=SignalType.HOLD)
    return fn


def test_result_shape_and_frozen():
    bars = [_bar(i, 100, 101, 99, 100) for i in range(10)]
    res = execution_sensitivity(bars, _flat_signal(2, 5))
    assert isinstance(res, ExecutionSensitivityResult)
    assert res.n_trades == 1
    assert len(res.same_bar_close_returns) == len(res.next_bar_open_returns) == 1
    with pytest.raises(Exception):
        res.optimism = 0.0  # type: ignore[misc]


def test_flat_market_zero_optimism():
    # every open and close is 100 -> same-bar-close and next-bar-open fills are identical -> optimism 0.
    bars = [_bar(i, 100, 100, 100, 100) for i in range(12)]
    res = execution_sensitivity(bars, _flat_signal(3, 8))
    assert res.optimism == pytest.approx(0.0)
    assert res.same_bar_close_mean == pytest.approx(res.next_bar_open_mean)


def test_rising_market_same_bar_close_is_optimistic():
    # price rises every bar: close[i] < open[i+1]. Entering at close[i] (cheaper) and exiting at close[j]
    # (before the next up-gap) beats entering at the higher next open -> same-bar-close looks better.
    bars = []
    price = 100.0
    for i in range(15):
        o = price
        c = price + 1.0          # close above open
        bars.append(_bar(i, o, c + 0.5, o - 0.5, c))
        price = c + 1.0          # gap up to next open
    res = execution_sensitivity(bars, _flat_signal(2, 9))
    assert res.n_trades == 1
    assert res.optimism > 0.0    # the current engine flatters the return
    assert res.same_bar_close_mean > res.next_bar_open_mean


def test_cost_charged_to_both_sides_equally():
    bars = [_bar(i, 100, 102, 98, 101) for i in range(10)]
    free = execution_sensitivity(bars, _flat_signal(2, 6), cost_per_trade=0.0)
    costed = execution_sensitivity(bars, _flat_signal(2, 6), cost_per_trade=0.01)
    # a per-trade cost lowers both conventions by exactly the same amount -> optimism (the gap) is unchanged
    assert costed.optimism == pytest.approx(free.optimism)
    assert costed.same_bar_close_mean == pytest.approx(free.same_bar_close_mean - 0.01)
    assert costed.next_bar_open_mean == pytest.approx(free.next_bar_open_mean - 0.01)


def test_multiple_trades_counted():
    bars = [_bar(i, 100, 105, 95, 100 + (i % 3)) for i in range(30)]

    def fn(bars, i):
        if i in (2, 12, 22):
            return Signal(signal_type=SignalType.BUY)
        if i in (6, 16, 26):
            return Signal(signal_type=SignalType.EXIT)
        return Signal(signal_type=SignalType.HOLD)

    res = execution_sensitivity(bars, fn)
    assert res.n_trades == 3


def test_dangling_open_position_is_dropped():
    # BUY with no matching EXIT -> no closed trade, both conventions agree on zero trades.
    bars = [_bar(i, 100, 101, 99, 100) for i in range(8)]

    def fn(bars, i):
        return Signal(signal_type=SignalType.BUY) if i == 2 else Signal(signal_type=SignalType.HOLD)

    res = execution_sensitivity(bars, fn)
    assert res.n_trades == 0


def test_exit_on_last_bar_has_no_next_open_and_is_excluded():
    n = 8
    bars = [_bar(i, 100, 101, 99, 100) for i in range(n)]
    # exit on the final bar: next-bar open does not exist -> trade excluded from BOTH conventions.
    res = execution_sensitivity(bars, _flat_signal(2, n - 1))
    assert res.n_trades == 0


def test_deterministic():
    bars = [_bar(i, 100 + i, 106 + i, 94 + i, 101 + i) for i in range(20)]
    a = execution_sensitivity(bars, _flat_signal(3, 11), cost_per_trade=0.001)
    b = execution_sensitivity(bars, _flat_signal(3, 11), cost_per_trade=0.001)
    assert a == b


def test_empty_bars_rejected():
    with pytest.raises(ValueError):
        execution_sensitivity([], _flat_signal(0, 1))
