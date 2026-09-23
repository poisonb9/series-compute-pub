"""Execution-model sensitivity: same-bar-close vs next-bar-open fills (research/offline).

The tested `Backtester` fills entries and exits at the close of the *same* bar the signal was formed on
(`src/lookahead_audit.py::_run_same_bar_close_execution_semantics` records this as a medium-severity
methodology warning, not future leakage). Consensus Q14/Q15 confirm that next-bar execution is the honest
anti-look-ahead convention for a real test, and that the size of the difference must be measured, not
assumed. This engine measures exactly that difference **without modifying the protected backtester**: it
replays a detector's BUY/EXIT signals under two fill conventions and reports how much the same-bar-close
assumption flatters the per-trade return.

- **same_bar_close**: a BUY signal at bar i fills at ``close[i]``; an EXIT at bar j fills at ``close[j]``.
  This mirrors the current backtester.
- **next_bar_open**: the same signals fill at ``open[i+1]`` and ``open[j+1]`` — the honest fill, since the
  close that formed the signal is only known once the bar has closed.

The gap between the two (``optimism = same_bar_mean - next_bar_mean`` per-trade net return) is the
"execution optimism" of the current engine for a given detector. Deterministic, stdlib only, long-only,
same cost charged to both sides so the comparison is clean. It opens no data, runs no backtest engine,
assigns no dataset role, and authorizes nothing. Built and tested on constructed bars (Tier 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.backtester import Signal, SignalType
from src.data_loader import OHLCVBar


@dataclass(frozen=True)
class ExecutionSensitivityResult:
    """Per-trade net returns of one detector under both fill conventions and their gap.

    ``optimism`` is ``same_bar_close_mean - next_bar_open_mean``; a positive value means the current
    same-bar-close engine reports a higher per-trade return than an honest next-bar fill would."""

    n_trades: int
    same_bar_close_returns: tuple[float, ...]
    next_bar_open_returns: tuple[float, ...]
    same_bar_close_mean: float
    next_bar_open_mean: float
    optimism: float


def _signal_bars(bars: list[OHLCVBar], signal_fn) -> tuple[list[int], list[int]]:
    """Replay the long-only state machine and return (entry_bar_indices, exit_bar_indices).

    A BUY while flat opens a position at bar i; an EXIT while in a position closes it at bar j. Signals
    are read exactly as the detector emits them, so both fill conventions see the same decision bars."""
    entries: list[int] = []
    exits: list[int] = []
    in_pos = False
    for i in range(len(bars)):
        sig = signal_fn(bars, i)
        stype = sig.signal_type if isinstance(sig, Signal) else SignalType.HOLD
        if not in_pos and stype is SignalType.BUY:
            entries.append(i)
            in_pos = True
        elif in_pos and stype is SignalType.EXIT:
            exits.append(i)
            in_pos = False
    # drop a dangling open position (no matched exit) so both conventions compare the same closed trades
    entries = entries[: len(exits)]
    return entries, exits


def _trade_returns(bars, entries, exits, price, cost) -> list[float]:
    """Net per-trade return under a price accessor ``price(bar_index) -> float | None``.

    A trade is skipped (never fabricated) if either leg's fill price is unavailable (e.g. next-bar open
    past the end of the series) — the caller matches trade counts by construction."""
    out: list[float] = []
    for e, x in zip(entries, exits):
        pe, px = price(e), price(x)
        if pe is None or px is None or pe <= 0:
            continue
        out.append((px - pe) / pe - cost)
    return out


def execution_sensitivity(
    bars: list[OHLCVBar], signal_fn, cost_per_trade: float = 0.0
) -> ExecutionSensitivityResult:
    """Measure a detector's per-trade returns under same-bar-close vs next-bar-open fills.

    ``signal_fn(bars, i) -> Signal`` is the detector (long-only BUY/EXIT/HOLD). Both conventions replay the
    identical signal decisions and are charged ``cost_per_trade`` once per trade, so the only difference is
    the fill price. Deterministic. Raises ValueError on empty input."""
    if not bars:
        raise ValueError("bars must be non-empty")
    entries, exits = _signal_bars(bars, signal_fn)

    def close_price(i: int):
        return bars[i].close

    def next_open_price(i: int):
        return bars[i + 1].open if i + 1 < len(bars) else None

    # next-bar fill can drop a trade whose exit is the last bar; keep the two conventions on the same
    # closed trades by intersecting on feasibility.
    feasible = [
        k for k, (e, x) in enumerate(zip(entries, exits))
        if e + 1 < len(bars) and x + 1 < len(bars)
    ]
    entries = [entries[k] for k in feasible]
    exits = [exits[k] for k in feasible]

    same = _trade_returns(bars, entries, exits, close_price, cost_per_trade)
    nxt = _trade_returns(bars, entries, exits, next_open_price, cost_per_trade)
    n = min(len(same), len(nxt))
    same, nxt = same[:n], nxt[:n]
    same_mean = sum(same) / n if n else 0.0
    next_mean = sum(nxt) / n if n else 0.0
    return ExecutionSensitivityResult(
        n_trades=n,
        same_bar_close_returns=tuple(same),
        next_bar_open_returns=tuple(nxt),
        same_bar_close_mean=same_mean,
        next_bar_open_mean=next_mean,
        optimism=same_mean - next_mean,
    )
