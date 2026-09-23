"""Execution-realism / latency layer for backtests (research/offline).

Engine #4 of the offline-evaluation roadmap: the safeguards that stop a backtest from lying about
fills — the difference between a scalp that "works" on paper and one that survives real execution.

It models, deterministically:
- **spread + slippage:** a buy fills a little higher, a sell a little lower (never in your favour);
- **price-touch != fill:** a stop/limit only fills if price actually traded through the level;
- **intrabar stop/target ambiguity:** when both the stop and the target lie inside one bar's range,
  you cannot know which hit first, so — conservatively — assume the ADVERSE one (the stop) filled.

Pure functions, deterministic, stdlib only. Research/offline: it adjusts hypothetical fills; it does
not open data, run a backtest, place an order, assign a role, or authorize anything. Built and tested
on constructed bars (Tier 1); applying it to real strategy runs is part of Tier-3 evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data_loader import OHLCVBar

LONG = "long"
SHORT = "short"


@dataclass(frozen=True)
class RealismConfig:
    """spread_bps: full bid/ask spread in basis points (half is applied per side).
    slippage_bps: extra adverse move per fill. assume_adverse_first: on an ambiguous
    intrabar bar, resolve to the stop (the pessimistic, honest choice).

    NOTE on the 2.0 default (decision 29): it was never measured, and Measurement 6
    (2026-09-23) showed it is 3.3x to 7.2x too HIGH on a typical hour and too LOW at the
    21:00 UTC rollover for GBPUSD and USDJPY. No single scalar is right, because the real
    spread varies by symbol and by hour. The default is kept only so existing callers keep
    their behaviour; for anything measured, build the config with
    ``src.measured_spread.config_for(symbol, hour_utc)`` instead."""

    spread_bps: float = 2.0
    slippage_bps: float = 1.0
    assume_adverse_first: bool = True

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError("spread_bps must be >= 0")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")


def adverse_fill(reference_price: float, is_buy: bool, cfg: RealismConfig) -> float:
    """Fill price after half-spread + slippage, always against the trader.

    A buy (long entry / short cover) fills higher; a sell (long exit / short entry) fills lower."""
    if reference_price <= 0:
        raise ValueError("reference_price must be > 0")
    cost = (cfg.spread_bps / 2.0 + cfg.slippage_bps) / 10000.0
    return reference_price * (1.0 + cost) if is_buy else reference_price * (1.0 - cost)


def level_is_touched(bar: OHLCVBar, level: float) -> bool:
    """price-touch: the level was reached only if it lies within the bar's traded range."""
    return bar.low <= level <= bar.high


def resolve_long_exit(
    bar: OHLCVBar, stop: float, target: float | None, cfg: RealismConfig
) -> tuple[str, float | None]:
    """Resolve a long position's exit within one bar, honestly.

    Returns (outcome, fill_price): outcome in {"stop", "target", "none"}. When both the stop and the
    target are touched in the same bar the true order is unknown, so with ``assume_adverse_first`` the
    stop wins. Fills include spread+slippage (a stop/target exit is a SELL -> fills lower)."""
    hit_stop = level_is_touched(bar, stop)
    hit_target = target is not None and level_is_touched(bar, target)
    if hit_stop and hit_target:
        if cfg.assume_adverse_first:
            return "stop", adverse_fill(stop, is_buy=False, cfg=cfg)
        return "target", adverse_fill(target, is_buy=False, cfg=cfg)
    if hit_stop:
        return "stop", adverse_fill(stop, is_buy=False, cfg=cfg)
    if hit_target:
        return "target", adverse_fill(target, is_buy=False, cfg=cfg)
    return "none", None


def realistic_entry_fill(reference_price: float, side: str, cfg: RealismConfig) -> float:
    """Entry fill for a new position: a long entry is a BUY (fills higher), a short entry a SELL."""
    if side not in (LONG, SHORT):
        raise ValueError("side must be 'long' or 'short'")
    return adverse_fill(reference_price, is_buy=(side == LONG), cfg=cfg)


def round_turn_cost_bps(cfg: RealismConfig) -> float:
    """Total round-trip execution cost in bps (entry + exit): spread once + slippage twice.
    Useful as a floor a scalp's per-trade edge must clear to be plausible."""
    return cfg.spread_bps + 2.0 * cfg.slippage_bps
