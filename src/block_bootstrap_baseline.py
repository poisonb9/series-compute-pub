"""Block-bootstrap general null (engine #3b of the offline-evaluation roadmap).

Engine #3 (`src.matched_baseline`) answers a **narrow** question — "holding exit, holding period and
costs fixed, does this exact entry *timing* beat random timestamps?" — by laying random entries on the
*real* series through the tested backtester. That is a valid null, but a narrow one: it keeps the
real path and only scrambles when you enter.

Consensus Q9 (`docs/CONSENSUS_Q9_..._ANSWER_20260722.md`) found the best *general* null is a
**dependence-preserving block bootstrap** with matched exposure and costs, not naive random entries:
resample contiguous blocks of the return series so autocorrelation and volatility clustering survive,
then lay down the *same* number of trades for the *same* holding (time-in-market matched) and charge
the *same* per-trade cost. The question it answers is the broad one — "does the strategy's realised
per-trade edge beat what this dependent market throws up by chance?" — and the two nulls are
complementary, not substitutes: keep the random-entry null as the narrow entry-timing anchor and add
this as the general one.

Two resamplers are provided:

- **stationary bootstrap** (Politis & Romano, 1994): geometric block lengths (mean = the expected
  block), circular — adapts to varying dependence lengths and is the safer default;
- **moving-block bootstrap**: fixed-length overlapping blocks, circular.

The block length is not arbitrary: `estimate_block_length` derives it from the series' own
autocorrelation (the first lag whose sample ACF falls inside the white-noise band), per Q9's "choose
the block from the covariance structure". Everything is deterministic given a seed, stdlib-only, and
fail-closed. It opens no data, runs no backtest, assigns no dataset role, and authorizes nothing.
Running it over the real quarantined data is Tier-3 evaluation. Built and tested on synthetic return
series (Tier 1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt


# --- dependence-aware block length -------------------------------------------


def _autocorrelation(values: list[float], lag: int) -> float:
    """Sample autocorrelation of ``values`` at ``lag`` (0.0 for a constant series)."""
    n = len(values)
    mean = sum(values) / n
    denom = sum((v - mean) ** 2 for v in values)
    if denom == 0.0:
        return 0.0
    num = sum((values[t] - mean) * (values[t + lag] - mean) for t in range(n - lag))
    return num / denom


def estimate_block_length(values: list[float], max_lag: int | None = None) -> int:
    """Dependence-aware expected block length from the series' autocorrelation.

    Finds the first lag whose sample ACF falls inside the +-1.96/sqrt(n) white-noise band (i.e. the
    dependence has decayed to noise) and returns twice that lag, floored at 1 and capped at n // 2. An
    IID series returns a small block; a strongly autocorrelated one returns a larger block. Deterministic."""
    n = len(values)
    if n < 4:
        return 1
    if max_lag is None:
        max_lag = max(1, min(n // 2, int(10 * sqrt(n) ** 0.5) + n // 20))
    max_lag = min(max_lag, n - 1)
    band = 1.96 / sqrt(n)
    first_noise_lag = max_lag
    for k in range(1, max_lag + 1):
        if abs(_autocorrelation(values, k)) < band:
            first_noise_lag = k
            break
    return max(1, min(n // 2, 2 * first_noise_lag))


# --- resamplers (circular, deterministic given rng) --------------------------


def stationary_bootstrap_series(
    values: list[float], expected_block: int, rng: random.Random
) -> list[float]:
    """Politis-Romano stationary bootstrap: circular blocks of geometric length (mean =
    ``expected_block``). Returns a resample the same length as ``values``."""
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    if expected_block < 1:
        raise ValueError("expected_block must be >= 1")
    p = 1.0 / expected_block
    out: list[float] = []
    idx = rng.randrange(n)
    while len(out) < n:
        out.append(values[idx])
        if rng.random() < p:
            idx = rng.randrange(n)  # start a new block
        else:
            idx = (idx + 1) % n     # continue the current block (wrap around)
    return out


def moving_block_bootstrap_series(
    values: list[float], block_len: int, rng: random.Random
) -> list[float]:
    """Moving-block bootstrap: concatenate circular overlapping blocks of fixed length ``block_len``,
    truncated to len(values)."""
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    if not 1 <= block_len <= n:
        raise ValueError("block_len must be in [1, len(values)]")
    out: list[float] = []
    while len(out) < n:
        start = rng.randrange(n)
        for j in range(block_len):
            out.append(values[(start + j) % n])
    return out[:n]


# --- matched null over the resampled series ----------------------------------


@dataclass(frozen=True)
class BootstrapNullResult:
    """One matched block-bootstrap null run.

    ``p_value`` is the one-sided upper p-value P(null_stat >= observed) with the standard +1 smoothing;
    ``null_stats`` holds the per-replicate statistics (length = ``n_reps``)."""

    p_value: float
    observed: float
    null_mean: float
    null_std: float
    block_len: int
    n_reps: int
    method: str
    statistic: str
    null_stats: tuple[float, ...]


def _trade_windows(n_bars: int, n_trades: int, holding: int) -> list[int]:
    """Contiguous, non-overlapping trade start indices matching exposure exactly. Fail-closed: raises
    if the series cannot hold ``n_trades`` non-overlapping windows of ``holding`` bars."""
    if n_trades * holding > n_bars:
        raise ValueError(
            f"series too short: {n_trades} trades x {holding} bars > {n_bars} bars available"
        )
    return [i * holding for i in range(n_trades)]


def _trade_returns(series: list[float], starts: list[int], holding: int, cost: float) -> list[float]:
    """Net per-trade return: sum of bar returns over the holding window minus the per-trade cost."""
    return [sum(series[s : s + holding]) - cost for s in starts]


def _statistic(trade_returns: list[float], kind: str) -> float:
    """Per-replicate summary statistic over the matched trades."""
    n = len(trade_returns)
    mean = sum(trade_returns) / n
    if kind == "mean":
        return mean
    if kind == "sharpe":
        var = sum((r - mean) ** 2 for r in trade_returns) / n  # population variance
        if var == 0.0:
            return 0.0
        return mean / sqrt(var)
    raise ValueError("statistic must be 'mean' or 'sharpe'")


def matched_block_bootstrap_null(
    bar_returns: list[float],
    n_trades: int,
    holding_bars: int,
    observed_stat: float,
    statistic: str = "mean",
    cost_per_trade: float = 0.0,
    n_reps: int = 1000,
    block_len: int | None = None,
    method: str = "stationary",
    seed: int = 0,
) -> BootstrapNullResult:
    """General dependence-preserving null for a strategy's per-trade edge.

    Resamples ``bar_returns`` in contiguous blocks (preserving autocorrelation and volatility
    clustering), lays ``n_trades`` non-overlapping trades of ``holding_bars`` bars each (time-in-market
    matched), charges ``cost_per_trade`` per trade, and computes ``statistic`` (``'mean'`` net per-trade
    return or ``'sharpe'``) on each of ``n_reps`` replicates. ``block_len`` defaults to the
    dependence-aware estimate. Returns the null distribution and the one-sided upper p-value of
    ``observed_stat`` against it. Deterministic given ``seed``."""
    n = len(bar_returns)
    if n == 0:
        raise ValueError("bar_returns must be non-empty")
    if n_trades < 1 or holding_bars < 1:
        raise ValueError("n_trades and holding_bars must be >= 1")
    if n_reps < 1:
        raise ValueError("n_reps must be >= 1")
    if method not in ("stationary", "moving"):
        raise ValueError("method must be 'stationary' or 'moving'")
    if statistic not in ("mean", "sharpe"):
        raise ValueError("statistic must be 'mean' or 'sharpe'")

    starts = _trade_windows(n, n_trades, holding_bars)  # fail-closed on too-short series
    if block_len is None:
        block_len = estimate_block_length(bar_returns)
    if not 1 <= block_len <= n:
        raise ValueError("block_len must be in [1, len(bar_returns)]")

    rng = random.Random(seed)
    null_stats: list[float] = []
    for _ in range(n_reps):
        if method == "stationary":
            series = stationary_bootstrap_series(bar_returns, block_len, rng)
        else:
            series = moving_block_bootstrap_series(bar_returns, block_len, rng)
        null_stats.append(_statistic(_trade_returns(series, starts, holding_bars, cost_per_trade), statistic))

    count_ge = sum(1 for s in null_stats if s >= observed_stat)
    p_value = (1 + count_ge) / (n_reps + 1)  # +1 smoothing: never exactly 0
    null_mean = sum(null_stats) / len(null_stats)
    null_var = sum((s - null_mean) ** 2 for s in null_stats) / len(null_stats)

    return BootstrapNullResult(
        p_value=p_value,
        observed=observed_stat,
        null_mean=null_mean,
        null_std=sqrt(null_var),
        block_len=block_len,
        n_reps=n_reps,
        method=method,
        statistic=statistic,
        null_stats=tuple(null_stats),
    )
