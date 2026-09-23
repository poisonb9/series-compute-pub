"""Probability of Backtest Overfitting via CSCV (engine #9 of the offline-evaluation roadmap).

When you tune a strategy by trying N configurations and keeping the one with the best in-sample (IS)
performance, the winner is partly skill and partly luck. PBO (Bailey, Borwein, Lopez de Prado & Zhu,
2017) measures how much of it is luck: the probability that the IS-best configuration performs at or
below the *median* out-of-sample (OOS). A PBO near 0.5 means the selection procedure is no better than
picking at random — the backtest is overfit.

**Combinatorially Symmetric Cross-Validation (CSCV).** Partition the T time-ordered observations into S
(even) groups. For every way to choose S/2 groups as IS, its complement is OOS — the split and its
mirror both appear, which is what makes the estimate *symmetric* and (unlike a single train/test cut)
robust to which half you happened to call "training". For each split:

  1. rank each configuration by its mean IS performance and take the best, ``n*``;
  2. find ``n*``'s relative rank omega in {~0..1} among all configs OOS;
  3. logit lambda = ln(omega / (1 - omega)); ``n*`` is "overfit on this split" when lambda <= 0
     (i.e. the IS winner landed at or below the OOS median).

PBO is the fraction of splits with lambda <= 0. The logit distribution's shape is itself diagnostic:
mass below zero is overfitting.

This engine is built **on top of #6** (`src.purged_cpcv`): the IS/OOS partitions are exactly the CPCV
splits with ``k_test = S/2``, so purging (drop IS bars whose label span overlaps an OOS bar) and embargo
(drop the trailing dependence window) carry straight over. Per Consensus Q8 the honest purge size is the
label span (the holding in bars) and the embargo has no universal formula, so this module also exposes a
**sensitivity sweep** over embargo: if PBO moves materially as embargo grows, the result was
boundary-sensitive, not robust.

Inputs are a plain performance matrix (rows = observations in time order, columns = configurations).
It opens no data, runs no backtest, assigns no dataset role, and authorizes nothing. Running it over the
real quarantined data is Tier-3 evaluation. Built and tested on synthetic matrices (Tier 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from src.purged_cpcv import cpcv_splits


@dataclass(frozen=True)
class PBOResult:
    """Result of a CSCV/PBO run.

    ``pbo`` is the probability of backtest overfitting in [0, 1]. ``logits`` holds one logit per split
    (order matches the CPCV split order); ``is_best_configs`` the IS-winning column per split; and
    ``oos_relative_ranks`` the IS winner's OOS relative rank omega in (0, 1) per split."""

    pbo: float
    n_configs: int
    n_splits: int
    logits: tuple[float, ...]
    is_best_configs: tuple[int, ...]
    oos_relative_ranks: tuple[float, ...]


def _column_means(perf: list[list[float]], rows: list[int]) -> list[float]:
    """Mean performance of every configuration over the given observation rows."""
    n_cols = len(perf[0])
    totals = [0.0] * n_cols
    for r in rows:
        row = perf[r]
        for c in range(n_cols):
            totals[c] += row[c]
    count = len(rows)
    return [t / count for t in totals]


def _relative_rank(values: list[float], target: int) -> float:
    """Midrank-based relative rank of ``values[target]`` among ``values``, mapped to (0, 1).

    Uses the mean rank for ties so the result is deterministic and never hits exactly 0 or 1 (which
    would make the logit infinite). omega <= 0.5 means the target sits at or below the median."""
    x = values[target]
    strictly_less = sum(1 for i, v in enumerate(values) if v < x)
    equal_others = sum(1 for i, v in enumerate(values) if v == x and i != target)
    midrank = strictly_less + equal_others / 2.0 + 0.5  # in (0, n)
    return midrank / len(values)


def probability_of_backtest_overfitting(
    perf: list[list[float]], n_groups: int, horizon: int = 0, embargo: int = 0,
) -> PBOResult:
    """CSCV probability of backtest overfitting over a ``T x N`` performance matrix.

    perf: rows are observations in time order, columns are configurations (performance per observation,
    e.g. per-bar return or per-period Sharpe contribution). n_groups (S): even number of CSCV groups;
    IS/OOS each get S/2. horizon/embargo: purge + embargo passed to the #6 CPCV splitter (Q8: purge =
    label span in bars). The number of splits is comb(S, S/2)."""
    if not perf or not perf[0]:
        raise ValueError("perf must be a non-empty T x N matrix")
    n_cols = len(perf[0])
    if any(len(row) != n_cols for row in perf):
        raise ValueError("perf must be rectangular")
    if n_cols < 2:
        raise ValueError("need at least 2 configurations to rank")
    if n_groups % 2 != 0:
        raise ValueError("n_groups must be even so IS and OOS are equal halves")

    # IS = train side (purged), OOS = test side of the #6 combinatorial splits with k_test = S/2.
    splits = cpcv_splits(n=len(perf), n_groups=n_groups, k_test=n_groups // 2,
                         horizon=horizon, embargo=embargo)

    logits: list[float] = []
    is_best: list[int] = []
    oos_ranks: list[float] = []
    overfit = 0
    for s in splits:
        if not s.train or not s.test:
            raise ValueError(
                "a split has an empty IS or OOS set after purge/embargo; "
                "reduce horizon/embargo or n_groups"
            )
        is_means = _column_means(perf, s.train)
        oos_means = _column_means(perf, s.test)
        n_star = max(range(n_cols), key=lambda c: is_means[c])  # first max on ties (deterministic)
        omega = _relative_rank(oos_means, n_star)
        lam = log(omega / (1.0 - omega))
        logits.append(lam)
        is_best.append(n_star)
        oos_ranks.append(omega)
        if lam <= 0.0:  # IS winner at or below the OOS median
            overfit += 1

    return PBOResult(
        pbo=overfit / len(splits),
        n_configs=n_cols,
        n_splits=len(splits),
        logits=tuple(logits),
        is_best_configs=tuple(is_best),
        oos_relative_ranks=tuple(oos_ranks),
    )


@dataclass(frozen=True)
class PBOSensitivityPoint:
    """One row of the embargo sensitivity sweep: the embargo used and the PBO it produced."""

    embargo: int
    pbo: float


def pbo_embargo_sweep(
    perf: list[list[float]], n_groups: int, embargos: list[int], horizon: int = 0,
) -> list[PBOSensitivityPoint]:
    """Q8 boundary-robustness sweep: recompute PBO across a small grid of embargo values.

    If PBO moves materially as embargo grows, the result was boundary-sensitive and not robust. The
    caller supplies the grid (e.g. ``[e, 2*e, 3*e]``); embargos must be non-negative."""
    if not embargos:
        raise ValueError("provide at least one embargo value")
    if any(e < 0 for e in embargos):
        raise ValueError("embargo values must be >= 0")
    return [
        PBOSensitivityPoint(
            embargo=e,
            pbo=probability_of_backtest_overfitting(
                perf, n_groups=n_groups, horizon=horizon, embargo=e
            ).pbo,
        )
        for e in embargos
    ]
