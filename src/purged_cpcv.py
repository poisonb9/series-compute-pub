"""Purged + embargoed Combinatorial Cross-Validation (engine #6 of the offline-evaluation roadmap).

Plain walk-forward tests one train/test cut. CPCV (Lopez de Prado) tests every C(N, k) way to pick k of
N time-ordered groups as the test set, giving many backtest paths instead of one — far harder for a
strategy to fit by luck. But splitting a TIME series naively leaks: a training observation whose outcome
horizon overlaps a test observation shares information with it. Two defences, both applied here:

- **Purging** — drop any training bar whose label interval [j, j+horizon] overlaps a test bar's interval
  [t, t+horizon]; equivalently, drop training bars within ``horizon`` of any test bar.
- **Embargo** — additionally drop training bars in the ``embargo`` window immediately AFTER each test
  bar, killing the serial-correlation leak that purging alone leaves at the test set's trailing edge.

This module produces the index splits only — deterministic, stdlib-only, look-ahead-safe. It opens no
data, runs no backtest, assigns no role, and authorizes nothing. Running these splits over the real
quarantined data is Tier-3 evaluation. Built and tested on plain index ranges (Tier 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb


@dataclass(frozen=True)
class CPCVSplit:
    """One combinatorial split: ``test`` are the held-out bar indices, ``train`` the surviving
    training indices after purge + embargo. ``test_groups`` records which group ids formed the test."""

    train: list[int]
    test: list[int]
    test_groups: tuple[int, ...]


def _contiguous_groups(n: int, n_groups: int) -> list[list[int]]:
    """Partition range(n) into ``n_groups`` contiguous, near-equal, time-ordered blocks."""
    base, extra = divmod(n, n_groups)
    groups: list[list[int]] = []
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < extra else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def cpcv_splits(
    n: int, n_groups: int, k_test: int, horizon: int = 0, embargo: int = 0,
) -> list[CPCVSplit]:
    """All C(n_groups, k_test) purged+embargoed splits over ``n`` time-ordered observations.

    horizon: label length in bars (purge radius). embargo: extra bars dropped after each test bar.
    The number of splits is exactly comb(n_groups, k_test)."""
    if n < n_groups:
        raise ValueError("n must be >= n_groups")
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if not 1 <= k_test < n_groups:
        raise ValueError("k_test must satisfy 1 <= k_test < n_groups")
    if horizon < 0 or embargo < 0:
        raise ValueError("horizon and embargo must be >= 0")

    groups = _contiguous_groups(n, n_groups)
    splits: list[CPCVSplit] = []
    for combo in combinations(range(n_groups), k_test):
        test_idx = sorted(i for g in combo for i in groups[g])
        forbidden: set[int] = set()
        for t in test_idx:
            forbidden.update(range(t - horizon, t + horizon + 1))  # purge radius
            forbidden.update(range(t + 1, t + embargo + 1))        # embargo (trailing)
        test_set = set(test_idx)
        train = [
            i
            for g in range(n_groups) if g not in combo
            for i in groups[g]
            if i not in forbidden and i not in test_set
        ]
        splits.append(CPCVSplit(train=train, test=test_idx, test_groups=combo))
    return splits


def n_paths(n_groups: int, k_test: int) -> int:
    """Number of CPCV backtest paths = comb(n_groups, k_test). Convenience for reporting."""
    return comb(n_groups, k_test)
