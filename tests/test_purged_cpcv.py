"""Tests for purged + embargoed CPCV (engine #6 of the offline-evaluation roadmap).

Plain index ranges only. Pins the invariants that make the splits trustworthy: the right number of
paths, train/test disjointness, that purging removes train bars within the horizon of any test bar, and
that embargo removes train bars in the trailing window after a test bar.
"""

from math import comb

import pytest

from src.purged_cpcv import CPCVSplit, cpcv_splits, n_paths


# --- shape --------------------------------------------------------------------


def test_number_of_paths_is_n_choose_k():
    splits = cpcv_splits(n=100, n_groups=6, k_test=2)
    assert len(splits) == comb(6, 2) == 15
    assert n_paths(6, 2) == 15


def test_every_index_in_test_covered_across_all_splits():
    # Union of all test sets should be every observation (each group appears in some combo).
    splits = cpcv_splits(n=60, n_groups=6, k_test=2)
    covered: set[int] = set()
    for s in splits:
        covered.update(s.test)
    assert covered == set(range(60))


def test_test_is_k_groups_worth_of_indices():
    splits = cpcv_splits(n=60, n_groups=6, k_test=2)  # 10 per group, evenly divisible
    for s in splits:
        assert len(s.test) == 20


# --- disjointness & no leakage ------------------------------------------------


def test_train_and_test_are_disjoint():
    for s in cpcv_splits(n=100, n_groups=5, k_test=2, horizon=3, embargo=2):
        assert set(s.train).isdisjoint(s.test)


def test_no_purge_no_embargo_train_is_complement():
    splits = cpcv_splits(n=60, n_groups=6, k_test=1, horizon=0, embargo=0)
    for s in splits:
        assert sorted(s.train + s.test) == list(range(60))


def test_purge_removes_train_within_horizon_of_test():
    horizon = 4
    for s in cpcv_splits(n=120, n_groups=6, k_test=2, horizon=horizon, embargo=0):
        test_set = set(s.test)
        for j in s.train:
            assert all(abs(j - t) > horizon for t in test_set)


def test_embargo_removes_trailing_window_after_test():
    embargo = 5
    for s in cpcv_splits(n=120, n_groups=6, k_test=2, horizon=0, embargo=embargo):
        test_set = set(s.test)
        for j in s.train:
            # no train bar may sit in (t, t+embargo] for any test bar t
            assert all(not (t < j <= t + embargo) for t in test_set)


def test_larger_horizon_never_grows_the_training_set():
    def train_size(h: int) -> int:
        return sum(len(s.train) for s in cpcv_splits(n=120, n_groups=6, k_test=2, horizon=h))

    sizes = [train_size(h) for h in (0, 2, 5, 10)]
    assert sizes == sorted(sizes, reverse=True)  # monotonically non-increasing


# --- validation ---------------------------------------------------------------


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        cpcv_splits(n=3, n_groups=5, k_test=2)      # n < n_groups
    with pytest.raises(ValueError):
        cpcv_splits(n=100, n_groups=1, k_test=1)    # n_groups < 2
    with pytest.raises(ValueError):
        cpcv_splits(n=100, n_groups=5, k_test=5)    # k_test not < n_groups
    with pytest.raises(ValueError):
        cpcv_splits(n=100, n_groups=5, k_test=0)    # k_test < 1
    with pytest.raises(ValueError):
        cpcv_splits(n=100, n_groups=5, k_test=2, horizon=-1)
    with pytest.raises(ValueError):
        cpcv_splits(n=100, n_groups=5, k_test=2, embargo=-1)


def test_split_is_frozen_dataclass():
    s = cpcv_splits(n=20, n_groups=4, k_test=1)[0]
    assert isinstance(s, CPCVSplit)
    with pytest.raises(Exception):
        s.train = []  # type: ignore[misc]
