"""Tests for CSCV probability of backtest overfitting (engine #9 of the offline-evaluation roadmap).

Synthetic performance matrices only. Pins the diagnostic's meaning: a genuinely skilled configuration
drives PBO low, pure noise drives it toward ~0.5, the logit sign matches the OOS-median rule, the run is
deterministic, it composes with the #6 purge/embargo, and the embargo sweep exposes boundary-sensitivity.
"""

import random
from math import comb

import pytest

from src.pbo import (
    PBOResult,
    PBOSensitivityPoint,
    pbo_embargo_sweep,
    probability_of_backtest_overfitting,
)


# --- shape --------------------------------------------------------------------


def test_number_of_splits_is_S_choose_half():
    perf = [[float(c) for c in range(4)] for _ in range(60)]
    res = probability_of_backtest_overfitting(perf, n_groups=6)
    assert res.n_splits == comb(6, 3) == 20
    assert res.n_configs == 4
    assert len(res.logits) == len(res.is_best_configs) == len(res.oos_relative_ranks) == 20


def test_result_is_frozen_dataclass():
    perf = [[float(c) for c in range(3)] for _ in range(40)]
    res = probability_of_backtest_overfitting(perf, n_groups=4)
    assert isinstance(res, PBOResult)
    with pytest.raises(Exception):
        res.pbo = 0.0  # type: ignore[misc]


# --- meaning ------------------------------------------------------------------


def test_one_config_dominates_everywhere_gives_zero_pbo():
    # Column 0 beats every other column on every observation -> IS best is always OOS best -> PBO 0.
    rng = random.Random(1)
    perf = []
    for _ in range(120):
        row = [rng.uniform(-1.0, 1.0) for _ in range(5)]
        row[0] = 100.0  # dominant, IS and OOS
        perf.append(row)
    res = probability_of_backtest_overfitting(perf, n_groups=6)
    assert res.pbo == 0.0
    assert set(res.is_best_configs) == {0}
    assert all(lam > 0 for lam in res.logits)


def test_perfectly_reversed_regimes_give_full_pbo():
    # Two regimes: in the first half config c scores +c, in the second half -c. Whatever wins IS is
    # engineered to be worst OOS (or exactly the median on balanced splits) -> lambda <= 0 everywhere.
    n = 120
    perf = []
    for t in range(n):
        first_half = t < n // 2
        perf.append([float(c) if first_half else float(-c) for c in range(4)])
    res = probability_of_backtest_overfitting(perf, n_groups=4)
    assert res.pbo == 1.0  # the canonical overfit signature


def test_pure_noise_mean_pbo_is_near_half():
    # Single-seed PBO has high variance (splits share groups); the *mean* over seeds must center on 0.5.
    pbos = []
    for seed in range(24):
        rng = random.Random(seed)
        perf = [[rng.gauss(0.0, 1.0) for _ in range(8)] for _ in range(160)]
        pbos.append(probability_of_backtest_overfitting(perf, n_groups=8).pbo)
    mean_pbo = sum(pbos) / len(pbos)
    assert 0.40 <= mean_pbo <= 0.60  # no real edge -> selection is ~coin-flip on average


def test_logit_sign_matches_oos_median_rule():
    rng = random.Random(7)
    perf = [[rng.gauss(0.0, 1.0) for _ in range(6)] for _ in range(90)]
    res = probability_of_backtest_overfitting(perf, n_groups=6)
    for lam, omega in zip(res.logits, res.oos_relative_ranks):
        assert (lam <= 0.0) == (omega <= 0.5)


# --- determinism --------------------------------------------------------------


def test_deterministic_across_runs():
    rng = random.Random(99)
    perf = [[rng.gauss(0.0, 1.0) for _ in range(5)] for _ in range(80)]
    a = probability_of_backtest_overfitting(perf, n_groups=6, horizon=2, embargo=3)
    b = probability_of_backtest_overfitting(perf, n_groups=6, horizon=2, embargo=3)
    assert a == b


# --- composes with #6 purge/embargo ------------------------------------------


def test_purge_embargo_are_passed_through_and_change_the_partition():
    # With heavy purge the IS sets shrink; the run must still be valid and may shift PBO.
    rng = random.Random(3)
    perf = [[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(120)]
    base = probability_of_backtest_overfitting(perf, n_groups=6, horizon=0, embargo=0)
    purged = probability_of_backtest_overfitting(perf, n_groups=6, horizon=3, embargo=2)
    assert 0.0 <= base.pbo <= 1.0 and 0.0 <= purged.pbo <= 1.0
    assert base.n_splits == purged.n_splits  # same combinatorics, different surviving IS bars


def test_empty_is_set_after_extreme_embargo_fails_closed():
    perf = [[float(c) for c in range(3)] for _ in range(40)]
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(perf, n_groups=4, embargo=1000)


# --- embargo sensitivity sweep (Q8) ------------------------------------------


def test_embargo_sweep_returns_a_point_per_value():
    rng = random.Random(11)
    perf = [[rng.gauss(0.0, 1.0) for _ in range(5)] for _ in range(150)]
    sweep = pbo_embargo_sweep(perf, n_groups=6, embargos=[0, 2, 4], horizon=1)
    assert [p.embargo for p in sweep] == [0, 2, 4]
    assert all(isinstance(p, PBOSensitivityPoint) and 0.0 <= p.pbo <= 1.0 for p in sweep)


def test_embargo_sweep_matches_direct_call():
    rng = random.Random(5)
    perf = [[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(100)]
    sweep = pbo_embargo_sweep(perf, n_groups=6, embargos=[3], horizon=2)
    direct = probability_of_backtest_overfitting(perf, n_groups=6, horizon=2, embargo=3)
    assert sweep[0].pbo == direct.pbo


# --- validation ---------------------------------------------------------------


def test_rejects_bad_parameters():
    good = [[float(c) for c in range(3)] for _ in range(40)]
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([], n_groups=4)              # empty
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[1.0]] * 40, n_groups=4)    # < 2 configs
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(good, n_groups=5)            # odd n_groups
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[1.0, 2.0], [3.0]], n_groups=2)  # ragged
    with pytest.raises(ValueError):
        pbo_embargo_sweep(good, n_groups=4, embargos=[])                 # empty grid
    with pytest.raises(ValueError):
        pbo_embargo_sweep(good, n_groups=4, embargos=[-1])               # negative embargo
