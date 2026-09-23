"""Tests for the block-bootstrap general null (engine #3b of the offline-evaluation roadmap).

Synthetic return series only. Pins: the block length tracks the series' dependence, the resamplers
preserve length and draw only from the input, the whole run is deterministic given a seed, the matched
null charges costs and matches exposure, the p-value has the right sign against extreme observed values,
and every out-of-policy input fails closed.
"""

import random

import pytest

from src.block_bootstrap_baseline import (
    BootstrapNullResult,
    estimate_block_length,
    matched_block_bootstrap_null,
    moving_block_bootstrap_series,
    stationary_bootstrap_series,
)


# --- dependence-aware block length -------------------------------------------


def test_iid_series_gives_a_small_block():
    rng = random.Random(1)
    iid = [rng.gauss(0.0, 1.0) for _ in range(400)]
    # No dependence -> ACF collapses into the band almost immediately -> small block.
    assert estimate_block_length(iid) <= 6


def test_autocorrelated_series_gives_a_larger_block_than_iid():
    rng = random.Random(2)
    iid = [rng.gauss(0.0, 1.0) for _ in range(400)]
    # Strong AR(1): x_t = 0.9 x_{t-1} + noise -> slow ACF decay -> larger block.
    ar = [0.0]
    for _ in range(399):
        ar.append(0.9 * ar[-1] + rng.gauss(0.0, 1.0))
    assert estimate_block_length(ar) > estimate_block_length(iid)


def test_block_length_is_at_least_one_and_bounded():
    assert estimate_block_length([1.0, 2.0]) == 1          # n < 4
    vals = [float(i % 3) for i in range(50)]
    bl = estimate_block_length(vals)
    assert 1 <= bl <= len(vals) // 2


def test_constant_series_block_length_is_defined():
    # Zero variance -> ACF is 0 everywhere -> first noise lag is lag 1 -> block 2, still in bounds.
    bl = estimate_block_length([5.0] * 40)
    assert 1 <= bl <= 20


# --- stationary bootstrap ----------------------------------------------------


def test_stationary_preserves_length_and_draws_from_input():
    values = [float(i) for i in range(30)]
    out = stationary_bootstrap_series(values, expected_block=5, rng=random.Random(0))
    assert len(out) == len(values)
    assert set(out) <= set(values)


def test_stationary_is_deterministic_given_seed():
    values = [random.Random(9).gauss(0, 1) for _ in range(50)]
    a = stationary_bootstrap_series(values, 4, random.Random(7))
    b = stationary_bootstrap_series(values, 4, random.Random(7))
    assert a == b


def test_stationary_expected_block_one_restarts_every_step():
    # p = 1 -> a fresh random index at every step (pure IID resample), still valid and length-preserving.
    values = [float(i) for i in range(20)]
    out = stationary_bootstrap_series(values, expected_block=1, rng=random.Random(3))
    assert len(out) == len(values)
    assert set(out) <= set(values)


def test_stationary_rejects_bad_block():
    with pytest.raises(ValueError):
        stationary_bootstrap_series([1.0, 2.0], expected_block=0, rng=random.Random(0))
    with pytest.raises(ValueError):
        stationary_bootstrap_series([], expected_block=1, rng=random.Random(0))


# --- moving-block bootstrap --------------------------------------------------


def test_moving_preserves_length_and_draws_from_input():
    values = [float(i) for i in range(40)]
    out = moving_block_bootstrap_series(values, block_len=6, rng=random.Random(0))
    assert len(out) == len(values)
    assert set(out) <= set(values)


def test_moving_full_length_block_is_a_circular_rotation():
    values = [float(i) for i in range(12)]
    out = moving_block_bootstrap_series(values, block_len=len(values), rng=random.Random(4))
    # A single circular block of the whole series is just a rotation -> same multiset.
    assert sorted(out) == sorted(values)


def test_moving_is_deterministic_given_seed():
    values = [random.Random(1).gauss(0, 1) for _ in range(60)]
    a = moving_block_bootstrap_series(values, 5, random.Random(2))
    b = moving_block_bootstrap_series(values, 5, random.Random(2))
    assert a == b


def test_moving_rejects_bad_block():
    with pytest.raises(ValueError):
        moving_block_bootstrap_series([1.0, 2.0, 3.0], block_len=0, rng=random.Random(0))
    with pytest.raises(ValueError):
        moving_block_bootstrap_series([1.0, 2.0, 3.0], block_len=4, rng=random.Random(0))


# --- matched null: shape and determinism -------------------------------------


def _returns(n, seed, drift=0.0):
    rng = random.Random(seed)
    return [drift + rng.gauss(0.0, 0.01) for _ in range(n)]


def test_null_result_shape_and_frozen():
    res = matched_block_bootstrap_null(_returns(200, 1), n_trades=5, holding_bars=10,
                                       observed_stat=0.0, n_reps=100, seed=0)
    assert isinstance(res, BootstrapNullResult)
    assert len(res.null_stats) == res.n_reps == 100
    assert 0.0 <= res.p_value <= 1.0
    with pytest.raises(Exception):
        res.p_value = 0.0  # type: ignore[misc]


def test_null_is_deterministic():
    a = matched_block_bootstrap_null(_returns(200, 2), 6, 8, observed_stat=0.001,
                                     n_reps=150, method="stationary", seed=42)
    b = matched_block_bootstrap_null(_returns(200, 2), 6, 8, observed_stat=0.001,
                                     n_reps=150, method="stationary", seed=42)
    assert a == b


def test_both_methods_produce_valid_distributions():
    for method in ("stationary", "moving"):
        res = matched_block_bootstrap_null(_returns(240, 3), 8, 6, observed_stat=0.0,
                                           n_reps=120, block_len=5, method=method, seed=1)
        assert res.method == method
        assert all(isinstance(s, float) for s in res.null_stats)


# --- matched null: meaning ---------------------------------------------------


def test_extreme_high_observed_gives_small_p_value():
    # An observed statistic far above anything the zero-mean market produces -> tiny p-value.
    res = matched_block_bootstrap_null(_returns(300, 5), n_trades=6, holding_bars=10,
                                       observed_stat=100.0, n_reps=200, seed=0)
    assert res.p_value == pytest.approx(1 / 201)  # only the +1 smoothing


def test_extreme_low_observed_gives_p_value_near_one():
    res = matched_block_bootstrap_null(_returns(300, 6), n_trades=6, holding_bars=10,
                                       observed_stat=-100.0, n_reps=200, seed=0)
    assert res.p_value == pytest.approx(1.0)  # every replicate beats it


def test_p_value_at_null_mean_is_roughly_central():
    res = matched_block_bootstrap_null(_returns(300, 7), n_trades=5, holding_bars=10,
                                       observed_stat=0.0, n_reps=400, seed=3)
    # A zero-drift market against a zero observed statistic -> p should sit around the middle.
    assert 0.2 <= res.p_value <= 0.8


def test_cost_lowers_the_null_distribution():
    free = matched_block_bootstrap_null(_returns(300, 8), 6, 10, observed_stat=0.0,
                                        cost_per_trade=0.0, n_reps=200, seed=4)
    charged = matched_block_bootstrap_null(_returns(300, 8), 6, 10, observed_stat=0.0,
                                           cost_per_trade=0.05, n_reps=200, seed=4)
    # Charging a per-trade cost shifts every trade's return down -> lower mean null statistic.
    assert charged.null_mean < free.null_mean


def test_sharpe_statistic_runs_and_differs_from_mean():
    mean_res = matched_block_bootstrap_null(_returns(300, 9), 6, 10, observed_stat=0.0,
                                            statistic="mean", n_reps=150, seed=2)
    sharpe_res = matched_block_bootstrap_null(_returns(300, 9), 6, 10, observed_stat=0.0,
                                              statistic="sharpe", n_reps=150, seed=2)
    assert sharpe_res.statistic == "sharpe"
    assert sharpe_res.null_stats != mean_res.null_stats


def test_default_block_len_is_the_dependence_estimate():
    rets = _returns(300, 10)
    res = matched_block_bootstrap_null(rets, 6, 10, observed_stat=0.0, n_reps=50, seed=0)
    assert res.block_len == estimate_block_length(rets)


# --- matched null: fail-closed -----------------------------------------------


def test_series_too_short_fails_closed():
    with pytest.raises(ValueError):
        matched_block_bootstrap_null(_returns(50, 1), n_trades=10, holding_bars=10,
                                     observed_stat=0.0)  # 10*10 > 50


def test_rejects_bad_parameters():
    good = _returns(200, 1)
    with pytest.raises(ValueError):
        matched_block_bootstrap_null([], 1, 1, observed_stat=0.0)
    with pytest.raises(ValueError):
        matched_block_bootstrap_null(good, 0, 10, observed_stat=0.0)          # n_trades < 1
    with pytest.raises(ValueError):
        matched_block_bootstrap_null(good, 5, 0, observed_stat=0.0)           # holding < 1
    with pytest.raises(ValueError):
        matched_block_bootstrap_null(good, 5, 10, observed_stat=0.0, n_reps=0)
    with pytest.raises(ValueError):
        matched_block_bootstrap_null(good, 5, 10, observed_stat=0.0, method="iid")
    with pytest.raises(ValueError):
        matched_block_bootstrap_null(good, 5, 10, observed_stat=0.0, statistic="median")
    with pytest.raises(ValueError):
        matched_block_bootstrap_null(good, 5, 10, observed_stat=0.0, block_len=0)
