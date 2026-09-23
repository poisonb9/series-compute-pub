"""Tests for multiplicity-adjusted significance (Deflated Sharpe + FWER adjustments).

Constructed inputs only; no data opened, no strategy run. Pins the math against known identities and
monotonicities so the anti-false-edge core cannot silently drift.
"""

import math

import pytest

from src.multiplicity_significance import (
    bonferroni_alpha,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    norm_cdf,
    norm_ppf,
    probabilistic_sharpe_ratio,
    sidak_alpha,
)


# --- Normal helpers -----------------------------------------------------------


def test_norm_cdf_known_points():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_norm_ppf_is_inverse_of_cdf():
    for x in (-2.5, -1.0, -0.3, 0.0, 0.7, 1.5, 2.8):
        assert norm_ppf(norm_cdf(x)) == pytest.approx(x, abs=1e-6)


def test_norm_ppf_rejects_out_of_range():
    for p in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            norm_ppf(p)


# --- Probabilistic Sharpe Ratio ----------------------------------------------


def test_psr_half_at_benchmark():
    # Observed Sharpe exactly at the benchmark -> 50/50.
    assert probabilistic_sharpe_ratio(0.1, n_obs=100, sharpe_benchmark=0.1) == pytest.approx(0.5)


def test_psr_increases_with_sample_length():
    a = probabilistic_sharpe_ratio(0.1, n_obs=50)
    b = probabilistic_sharpe_ratio(0.1, n_obs=500)
    assert b > a > 0.5


def test_psr_increases_with_sharpe():
    lo = probabilistic_sharpe_ratio(0.05, n_obs=250)
    hi = probabilistic_sharpe_ratio(0.20, n_obs=250)
    assert hi > lo


def test_psr_negative_sharpe_below_half():
    assert probabilistic_sharpe_ratio(-0.1, n_obs=250) < 0.5


def test_psr_fails_closed_on_unstable_denominator():
    # Extreme skew/kurtosis can drive the denominator non-positive -> fail closed to 0.
    assert probabilistic_sharpe_ratio(0.5, n_obs=100, skew=5.0, kurtosis=3.0) == 0.0


def test_psr_rejects_tiny_sample():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, n_obs=1)


# --- Deflated Sharpe / expected max ------------------------------------------


def test_expected_max_sharpe_zero_for_single_trial():
    assert expected_max_sharpe(0.04, num_trials=1) == 0.0


def test_expected_max_sharpe_grows_with_trials():
    a = expected_max_sharpe(0.04, num_trials=10)
    b = expected_max_sharpe(0.04, num_trials=1000)
    assert b > a > 0.0


def test_dsr_is_below_psr_when_many_trials():
    kw = dict(sharpe=0.15, n_obs=250)
    psr = probabilistic_sharpe_ratio(**kw)
    dsr = deflated_sharpe_ratio(**kw, num_trials=100, sharpe_variance=0.04)
    assert dsr < psr


def test_dsr_equals_psr_for_single_trial():
    kw = dict(sharpe=0.15, n_obs=250)
    psr = probabilistic_sharpe_ratio(**kw)
    dsr = deflated_sharpe_ratio(**kw, num_trials=1, sharpe_variance=0.04)
    assert dsr == pytest.approx(psr)


def test_more_trials_lower_dsr():
    kw = dict(sharpe=0.15, n_obs=250, sharpe_variance=0.04)
    assert deflated_sharpe_ratio(**kw, num_trials=1000) < deflated_sharpe_ratio(**kw, num_trials=10)


# --- Family-wise alpha --------------------------------------------------------


def test_bonferroni_divides_alpha():
    assert bonferroni_alpha(0.05, 5) == pytest.approx(0.01)


def test_sidak_close_to_bonferroni_but_slightly_larger():
    b = bonferroni_alpha(0.05, 10)
    s = sidak_alpha(0.05, 10)
    assert s > b
    assert s == pytest.approx(1 - (1 - 0.05) ** (1 / 10))


def test_sidak_identity_single_test():
    assert sidak_alpha(0.05, 1) == pytest.approx(0.05)


@pytest.mark.parametrize("fn", [bonferroni_alpha, sidak_alpha])
def test_alpha_adjusters_validate(fn):
    with pytest.raises(ValueError):
        fn(0.0, 5)
    with pytest.raises(ValueError):
        fn(0.05, 0)
