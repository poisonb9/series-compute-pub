"""Multiplicity-adjusted significance for strategy evaluation (research/offline).

Engine #2 of the offline-evaluation roadmap: the anti-false-edge core. When you test many strategies,
markets, parameters and windows, the best-looking Sharpe is inflated by selection. This module gives
the deterministic statistics that deflate it:

- Probabilistic Sharpe Ratio (PSR) — Bailey & Lopez de Prado: the probability that the true Sharpe
  exceeds a benchmark, given sample length, skewness and kurtosis of the returns.
- Deflated Sharpe Ratio (DSR) — PSR against a benchmark that accounts for the NUMBER OF TRIALS, so an
  edge that is merely the max of many noisy trials is exposed.
- Bonferroni / Sidak family-wise alpha adjustments for the portfolio multiplicity budget.

Pure math, deterministic, stdlib only. Research/offline: it computes significance, it does not open
data, run a backtest, assign a role, or authorize anything. Built and tested on constructed inputs
(Tier 1); using it on real strategy returns is part of Tier-3 evaluation.

References: Bailey, D. & Lopez de Prado, M. (2014) "The Deflated Sharpe Ratio".
"""

from __future__ import annotations

import math

EULER_MASCHERONI = 0.5772156649015329


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, ~1e-9 accuracy)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe_ratio(
    sharpe: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    """P(true Sharpe > benchmark). ``sharpe`` is the per-observation Sharpe (mean/std of returns),
    ``kurtosis`` is the 4th standardised moment (normal = 3). Returns a probability in (0, 1)."""
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    denom = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe
    if denom <= 0:
        # Heavy tails make the estimate unstable; fail closed to "not significant".
        return 0.0
    z = (sharpe - sharpe_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return norm_cdf(z)


def expected_max_sharpe(sharpe_variance: float, num_trials: int) -> float:
    """Expected maximum Sharpe across ``num_trials`` independent noise trials (the DSR benchmark).

    E[max] ~= sqrt(Var(SR)) * ((1-g)*Z(1-1/N) + g*Z(1-1/(N*e))), g = Euler-Mascheroni."""
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance must be >= 0")
    if num_trials == 1:
        return 0.0
    sd = math.sqrt(sharpe_variance)
    z1 = norm_ppf(1.0 - 1.0 / num_trials)
    z2 = norm_ppf(1.0 - 1.0 / (num_trials * math.e))
    return sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    sharpe: float, n_obs: int, num_trials: int, sharpe_variance: float,
    skew: float = 0.0, kurtosis: float = 3.0,
) -> float:
    """P(true Sharpe > the trials-adjusted benchmark). DSR < PSR whenever num_trials > 1 —
    the more strategies you tried, the higher the bar the winner must clear."""
    benchmark = expected_max_sharpe(sharpe_variance, num_trials)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, sharpe_benchmark=benchmark)


def bonferroni_alpha(alpha: float, num_tests: int) -> float:
    """Family-wise significance per test under Bonferroni (conservative)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if num_tests < 1:
        raise ValueError("num_tests must be >= 1")
    return alpha / num_tests


def sidak_alpha(alpha: float, num_tests: int) -> float:
    """Family-wise significance per test under Sidak (slightly less conservative than Bonferroni)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if num_tests < 1:
        raise ValueError("num_tests must be >= 1")
    return 1.0 - (1.0 - alpha) ** (1.0 / num_tests)
