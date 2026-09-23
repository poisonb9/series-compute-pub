"""Position sizing: fixed-fractional default, de-rated capped Kelly option (engine #7, roadmap).

Consensus Q11 (`docs/CONSENSUS_Q11_..._ANSWER_20260722.md`) downgraded this engine. The claim "Kelly
beats fixed-fractional after noisy CV edges" is **WEAK** (almost no apples-to-apples test), while "full
Kelly is unsafe under estimation error" is **STRONG**. So:

- **the safe default is fixed-fractional** — risk a fixed, small fraction, independent of a noisy edge
  estimate;
- **Kelly is offered only heavily de-rated** (quarter-to-half), with an **uncertainty shrinkage** driven
  by out-of-sample confidence (the DSR of engine #2 / OOS stability, never ML), **volatility targeting**,
  a **regime gate**, and a **hard leverage cap** — all mandatory. Q11's cited live configuration is
  0.40-Kelly + 15% vol target + 2.0 max leverage.

Everything here is a pure, deterministic, stdlib-only function. It **fails closed**: a non-positive edge,
an unfavourable regime, or degenerate inputs size to **zero** (never a short, never a negative fraction).
It sizes nothing real — it computes a fraction from numbers a caller supplies. It opens no data, runs no
backtest, and authorizes nothing. Built and tested on constructed inputs (Tier 1).
"""

from __future__ import annotations

from dataclasses import dataclass

# Q11's cited safe live defaults (calibration anchors, not optimised values).
DEFAULT_KELLY_DERATING = 0.5     # half-Kelly
DEFAULT_MAX_LEVERAGE = 2.0       # hard cap
DEFAULT_VOL_TARGET = 0.15        # 15% target volatility


# --- primitive fractions -----------------------------------------------------


def fixed_fractional(risk_fraction: float) -> float:
    """The safe default: a fixed fraction of capital, independent of any edge estimate. Clamped to
    [0, 1]; a negative request fails closed to 0."""
    if risk_fraction != risk_fraction:  # NaN
        raise ValueError("risk_fraction must be a real number")
    return max(0.0, min(1.0, risk_fraction))


def kelly_fraction_binary(win_prob: float, win_loss_ratio: float) -> float:
    """Full Kelly fraction for a binary bet: f* = p - (1 - p) / b. A non-positive edge returns 0.0
    (fail-closed — no bet). ``win_loss_ratio`` (b) is the payoff of a win per unit risked."""
    if not 0.0 <= win_prob <= 1.0:
        raise ValueError("win_prob must be in [0, 1]")
    if win_loss_ratio <= 0.0:
        raise ValueError("win_loss_ratio must be > 0")
    f = win_prob - (1.0 - win_prob) / win_loss_ratio
    return max(0.0, f)


def kelly_fraction_moments(mean_return: float, variance: float) -> float:
    """Continuous Kelly fraction from return moments: f* = mean / variance. A non-positive mean or a
    non-positive variance returns 0.0 (fail-closed)."""
    if variance <= 0.0:
        return 0.0
    return max(0.0, mean_return / variance)


# --- de-rating controls ------------------------------------------------------


def derate(kelly_f: float, derating: float = DEFAULT_KELLY_DERATING) -> float:
    """Multiply a full-Kelly fraction by a de-rating in [0, 1] (0.5 = half-Kelly, 0.25 = quarter)."""
    if not 0.0 <= derating <= 1.0:
        raise ValueError("derating must be in [0, 1]")
    return max(0.0, kelly_f) * derating


def uncertainty_shrinkage(fraction: float, confidence: float) -> float:
    """Shrink toward zero by out-of-sample ``confidence`` in [0, 1] (from DSR/OOS stability, not ML).
    Confidence 1.0 leaves it unchanged; 0.0 sizes to zero. Q11: shrinkage adapts to forecast quality
    better than a fixed half."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return max(0.0, fraction) * confidence


def vol_target_scale(
    fraction: float, realized_vol: float, target_vol: float = DEFAULT_VOL_TARGET
) -> float:
    """Scale by target/realized volatility so the position runs at ``target_vol``. Higher realized
    volatility shrinks the size. Non-positive realized volatility fails closed to 0."""
    if target_vol <= 0.0:
        raise ValueError("target_vol must be > 0")
    if realized_vol <= 0.0:
        return 0.0
    return max(0.0, fraction) * (target_vol / realized_vol)


def apply_leverage_cap(fraction: float, max_leverage: float = DEFAULT_MAX_LEVERAGE) -> float:
    """Hard cap: the returned fraction never exceeds ``max_leverage`` (and never goes below 0)."""
    if max_leverage < 0.0:
        raise ValueError("max_leverage must be >= 0")
    return max(0.0, min(fraction, max_leverage))


# --- orchestration -----------------------------------------------------------


@dataclass(frozen=True)
class SizingResult:
    """A sized position and its audit trail.

    ``final_fraction`` is a NOTIONAL exposure fraction of capital, not a loss budget. The Kelly path
    computes ``f* = mean / variance``, whose textbook meaning is the fraction of wealth ALLOCATED, so
    values above 1.0 are leverage and are legitimate up to ``max_leverage`` -- that is precisely why the
    hard cap is named after leverage. Read as a loss budget it is wrong by orders of magnitude: 1.25 means
    1.25x exposure, NOT risking 125% of the account. To obtain a loss budget, combine this fraction with a
    stop distance; this module never sees a stop distance and therefore cannot express risk-of-loss.

    The intermediate stages are exposed so a reviewer can see exactly which control bound the size, and
    ``capped`` flags whether the hard leverage cap was the binding one."""

    method: str
    raw_fraction: float
    after_derating: float
    after_uncertainty: float
    after_vol_scale: float
    after_regime: float
    final_fraction: float
    capped: bool


def position_size(
    method: str = "fixed_fractional",
    *,
    risk_fraction: float = 0.01,
    kelly_f: float = 0.0,
    derating: float = DEFAULT_KELLY_DERATING,
    confidence: float = 1.0,
    realized_vol: float = DEFAULT_VOL_TARGET,
    target_vol: float = DEFAULT_VOL_TARGET,
    regime_scale: float = 1.0,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
) -> SizingResult:
    """Produce a final risk fraction with a full audit trail.

    ``method='fixed_fractional'`` (default) risks ``risk_fraction`` and ignores ``kelly_f`` — the safe
    default Q11 recommends. ``method='kelly'`` takes a **full**-Kelly ``kelly_f`` (from
    ``kelly_fraction_*``) and applies, in order, de-rating -> uncertainty shrinkage -> volatility
    targeting -> regime gate -> hard leverage cap. Every stage fails closed: a non-positive edge,
    zero confidence, unfavourable regime, or non-positive realized vol all drive the size to 0. Never
    returns a negative fraction (no implicit shorting)."""
    if method not in ("fixed_fractional", "kelly"):
        raise ValueError("method must be 'fixed_fractional' or 'kelly'")
    if not 0.0 <= regime_scale <= 1.0:
        raise ValueError("regime_scale must be in [0, 1]")

    if method == "fixed_fractional":
        raw = fixed_fractional(risk_fraction)
        after_regime = raw * regime_scale
        final = apply_leverage_cap(after_regime, max_leverage)
        return SizingResult(
            method=method, raw_fraction=raw, after_derating=raw, after_uncertainty=raw,
            after_vol_scale=raw, after_regime=after_regime, final_fraction=final,
            capped=after_regime > max_leverage,
        )

    raw = max(0.0, kelly_f)
    after_derating = derate(raw, derating)
    after_uncertainty = uncertainty_shrinkage(after_derating, confidence)
    after_vol_scale = vol_target_scale(after_uncertainty, realized_vol, target_vol)
    after_regime = after_vol_scale * regime_scale
    final = apply_leverage_cap(after_regime, max_leverage)
    return SizingResult(
        method=method, raw_fraction=raw, after_derating=after_derating,
        after_uncertainty=after_uncertainty, after_vol_scale=after_vol_scale,
        after_regime=after_regime, final_fraction=final, capped=after_regime > max_leverage,
    )
