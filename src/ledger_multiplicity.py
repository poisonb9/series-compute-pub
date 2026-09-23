"""Cross-session multiplicity / alpha budget ledger (research/offline).

Consensus Q16 (`docs/CONSENSUS_Q16_..._ANSWER_20260723.md`) established, with strong evidence, that a
single frozen confirmatory holdout must be treated as **one lifetime multiple-testing family across
sessions**: every test ever run against it counts, unreported trials are the central source of silent
mining, and an unknown future stream of candidates calls for **online** error control, not per-session
resets. This engine implements that discipline as deterministic, stdlib-only accounting:

1. **A persistent, append-only trial ledger** — every test (candidate x market x timeframe x params) is
   logged with its p-value and decision, JSON-serialisable, never dropped (dropping a prior attempt is
   silent optional stopping). Its trial count is the multiplicity number that feeds the Deflated Sharpe
   (#2) and PBO (#9) deflation (Q12 gap #1).
2. **FWER alpha-spending** — a pre-declared total budget split into slices summing to <= alpha; provably
   controls the family-wise error (union bound); hard-stops when exhausted. Simple and unimpeachable, the
   conservative default Q16 endorses when few strong candidates are expected.
3. **LORD++ online FDR** (Ramdas, Yang, Wainwright & Jordan, 2017) — the alpha-wealth rule for an unknown
   stream: the test level rises after rejections and decays otherwise, controlling online FDR without
   knowing the number or timing of future tests.
4. **A reserve recommendation** — Q16's "reserve most of the budget for one or two strong finalists"
   made explicit against the current ledger.

It runs no test, opens no data, assigns no dataset role, and authorises nothing. It records and budgets;
the decision to spend remains human. Built and tested on constructed p-value streams (Tier 1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# --- persistent, append-only trial ledger ------------------------------------


@dataclass(frozen=True)
class Trial:
    """One test ever run against the confirmatory holdout. Immutable once logged."""

    trial_id: str
    label: str          # e.g. "connors-rsi2 | BTC | 1d | p=RSI2<10"
    p_value: float
    rejected: bool
    alpha_level: float  # the threshold this trial was judged at (spending slice or LORD++ level)


@dataclass
class TrialLedger:
    """Append-only lifetime family of confirmatory tests. Serialisable; never drops a trial."""

    holdout_id: str
    total_alpha: float
    trials: list[Trial] = field(default_factory=list)

    def add(self, trial: Trial) -> None:
        if any(t.trial_id == trial.trial_id for t in self.trials):
            raise ValueError(f"duplicate trial_id {trial.trial_id!r}: a trial is never overwritten")
        self.trials.append(trial)

    @property
    def multiplicity_count(self) -> int:
        """Number of tests ever run — the trial count that deflates DSR (#2) / PBO (#9)."""
        return len(self.trials)

    @property
    def alpha_spent(self) -> float:
        """Total alpha allocated across all logged trials (FWER accounting)."""
        return sum(t.alpha_level for t in self.trials)

    @property
    def alpha_remaining(self) -> float:
        return self.total_alpha - self.alpha_spent

    def to_dict(self) -> dict:
        return {
            "holdout_id": self.holdout_id,
            "total_alpha": self.total_alpha,
            "trials": [asdict(t) for t in self.trials],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrialLedger":
        led = cls(holdout_id=d["holdout_id"], total_alpha=d["total_alpha"])
        for t in d.get("trials", []):
            led.trials.append(Trial(**t))
        return led


# --- FWER alpha-spending ------------------------------------------------------


def alpha_spending_ok(ledger: TrialLedger, requested_slice: float) -> bool:
    """True iff spending ``requested_slice`` on the next test keeps total spend <= total_alpha.

    A False result is the hard stop: the family budget is exhausted and the holdout must be retired /
    renewed with genuinely fresh data before more testing (Q16)."""
    if requested_slice < 0.0:
        raise ValueError("requested_slice must be >= 0")
    return ledger.alpha_spent + requested_slice <= ledger.total_alpha + 1e-12


# --- LORD++ online FDR --------------------------------------------------------

_GAMMA_EXPONENT = 1.6
_GAMMA_HORIZON = 100_000
# gamma_j = j^-1.6 / Z with Z summed over a long fixed horizon, so the sequence is fixed in advance and
# sums to <= 1 (the truncated tail only makes it more conservative). Computed once at import.
_GAMMA_Z = sum(j ** (-_GAMMA_EXPONENT) for j in range(1, _GAMMA_HORIZON + 1))


def _gamma(k: int) -> float:
    """LORD++ weight for lag k (k>=1); 0 for k<=0. Non-increasing, sums to <= 1."""
    if k <= 0:
        return 0.0
    if k > _GAMMA_HORIZON:
        # Beyond the normalization horizon the weight is defined as 0 so the full
        # sequence sums to exactly Z/Z = 1 (LM-4 fix): _GAMMA_Z only accumulates
        # j in [1, _GAMMA_HORIZON], so a non-zero tail here would push the infinite
        # sum above 1 and break the LORD++ FDR-control condition.
        return 0.0
    return (k ** (-_GAMMA_EXPONENT)) / _GAMMA_Z


@dataclass(frozen=True)
class LordResult:
    """Per-test online-FDR levels and decisions for a p-value stream."""

    alpha_levels: tuple[float, ...]
    rejected: tuple[bool, ...]
    n_rejections: int


def lord_plus_plus(
    p_values: list[float], alpha: float = 0.05, w0: float | None = None
) -> LordResult:
    """LORD++ online FDR over a sequence of p-values.

    At test ``t`` the level is
    ``alpha_t = w0*gamma_t + (alpha-w0)*gamma_{t-tau_1} + alpha*sum_{j>=2} gamma_{t-tau_j}``
    where ``tau_1, tau_2, ...`` are the earlier rejection times; test ``t`` is rejected iff
    ``p_t <= alpha_t``. ``w0`` (initial alpha-wealth, default ``alpha/2``) must lie in ``(0, alpha]``.
    Deterministic; controls online FDR at level ``alpha`` for independent nulls (Ramdas et al., 2017)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if w0 is None:
        w0 = alpha / 2.0
    if not 0.0 < w0 <= alpha:
        raise ValueError("w0 must be in (0, alpha]")
    if any(not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p-values must be in [0, 1]")

    rej_times: list[int] = []
    levels: list[float] = []
    rejected: list[bool] = []
    for t in range(1, len(p_values) + 1):
        a = w0 * _gamma(t)
        if rej_times:
            a += (alpha - w0) * _gamma(t - rej_times[0])
            for tau in rej_times[1:]:
                a += alpha * _gamma(t - tau)
        levels.append(a)
        is_rej = p_values[t - 1] <= a
        rejected.append(is_rej)
        if is_rej:
            rej_times.append(t)
    return LordResult(
        alpha_levels=tuple(levels),
        rejected=tuple(rejected),
        n_rejections=len(rej_times),
    )


# --- reserve recommendation (Q16) --------------------------------------------


@dataclass(frozen=True)
class ReserveRecommendation:
    """Whether to spend now or reserve, per Q16's 'reserve for one or two strong finalists'."""

    can_afford: bool
    alpha_remaining: float
    reserve_advised: bool
    reason: str


def reserve_recommendation(
    ledger: TrialLedger, requested_slice: float, expected_strong_finalists_remaining: int
) -> ReserveRecommendation:
    """Advise spending vs reserving. Q16: with few strong candidates, reserve most budget for the
    strongest 1-2 finalists rather than slicing it across weak arrivals; renew the holdout with fresh
    data instead of over-spending."""
    remaining = ledger.alpha_remaining
    affordable = alpha_spending_ok(ledger, requested_slice)
    if not affordable:
        return ReserveRecommendation(
            can_afford=False, alpha_remaining=remaining, reserve_advised=True,
            reason="budget exhausted: retire and renew the holdout with genuinely fresh data (Q16)",
        )
    # if the remaining budget barely covers the strong finalists still expected, don't spend on a weak one
    needed_for_finalists = requested_slice * max(1, expected_strong_finalists_remaining)
    if needed_for_finalists > remaining:
        return ReserveRecommendation(
            can_afford=True, alpha_remaining=remaining, reserve_advised=True,
            reason=("affordable now, but reserving is advised: remaining budget is needed for the "
                    f"{expected_strong_finalists_remaining} strong finalists still expected (Q16)"),
        )
    return ReserveRecommendation(
        can_afford=True, alpha_remaining=remaining, reserve_advised=False,
        reason="affordable and leaves enough budget for expected strong finalists",
    )
