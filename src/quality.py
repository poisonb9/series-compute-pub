from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import fsum

from src.backtester import BacktestResult
from src.metrics import BacktestMetrics


class BacktestQualityStatus(Enum):
    VALID_FOR_PIPELINE_TEST = "valid_for_pipeline_test"
    INTERPRETABLE_WITH_WARNINGS = "interpretable_with_warnings"
    NOT_INTERPRETABLE = "not_interpretable"
    INVALID = "invalid"


@dataclass(frozen=True)
class BacktestQualityReport:
    status: BacktestQualityStatus
    errors: list[str]
    warnings: list[str]
    passed_checks: list[str]

    @property
    def is_usable(self) -> bool:
        return self.status in {
            BacktestQualityStatus.VALID_FOR_PIPELINE_TEST,
            BacktestQualityStatus.INTERPRETABLE_WITH_WARNINGS,
        }


def evaluate_backtest_quality(
    result: BacktestResult,
    metrics: BacktestMetrics,
    *,
    min_trades_for_interpretation: int = 30,
    require_fees: bool = True,
    research_only_text: str | None = None,
) -> BacktestQualityReport:
    errors: list[str] = []
    warnings: list[str] = []
    passed_checks: list[str] = []

    if result.initial_balance <= 0 or metrics.initial_balance <= 0:
        errors.append("initial_balance must be greater than zero.")
    else:
        passed_checks.append("initial_balance_positive")

    if not result.equity_curve:
        errors.append("equity_curve cannot be empty.")
    else:
        passed_checks.append("equity_curve_present")
        if _is_close(result.final_balance, result.equity_curve[-1]):
            passed_checks.append("final_balance_matches_equity_curve")
        else:
            errors.append("final_balance does not match the last equity curve point.")

    _check_metric_consistency(result, metrics, errors, passed_checks)

    if not result.trades:
        warnings.append("zero trades; backtest is not interpretable.")
    elif metrics.total_trades < min_trades_for_interpretation:
        warnings.append(
            "total trades "
            f"({metrics.total_trades}) is below min_trades_for_interpretation "
            f"({min_trades_for_interpretation})."
        )
    else:
        passed_checks.append("minimum_trade_sample_reached")

    if metrics.profit_factor is None:
        warnings.append("profit_factor is undefined.")
    else:
        passed_checks.append("profit_factor_defined")

    if metrics.payoff_ratio is None:
        warnings.append("payoff_ratio is undefined.")
    else:
        passed_checks.append("payoff_ratio_defined")

    if require_fees and result.trades:
        if all(_is_close(trade.fees, 0.0) for trade in result.trades):
            warnings.append("fees are zero while require_fees=True.")
        else:
            passed_checks.append("fees_present")

    if research_only_text is not None:
        if _has_research_only_warning(research_only_text):
            passed_checks.append("research_only_warning_present")
        else:
            warnings.append(
                "research/backtest-only safety text is missing or insufficient."
            )

    status = _classify_status(errors, result.trades, metrics.total_trades, min_trades_for_interpretation)

    return BacktestQualityReport(
        status=status,
        errors=errors,
        warnings=warnings,
        passed_checks=passed_checks,
    )


def _check_metric_consistency(
    result: BacktestResult,
    metrics: BacktestMetrics,
    errors: list[str],
    passed_checks: list[str],
) -> None:
    net_pnls = [trade.net_pnl for trade in result.trades]
    winning_trades = sum(1 for net_pnl in net_pnls if net_pnl > 0)
    losing_trades = sum(1 for net_pnl in net_pnls if net_pnl < 0)
    gross_profit = fsum(net_pnl for net_pnl in net_pnls if net_pnl > 0)
    gross_loss = abs(fsum(net_pnl for net_pnl in net_pnls if net_pnl < 0))
    net_profit = fsum(net_pnls)

    metric_errors = []

    if metrics.total_trades != len(result.trades):
        metric_errors.append("metrics.total_trades does not match result.trades.")
    if metrics.winning_trades != winning_trades:
        metric_errors.append("metrics.winning_trades does not match result.trades.")
    if metrics.losing_trades != losing_trades:
        metric_errors.append("metrics.losing_trades does not match result.trades.")
    if not _is_close(metrics.gross_profit, gross_profit):
        metric_errors.append("metrics.gross_profit does not match result.trades.")
    if not _is_close(metrics.gross_loss, gross_loss):
        metric_errors.append("metrics.gross_loss does not match result.trades.")
    if not _is_close(metrics.net_profit, net_profit):
        metric_errors.append("metrics.net_profit does not match result.trades.")
    if not _is_close(metrics.initial_balance, result.initial_balance):
        metric_errors.append("metrics.initial_balance does not match result.")
    if not _is_close(metrics.final_balance, result.final_balance):
        metric_errors.append("metrics.final_balance does not match result.")

    if metric_errors:
        errors.extend(metric_errors)
    else:
        passed_checks.append("metrics_match_result")


def _classify_status(
    errors: list[str],
    trades: object,
    total_trades: int,
    min_trades_for_interpretation: int,
) -> BacktestQualityStatus:
    if errors:
        return BacktestQualityStatus.INVALID
    if not trades:
        return BacktestQualityStatus.NOT_INTERPRETABLE
    if total_trades < min_trades_for_interpretation:
        return BacktestQualityStatus.VALID_FOR_PIPELINE_TEST
    return BacktestQualityStatus.INTERPRETABLE_WITH_WARNINGS


def _has_research_only_warning(text: str) -> bool:
    normalized = text.lower()
    return "research/backtest-only" in normalized or (
        "research" in normalized and "backtest-only" in normalized
    )


def _is_close(left: float, right: float) -> bool:
    tolerance = 1e-6 * max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance
