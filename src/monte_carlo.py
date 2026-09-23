from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.backtester import BacktestResult
from src.metrics import BacktestMetrics


class MonteCarloStatus(Enum):
    ROBUST_CANDIDATE = "robust_candidate"
    FRAGILE = "fragile"
    NOT_ENOUGH_TRADES = "not_enough_trades"
    INVALID = "invalid"


@dataclass(frozen=True)
class MonteCarloConfig:
    iterations: int = 500
    seed: int = 42
    min_trades: int = 20
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be greater than zero.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer.")
        if self.min_trades < 1:
            raise ValueError("min_trades must be greater than or equal to one.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one.")


@dataclass(frozen=True)
class MonteCarloSimulation:
    iteration: int
    final_balance: float
    net_profit: float
    return_percent: float
    max_drawdown_amount: float
    max_drawdown_percent: float
    trades_count: int


@dataclass(frozen=True)
class MonteCarloSummary:
    status: MonteCarloStatus
    iterations: int
    original_net_profit: float
    original_return_percent: float
    median_net_profit: float
    p05_net_profit: float
    p95_net_profit: float
    probability_of_profit: float
    median_max_drawdown_percent: float
    p95_max_drawdown_percent: float
    worst_max_drawdown_percent: float
    warnings: list[str]
    errors: list[str]
    simulations: list[MonteCarloSimulation]

    @property
    def is_usable(self) -> bool:
        return self.status in {
            MonteCarloStatus.ROBUST_CANDIDATE,
            MonteCarloStatus.FRAGILE,
        }


def run_trade_bootstrap_monte_carlo(
    result: BacktestResult,
    metrics: BacktestMetrics,
    config: MonteCarloConfig | None = None,
) -> MonteCarloSummary:
    cfg = config or MonteCarloConfig()
    errors: list[str] = []
    warnings: list[str] = []

    if result.initial_balance <= 0 or metrics.initial_balance <= 0:
        errors.append("initial_balance must be greater than zero.")
    if not result.equity_curve:
        errors.append("equity_curve cannot be empty.")
    if metrics.total_trades != len(result.trades):
        errors.append("metrics.total_trades does not match result.trades.")

    if errors:
        return _empty_summary(MonteCarloStatus.INVALID, metrics, cfg, warnings, errors)

    net_pnls = [trade.net_pnl for trade in result.trades]
    if not net_pnls:
        warnings.append("No trades available for Monte Carlo bootstrap.")
        return _empty_summary(
            MonteCarloStatus.NOT_ENOUGH_TRADES,
            metrics,
            cfg,
            warnings,
            errors,
        )

    if len(net_pnls) < cfg.min_trades:
        warnings.append(
            f"Trades count ({len(net_pnls)}) is below min_trades ({cfg.min_trades})."
        )

    rng = random.Random(cfg.seed)
    simulations = [
        _run_single_simulation(
            iteration=iteration,
            initial_balance=result.initial_balance,
            sampled_pnls=[rng.choice(net_pnls) for _ in net_pnls],
        )
        for iteration in range(1, cfg.iterations + 1)
    ]

    net_profits = [simulation.net_profit for simulation in simulations]
    drawdown_percents = [
        simulation.max_drawdown_percent for simulation in simulations
    ]
    probability_of_profit = (
        sum(1 for net_profit in net_profits if net_profit > 0) / len(net_profits)
    )
    p05_net_profit = calculate_percentile(net_profits, 0.05)

    if len(net_pnls) < cfg.min_trades:
        status = MonteCarloStatus.NOT_ENOUGH_TRADES
    elif probability_of_profit >= 0.70 and p05_net_profit > 0:
        status = MonteCarloStatus.ROBUST_CANDIDATE
    else:
        status = MonteCarloStatus.FRAGILE

    return MonteCarloSummary(
        status=status,
        iterations=cfg.iterations,
        original_net_profit=metrics.net_profit,
        original_return_percent=metrics.return_percent,
        median_net_profit=calculate_percentile(net_profits, 0.50),
        p05_net_profit=p05_net_profit,
        p95_net_profit=calculate_percentile(net_profits, 0.95),
        probability_of_profit=probability_of_profit,
        median_max_drawdown_percent=calculate_percentile(drawdown_percents, 0.50),
        p95_max_drawdown_percent=calculate_percentile(drawdown_percents, 0.95),
        worst_max_drawdown_percent=max(drawdown_percents),
        warnings=warnings,
        errors=errors,
        simulations=simulations,
    )


def export_monte_carlo_simulations_csv(
    summary: MonteCarloSummary,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "final_balance",
        "net_profit",
        "return_percent",
        "max_drawdown_amount",
        "max_drawdown_percent",
        "trades_count",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for simulation in summary.simulations:
            writer.writerow(
                {
                    "iteration": simulation.iteration,
                    "final_balance": simulation.final_balance,
                    "net_profit": simulation.net_profit,
                    "return_percent": simulation.return_percent,
                    "max_drawdown_amount": simulation.max_drawdown_amount,
                    "max_drawdown_percent": simulation.max_drawdown_percent,
                    "trades_count": simulation.trades_count,
                }
            )

    return output_path


def generate_monte_carlo_markdown_summary(
    summary: MonteCarloSummary,
    *,
    title: str = "Monte Carlo Bootstrap Summary",
    notes: str | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "> Research/backtest-only report. This is not live trading, not an order, and not investment advice.",
        "",
        "Monte Carlo bootstrap does not prove edge and does not authorize live trading.",
        "",
        "## Summary",
        "",
        f"- Status: {summary.status.value}",
        f"- Iterations: {summary.iterations}",
        f"- Original net profit: {summary.original_net_profit:.2f}",
        f"- Original return percent: {summary.original_return_percent:.2f}%",
        f"- Median net profit: {summary.median_net_profit:.2f}",
        f"- P05 net profit: {summary.p05_net_profit:.2f}",
        f"- P95 net profit: {summary.p95_net_profit:.2f}",
        f"- Probability of profit: {summary.probability_of_profit:.2%}",
        f"- Median max drawdown percent: {summary.median_max_drawdown_percent:.2f}%",
        f"- P95 max drawdown percent: {summary.p95_max_drawdown_percent:.2f}%",
        f"- Worst max drawdown percent: {summary.worst_max_drawdown_percent:.2f}%",
        "",
        "## Warnings",
        "",
    ]

    if summary.warnings:
        lines.extend(f"- {warning}" for warning in summary.warnings)
    else:
        lines.append("No warnings.")

    lines.extend(["", "## Errors", ""])
    if summary.errors:
        lines.extend(f"- {error}" for error in summary.errors)
    else:
        lines.append("No errors.")

    if notes:
        lines.extend(["", "## Notes", "", notes.strip()])

    return "\n".join(lines).rstrip() + "\n"


def calculate_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values cannot be empty.")
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between zero and one.")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def _run_single_simulation(
    *,
    iteration: int,
    initial_balance: float,
    sampled_pnls: list[float],
) -> MonteCarloSimulation:
    equity_curve = [initial_balance]
    balance = initial_balance

    for pnl in sampled_pnls:
        balance += pnl
        equity_curve.append(balance)

    max_drawdown_amount, max_drawdown_percent = _calculate_max_drawdown(equity_curve)
    net_profit = balance - initial_balance
    return_percent = (net_profit / initial_balance) * 100

    return MonteCarloSimulation(
        iteration=iteration,
        final_balance=balance,
        net_profit=net_profit,
        return_percent=return_percent,
        max_drawdown_amount=max_drawdown_amount,
        max_drawdown_percent=max_drawdown_percent,
        trades_count=len(sampled_pnls),
    )


def _calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    running_peak = equity_curve[0]
    max_drawdown_amount = 0.0
    max_drawdown_percent = 0.0

    for equity in equity_curve:
        if equity > running_peak:
            running_peak = equity

        drawdown_amount = running_peak - equity
        max_drawdown_amount = max(max_drawdown_amount, drawdown_amount)
        if running_peak > 0:
            max_drawdown_percent = max(
                max_drawdown_percent,
                (drawdown_amount / running_peak) * 100,
            )

    return max_drawdown_amount, max_drawdown_percent


def _empty_summary(
    status: MonteCarloStatus,
    metrics: BacktestMetrics,
    config: MonteCarloConfig,
    warnings: list[str],
    errors: list[str],
) -> MonteCarloSummary:
    return MonteCarloSummary(
        status=status,
        iterations=config.iterations,
        original_net_profit=metrics.net_profit,
        original_return_percent=metrics.return_percent,
        median_net_profit=0.0,
        p05_net_profit=0.0,
        p95_net_profit=0.0,
        probability_of_profit=0.0,
        median_max_drawdown_percent=0.0,
        p95_max_drawdown_percent=0.0,
        worst_max_drawdown_percent=0.0,
        warnings=warnings,
        errors=errors,
        simulations=[],
    )
