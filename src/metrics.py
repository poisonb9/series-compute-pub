from __future__ import annotations

from dataclasses import dataclass
from math import fsum

from src.backtester import BacktestResult


@dataclass
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float | None
    payoff_ratio: float | None
    expectancy: float
    max_drawdown_amount: float
    max_drawdown_percent: float
    recovery_factor: float | None
    max_consecutive_wins: int
    max_consecutive_losses: int
    final_balance: float
    initial_balance: float
    return_percent: float


def calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    if not equity_curve:
        raise ValueError("equity_curve cannot be empty.")

    running_peak = equity_curve[0]
    max_drawdown_amount = 0.0
    max_drawdown_percent = 0.0

    for equity in equity_curve:
        if equity > running_peak:
            running_peak = equity

        drawdown_amount = running_peak - equity
        if drawdown_amount > max_drawdown_amount:
            max_drawdown_amount = drawdown_amount

        if running_peak > 0:
            drawdown_percent = (drawdown_amount / running_peak) * 100
            if drawdown_percent > max_drawdown_percent:
                max_drawdown_percent = drawdown_percent

    return max_drawdown_amount, max_drawdown_percent


def calculate_metrics(result: BacktestResult) -> BacktestMetrics:
    if result.initial_balance <= 0:
        raise ValueError("initial_balance must be greater than zero.")
    if not result.equity_curve:
        raise ValueError("equity_curve cannot be empty.")

    net_pnls = [trade.net_pnl for trade in result.trades]
    winning_pnls = [net_pnl for net_pnl in net_pnls if net_pnl > 0]
    losing_pnls = [net_pnl for net_pnl in net_pnls if net_pnl < 0]

    total_trades = len(result.trades)
    winning_trades = len(winning_pnls)
    losing_trades = len(losing_pnls)
    win_rate = winning_trades / total_trades if total_trades else 0.0

    gross_profit = fsum(winning_pnls)
    gross_loss = abs(fsum(losing_pnls))
    net_profit = fsum(net_pnls)

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    if winning_trades and losing_trades:
        average_win = gross_profit / winning_trades
        average_loss = gross_loss / losing_trades
        payoff_ratio = average_win / average_loss
    else:
        payoff_ratio = None

    expectancy = net_profit / total_trades if total_trades else 0.0
    max_drawdown_amount, max_drawdown_percent = calculate_max_drawdown(
        result.equity_curve
    )
    recovery_factor = (
        net_profit / max_drawdown_amount if max_drawdown_amount > 0 else None
    )

    max_consecutive_wins, max_consecutive_losses = _calculate_streaks(net_pnls)
    return_percent = (
        (result.final_balance - result.initial_balance) / result.initial_balance
    ) * 100

    return BacktestMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        expectancy=expectancy,
        max_drawdown_amount=max_drawdown_amount,
        max_drawdown_percent=max_drawdown_percent,
        recovery_factor=recovery_factor,
        max_consecutive_wins=max_consecutive_wins,
        max_consecutive_losses=max_consecutive_losses,
        final_balance=result.final_balance,
        initial_balance=result.initial_balance,
        return_percent=return_percent,
    )


def _calculate_streaks(net_pnls: list[float]) -> tuple[int, int]:
    current_wins = 0
    current_losses = 0
    max_wins = 0
    max_losses = 0

    for net_pnl in net_pnls:
        if net_pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif net_pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    return max_wins, max_losses
