from datetime import datetime, timedelta

import pytest

from src.backtester import BacktestResult, Trade, TradeDirection
from src.metrics import BacktestMetrics, calculate_metrics


def _trade(net_pnl: float, index: int) -> Trade:
    entry_timestamp = datetime(2026, 1, 1) + timedelta(minutes=index * 2)
    exit_timestamp = entry_timestamp + timedelta(minutes=1)
    return Trade(
        direction=TradeDirection.LONG,
        entry_timestamp=entry_timestamp,
        exit_timestamp=exit_timestamp,
        entry_price=100.0,
        exit_price=100.0 + net_pnl,
        quantity=1.0,
        gross_pnl=net_pnl,
        fees=0.0,
        net_pnl=net_pnl,
        exit_reason="test",
    )


def _mixed_result() -> BacktestResult:
    net_pnls = [100.0, 50.0, -40.0, -60.0, 30.0, -20.0]
    return BacktestResult(
        trades=[_trade(net_pnl, index) for index, net_pnl in enumerate(net_pnls)],
        equity_curve=[1000.0, 1100.0, 1150.0, 1110.0, 1050.0, 1080.0, 1060.0],
        final_balance=1060.0,
        initial_balance=1000.0,
    )


def test_calculates_metrics_for_winning_and_losing_trades():
    metrics = calculate_metrics(_mixed_result())

    assert isinstance(metrics, BacktestMetrics)
    assert metrics.net_profit == 60.0
    assert metrics.max_drawdown_amount == 100.0


def test_trade_counts_are_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.total_trades == 6
    assert metrics.winning_trades == 3
    assert metrics.losing_trades == 3


def test_win_rate_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.win_rate == pytest.approx(0.5)


def test_profit_loss_and_net_profit_are_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.gross_profit == 180.0
    assert metrics.gross_loss == 120.0
    assert metrics.net_profit == 60.0


def test_profit_factor_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.profit_factor == pytest.approx(1.5)


def test_payoff_ratio_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.payoff_ratio == pytest.approx(1.5)


def test_expectancy_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.expectancy == pytest.approx(10.0)


def test_max_drawdown_amount_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.max_drawdown_amount == pytest.approx(100.0)


def test_max_drawdown_percent_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.max_drawdown_percent == pytest.approx((100.0 / 1150.0) * 100)


def test_recovery_factor_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.recovery_factor == pytest.approx(0.6)


def test_max_consecutive_wins_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.max_consecutive_wins == 2


def test_max_consecutive_losses_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.max_consecutive_losses == 2


def test_no_trades_returns_coherent_metrics():
    result = BacktestResult(
        trades=[],
        equity_curve=[1000.0, 1000.0],
        final_balance=1000.0,
        initial_balance=1000.0,
    )

    metrics = calculate_metrics(result)

    assert metrics.total_trades == 0
    assert metrics.winning_trades == 0
    assert metrics.losing_trades == 0
    assert metrics.win_rate == 0.0
    assert metrics.gross_profit == 0.0
    assert metrics.gross_loss == 0.0
    assert metrics.net_profit == 0.0
    assert metrics.profit_factor is None
    assert metrics.payoff_ratio is None
    assert metrics.expectancy == 0.0
    assert metrics.max_consecutive_wins == 0
    assert metrics.max_consecutive_losses == 0


def test_zero_gross_loss_returns_none_profit_factor():
    result = BacktestResult(
        trades=[_trade(100.0, 0), _trade(50.0, 1)],
        equity_curve=[1000.0, 1100.0, 1150.0],
        final_balance=1150.0,
        initial_balance=1000.0,
    )

    metrics = calculate_metrics(result)

    assert metrics.profit_factor is None


def test_missing_wins_or_losses_returns_none_payoff_ratio():
    winners_only = BacktestResult(
        trades=[_trade(100.0, 0)],
        equity_curve=[1000.0, 1100.0],
        final_balance=1100.0,
        initial_balance=1000.0,
    )
    losers_only = BacktestResult(
        trades=[_trade(-100.0, 0)],
        equity_curve=[1000.0, 900.0],
        final_balance=900.0,
        initial_balance=1000.0,
    )

    assert calculate_metrics(winners_only).payoff_ratio is None
    assert calculate_metrics(losers_only).payoff_ratio is None


def test_empty_equity_curve_raises_value_error():
    result = BacktestResult(
        trades=[],
        equity_curve=[],
        final_balance=1000.0,
        initial_balance=1000.0,
    )

    with pytest.raises(ValueError, match="equity_curve"):
        calculate_metrics(result)


def test_non_positive_initial_balance_raises_value_error():
    result = BacktestResult(
        trades=[],
        equity_curve=[0.0],
        final_balance=0.0,
        initial_balance=0.0,
    )

    with pytest.raises(ValueError, match="initial_balance"):
        calculate_metrics(result)


def test_return_percent_is_correct():
    metrics = calculate_metrics(_mixed_result())

    assert metrics.return_percent == pytest.approx(6.0)
