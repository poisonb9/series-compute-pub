from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.backtester import BacktestResult, Trade, TradeDirection
from src.metrics import BacktestMetrics, calculate_metrics
from src.monte_carlo import (
    MonteCarloConfig,
    MonteCarloStatus,
    calculate_percentile,
    export_monte_carlo_simulations_csv,
    generate_monte_carlo_markdown_summary,
    run_trade_bootstrap_monte_carlo,
)


def _trade(net_pnl: float, index: int) -> Trade:
    entry_timestamp = datetime(2026, 1, 1) + timedelta(minutes=index * 2)
    return Trade(
        direction=TradeDirection.LONG,
        entry_timestamp=entry_timestamp,
        exit_timestamp=entry_timestamp + timedelta(minutes=1),
        entry_price=100.0,
        exit_price=100.0 + net_pnl,
        quantity=1.0,
        gross_pnl=net_pnl,
        fees=0.0,
        net_pnl=net_pnl,
        exit_reason="test",
    )


def _result(net_pnls: list[float], *, initial_balance: float = 1000.0) -> BacktestResult:
    balance = initial_balance
    equity_curve = [balance]
    trades = []
    for index, net_pnl in enumerate(net_pnls):
        balance += net_pnl
        equity_curve.append(balance)
        trades.append(_trade(net_pnl, index))

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        final_balance=balance,
        initial_balance=initial_balance,
    )


def _manual_metrics(initial_balance: float = 1000.0) -> BacktestMetrics:
    return BacktestMetrics(
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        gross_profit=0.0,
        gross_loss=0.0,
        net_profit=0.0,
        profit_factor=None,
        payoff_ratio=None,
        expectancy=0.0,
        max_drawdown_amount=0.0,
        max_drawdown_percent=0.0,
        recovery_factor=None,
        max_consecutive_wins=0,
        max_consecutive_losses=0,
        final_balance=initial_balance,
        initial_balance=initial_balance,
        return_percent=0.0,
    )


def test_simulations_are_deterministic_with_same_seed():
    result = _result([10, -3, 5, 2, -1])
    metrics = calculate_metrics(result)
    config = MonteCarloConfig(iterations=20, seed=7, min_trades=1)

    first = run_trade_bootstrap_monte_carlo(result, metrics, config)
    second = run_trade_bootstrap_monte_carlo(result, metrics, config)

    assert first.simulations == second.simulations


def test_different_seeds_can_generate_different_distributions():
    result = _result([10, -8, 6, -3, 5])
    metrics = calculate_metrics(result)

    first = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=20, seed=1, min_trades=1),
    )
    second = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=20, seed=2, min_trades=1),
    )

    assert first.simulations != second.simulations


def test_returns_not_enough_trades_when_below_minimum():
    result = _result([10, -2])
    metrics = calculate_metrics(result)

    summary = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=10, seed=1, min_trades=5),
    )

    assert summary.status is MonteCarloStatus.NOT_ENOUGH_TRADES
    assert summary.simulations
    assert summary.warnings


def test_returns_invalid_when_initial_balance_is_non_positive():
    result = _result([], initial_balance=0.0)
    metrics = _manual_metrics(initial_balance=0.0)

    summary = run_trade_bootstrap_monte_carlo(result, metrics)

    assert summary.status is MonteCarloStatus.INVALID
    assert summary.errors
    assert not summary.is_usable


def test_probability_of_profit_is_correct_when_all_simulations_profit():
    result = _result([1, 2, 3])
    metrics = calculate_metrics(result)

    summary = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=10, seed=1, min_trades=1),
    )

    assert summary.probability_of_profit == 1.0


def test_calculates_max_drawdown_in_loss_sequence():
    result = _result([10, -5, -10, 2])
    metrics = calculate_metrics(result)

    summary = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=1, seed=2, min_trades=1),
    )

    assert summary.simulations[0].max_drawdown_amount > 0
    assert summary.simulations[0].max_drawdown_percent > 0


def test_percentiles_are_calculated_with_linear_interpolation():
    values = [0, 10, 20, 30]

    assert calculate_percentile(values, 0.05) == pytest.approx(1.5)
    assert calculate_percentile(values, 0.50) == pytest.approx(15.0)
    assert calculate_percentile(values, 0.95) == pytest.approx(28.5)


def test_export_monte_carlo_simulations_csv_creates_file(tmp_path):
    result = _result([1, -1, 2])
    metrics = calculate_metrics(result)
    summary = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=5, seed=1, min_trades=1),
    )
    output_path = tmp_path / "monte_carlo" / "simulations.csv"

    returned_path = export_monte_carlo_simulations_csv(summary, output_path)

    assert returned_path == output_path
    assert output_path.exists()


def test_export_monte_carlo_simulations_csv_contains_expected_header(tmp_path):
    result = _result([1, -1, 2])
    metrics = calculate_metrics(result)
    summary = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=5, seed=1, min_trades=1),
    )
    output_path = tmp_path / "simulations.csv"

    export_monte_carlo_simulations_csv(summary, output_path)

    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "iteration,final_balance,net_profit,return_percent,"
        "max_drawdown_amount,max_drawdown_percent,trades_count"
    )


def test_markdown_contains_research_backtest_only_warning():
    result = _result([1, -1, 2])
    metrics = calculate_metrics(result)
    summary = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=5, seed=1, min_trades=1),
    )

    markdown = generate_monte_carlo_markdown_summary(summary)

    assert "Research/backtest-only" in markdown


def test_markdown_states_monte_carlo_does_not_prove_edge():
    result = _result([1, -1, 2])
    metrics = calculate_metrics(result)
    summary = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=5, seed=1, min_trades=1),
    )

    markdown = generate_monte_carlo_markdown_summary(summary)

    assert "does not prove edge" in markdown
    assert "does not authorize live trading" in markdown


def test_no_trades_does_not_break_system():
    result = _result([])
    metrics = calculate_metrics(result)

    summary = run_trade_bootstrap_monte_carlo(result, metrics)

    assert summary.status is MonteCarloStatus.NOT_ENOUGH_TRADES
    assert summary.simulations == []
    assert summary.warnings


def test_invalid_monte_carlo_config_raises_value_error():
    with pytest.raises(ValueError):
        MonteCarloConfig(iterations=0)
    with pytest.raises(ValueError):
        MonteCarloConfig(seed=True)
    with pytest.raises(ValueError):
        MonteCarloConfig(min_trades=0)
    with pytest.raises(ValueError):
        MonteCarloConfig(confidence_level=1.0)


def test_monte_carlo_module_does_not_use_api_broker_or_live_language(tmp_path):
    result = _result([1, -1, 2])
    metrics = calculate_metrics(result)
    summary = run_trade_bootstrap_monte_carlo(
        result,
        metrics,
        MonteCarloConfig(iterations=5, seed=1, min_trades=1),
    )
    output_path = tmp_path / "simulations.csv"

    export_monte_carlo_simulations_csv(summary, output_path)

    csv_content = output_path.read_text(encoding="utf-8").lower()
    assert "api" not in csv_content
    assert "broker" not in csv_content
    assert "live" not in csv_content
