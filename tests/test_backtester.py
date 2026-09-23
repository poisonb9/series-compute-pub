from datetime import datetime, timedelta

import pytest

from src.backtester import (
    BacktestConfig,
    Backtester,
    Signal,
    SignalType,
    TradeDirection,
)
from src.data_loader import OHLCVBar


def _bar(
    minute: int,
    open_price: float = 10.0,
    high: float = 11.0,
    low: float = 9.0,
    close: float = 10.0,
    volume: float = 100.0,
) -> OHLCVBar:
    return OHLCVBar(
        timestamp=datetime(2026, 1, 1) + timedelta(minutes=minute),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _strategy(signals: list[Signal]):
    def strategy_fn(bars: list[OHLCVBar], index: int) -> Signal:
        return signals[index]

    return strategy_fn


def test_backtest_config_rejects_live_orders():
    with pytest.raises(ValueError, match="Live orders"):
        BacktestConfig(allow_live_orders=True)


def test_backtest_config_rejects_non_positive_initial_balance():
    with pytest.raises(ValueError, match="initial_balance"):
        BacktestConfig(initial_balance=0)


def test_backtest_config_rejects_non_positive_position_size():
    with pytest.raises(ValueError, match="position_size"):
        BacktestConfig(position_size=0)


def test_backtest_config_rejects_negative_fee_rate():
    with pytest.raises(ValueError, match="fee_rate"):
        BacktestConfig(fee_rate=-0.01)


def test_backtester_rejects_empty_bars():
    with pytest.raises(ValueError, match="bars cannot be empty"):
        Backtester().run([], lambda bars, index: Signal(SignalType.HOLD), BacktestConfig())


def test_buy_then_exit_generates_long_trade():
    bars = [_bar(0, close=10), _bar(1, close=12, high=12, low=10)]
    signals = [Signal(SignalType.BUY), Signal(SignalType.EXIT, reason="strategy_exit")]

    result = Backtester().run(bars, _strategy(signals), BacktestConfig(fee_rate=0))

    trade = result.trades[0]
    assert trade.direction is TradeDirection.LONG
    assert trade.entry_price == 10
    assert trade.exit_price == 12
    assert trade.gross_pnl == 2
    assert trade.exit_reason == "strategy_exit"


def test_sell_then_exit_generates_short_trade():
    bars = [_bar(0, close=10), _bar(1, close=8, high=10, low=8)]
    signals = [Signal(SignalType.SELL), Signal(SignalType.EXIT)]

    result = Backtester().run(bars, _strategy(signals), BacktestConfig(fee_rate=0))

    trade = result.trades[0]
    assert trade.direction is TradeDirection.SHORT
    assert trade.entry_price == 10
    assert trade.exit_price == 8
    assert trade.gross_pnl == 2


def test_long_take_profit_closes_at_take_profit():
    bars = [_bar(0), _bar(1, high=12.5, low=9.5, close=11)]
    signals = [
        Signal(SignalType.BUY, stop_loss=8, take_profit=12),
        Signal(SignalType.HOLD),
    ]

    result = Backtester().run(bars, _strategy(signals), BacktestConfig(fee_rate=0))

    assert result.trades[0].exit_price == 12
    assert result.trades[0].exit_reason == "take_profit"


def test_long_stop_loss_closes_at_stop_loss():
    bars = [_bar(0), _bar(1, high=10.5, low=8.5, close=9)]
    signals = [
        Signal(SignalType.BUY, stop_loss=9, take_profit=12),
        Signal(SignalType.HOLD),
    ]

    result = Backtester().run(bars, _strategy(signals), BacktestConfig(fee_rate=0))

    assert result.trades[0].exit_price == 9
    assert result.trades[0].exit_reason == "stop_loss"


def test_short_take_profit_closes_at_take_profit():
    bars = [_bar(0), _bar(1, high=10.5, low=7.5, close=9)]
    signals = [
        Signal(SignalType.SELL, stop_loss=12, take_profit=8),
        Signal(SignalType.HOLD),
    ]

    result = Backtester().run(bars, _strategy(signals), BacktestConfig(fee_rate=0))

    assert result.trades[0].exit_price == 8
    assert result.trades[0].exit_reason == "take_profit"


def test_short_stop_loss_closes_at_stop_loss():
    bars = [_bar(0), _bar(1, high=12.5, low=9.5, close=11)]
    signals = [
        Signal(SignalType.SELL, stop_loss=12, take_profit=8),
        Signal(SignalType.HOLD),
    ]

    result = Backtester().run(bars, _strategy(signals), BacktestConfig(fee_rate=0))

    assert result.trades[0].exit_price == 12
    assert result.trades[0].exit_reason == "stop_loss"


def test_stop_has_priority_when_stop_and_take_are_hit_in_same_bar():
    bars = [_bar(0), _bar(1, high=12.5, low=8.5, close=10)]
    signals = [
        Signal(SignalType.BUY, stop_loss=9, take_profit=12),
        Signal(SignalType.HOLD),
    ]

    result = Backtester().run(bars, _strategy(signals), BacktestConfig(fee_rate=0))

    assert result.trades[0].exit_price == 9
    assert result.trades[0].exit_reason == "stop_loss"


def test_open_position_closes_at_end_of_data():
    bars = [_bar(0, close=10), _bar(1, close=11)]
    signals = [Signal(SignalType.BUY), Signal(SignalType.HOLD)]

    result = Backtester().run(bars, _strategy(signals), BacktestConfig(fee_rate=0))

    assert result.trades[0].exit_price == 11
    assert result.trades[0].exit_reason == "end_of_data"


def test_equity_curve_matches_bar_count():
    bars = [_bar(0), _bar(1), _bar(2)]
    signals = [
        Signal(SignalType.BUY),
        Signal(SignalType.HOLD),
        Signal(SignalType.EXIT),
    ]

    result = Backtester().run(bars, _strategy(signals), BacktestConfig())

    assert len(result.equity_curve) == len(bars)


def test_fees_are_deducted_from_result():
    bars = [_bar(0, close=10), _bar(1, close=12, high=12, low=10)]
    signals = [Signal(SignalType.BUY), Signal(SignalType.EXIT)]
    config = BacktestConfig(position_size=2, fee_rate=0.01)

    result = Backtester().run(bars, _strategy(signals), config)

    trade = result.trades[0]
    assert trade.gross_pnl == 4
    assert trade.fees == pytest.approx(0.44)
    assert trade.net_pnl == pytest.approx(3.56)


def test_final_balance_matches_initial_balance_plus_net_pnl():
    bars = [
        _bar(0, close=10),
        _bar(1, close=12, high=12, low=10),
        _bar(2, close=10),
        _bar(3, close=8, high=10, low=8),
    ]
    signals = [
        Signal(SignalType.BUY),
        Signal(SignalType.EXIT),
        Signal(SignalType.SELL),
        Signal(SignalType.EXIT),
    ]
    config = BacktestConfig(initial_balance=1000, fee_rate=0.001)

    result = Backtester().run(bars, _strategy(signals), config)

    expected = config.initial_balance + sum(trade.net_pnl for trade in result.trades)
    assert result.final_balance == pytest.approx(expected)
