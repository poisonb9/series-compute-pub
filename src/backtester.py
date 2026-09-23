from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.data_loader import OHLCVBar


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class Signal:
    signal_type: SignalType
    stop_loss: float | None = None
    take_profit: float | None = None
    reason: str = ""


@dataclass
class BacktestConfig:
    initial_balance: float = 1000.0
    position_size: float = 1.0
    fee_rate: float = 0.0004
    allow_live_orders: bool = False

    def __post_init__(self) -> None:
        if self.allow_live_orders:
            raise ValueError("Live orders are not allowed.")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be greater than zero.")
        if self.position_size <= 0:
            raise ValueError("position_size must be greater than zero.")
        if self.fee_rate < 0:
            raise ValueError("fee_rate must be greater than or equal to zero.")


@dataclass
class Trade:
    direction: TradeDirection
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[float]
    final_balance: float
    initial_balance: float


@dataclass
class _OpenPosition:
    direction: TradeDirection
    entry_timestamp: datetime
    entry_price: float
    quantity: float
    stop_loss: float | None
    take_profit: float | None
    entry_fee: float


class Backtester:
    def run(
        self,
        bars: list[OHLCVBar],
        strategy_fn: Callable[[list[OHLCVBar], int], Signal],
        config: BacktestConfig,
    ) -> BacktestResult:
        if not bars:
            raise ValueError("bars cannot be empty.")

        trades: list[Trade] = []
        equity_curve: list[float] = []
        balance = config.initial_balance
        position: _OpenPosition | None = None

        for index, bar in enumerate(bars):
            signal = strategy_fn(bars, index)
            if not isinstance(signal, Signal):
                raise TypeError("strategy_fn must return a Signal.")

            if position is None:
                position = self._open_position(bar, signal, config)
            else:
                exit_price, exit_reason, is_reversal = self._resolve_exit(position, bar, signal)
                if exit_price is not None and exit_reason is not None:
                    trade = self._close_position(
                        position,
                        bar.timestamp,
                        exit_price,
                        exit_reason,
                        config.fee_rate,
                    )
                    trades.append(trade)
                    balance += trade.net_pnl
                    position = None
                    # Design (b), CLOSE AND REVERSE. Chosen by the writer in pack 057
                    # and ratified by the project owner on 2026-07-27, verbatim: "fecha e reverte
                    # mesmo". After a reversal close, open the opposite position on the
                    # SAME bar, so a crossover system stays in the market as its signal
                    # intends. A stop-loss or take-profit exit must NOT be re-entered on
                    # the same bar -- that would defeat the protection that just fired.
                    # Cost note: a reversal pays two fees (exit + entry), by design.
                    #
                    # `is_reversal` is a STRUCTURAL flag, deliberately not derived from
                    # `exit_reason`. Pack 057 keyed this branch off the reason TEXT
                    # (`exit_reason == "reverse_signal_exit"`) while the resolver
                    # returned `signal.reason or "reverse_signal_exit"`. Every real
                    # strategy fills `reason` (`sma_death_cross`), so the `or` never hit
                    # its default, the comparison never matched, and the opposite leg
                    # NEVER OPENED: the engine closed without reversing, silently, for
                    # any strategy that names its signals. Audit 047 measured it.
                    # A data field must never double as a control discriminator.
                    if is_reversal:
                        position = self._open_position(bar, signal, config)

            equity_curve.append(self._equity_at_close(balance, position, bar.close))

        if position is not None:
            last_bar = bars[-1]
            trade = self._close_position(
                position,
                last_bar.timestamp,
                last_bar.close,
                "end_of_data",
                config.fee_rate,
            )
            trades.append(trade)
            balance += trade.net_pnl
            equity_curve[-1] = balance

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            final_balance=balance,
            initial_balance=config.initial_balance,
        )

    @staticmethod
    def _open_position(
        bar: OHLCVBar,
        signal: Signal,
        config: BacktestConfig,
    ) -> _OpenPosition | None:
        if signal.signal_type is SignalType.BUY:
            direction = TradeDirection.LONG
        elif signal.signal_type is SignalType.SELL:
            direction = TradeDirection.SHORT
        else:
            return None

        entry_fee = abs(bar.close * config.position_size) * config.fee_rate
        return _OpenPosition(
            direction=direction,
            entry_timestamp=bar.timestamp,
            entry_price=bar.close,
            quantity=config.position_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_fee=entry_fee,
        )

    @staticmethod
    def _resolve_exit(
        position: _OpenPosition,
        bar: OHLCVBar,
        signal: Signal,
    ) -> tuple[float | None, str | None, bool]:
        """(preço de saída, motivo, is_reversal).

        `is_reversal` diz se a saída foi causada por um sinal OPOSTO e, portanto, se
        a perna contrária deve abrir na mesma barra. É devolvido como terceiro valor
        de propósito: quem decide isso é a ESTRUTURA do caso, nunca o texto de
        `motivo`, que é dado de relatório e pertence à estratégia.
        """
        if position.direction is TradeDirection.LONG:
            if position.stop_loss is not None and bar.low <= position.stop_loss:
                return position.stop_loss, "stop_loss", False
            if position.take_profit is not None and bar.high >= position.take_profit:
                return position.take_profit, "take_profit", False
        else:
            if position.stop_loss is not None and bar.high >= position.stop_loss:
                return position.stop_loss, "stop_loss", False
            if position.take_profit is not None and bar.low <= position.take_profit:
                return position.take_profit, "take_profit", False

        if signal.signal_type is SignalType.EXIT:
            return bar.close, signal.reason or "signal_exit", False

        # Reversal support (Phase 057, 2026-07-26). Before this, an entry signal
        # OPPOSITE to the open position was silently dropped: `run` only opens when
        # flat, and this resolver only closed on EXIT/stop/take-profit. A strategy
        # that signals reversal via SELL-while-long (the SMA crossover with
        # allow_short=True does exactly that, and never emits EXIT) therefore opened
        # one position and never closed it until end_of_data. Pack 056 measured the
        # damage on real data: 11,114 crossover signals across 9 series produced
        # 9 trades, and zero EXIT signals ever existed to close them.
        # Precedence is deliberate: stop-loss and take-profit above still win, so a
        # protective exit is never overridden by a reversal signal on the same bar.
        if signal.signal_type is SignalType.SELL and position.direction is TradeDirection.LONG:
            return bar.close, signal.reason or "reverse_signal_exit", True
        if signal.signal_type is SignalType.BUY and position.direction is TradeDirection.SHORT:
            return bar.close, signal.reason or "reverse_signal_exit", True

        return None, None, False

    @staticmethod
    def _close_position(
        position: _OpenPosition,
        exit_timestamp: datetime,
        exit_price: float,
        exit_reason: str,
        fee_rate: float,
    ) -> Trade:
        if position.direction is TradeDirection.LONG:
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity

        exit_fee = abs(exit_price * position.quantity) * fee_rate
        total_fees = position.entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees

        return Trade(
            direction=position.direction,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=exit_timestamp,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            gross_pnl=gross_pnl,
            fees=total_fees,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
        )

    @staticmethod
    def _equity_at_close(
        balance: float,
        position: _OpenPosition | None,
        close_price: float,
    ) -> float:
        if position is None:
            return balance

        if position.direction is TradeDirection.LONG:
            unrealized_pnl = (close_price - position.entry_price) * position.quantity
        else:
            unrealized_pnl = (position.entry_price - close_price) * position.quantity

        return balance + unrealized_pnl - position.entry_fee
