"""Risk calculations for simulated trades only."""

from __future__ import annotations

from numbers import Real


def _positive_number(value: Real, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    numeric_value = float(value)
    if numeric_value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return numeric_value


def calculate_position_size(
    account_balance: Real,
    risk_percent: Real,
    stop_distance: Real,
    entry_price: Real | None = None,
) -> dict[str, float]:
    """Calculate simulated position size from risk and stop distance.

    stop_distance must be the monetary distance between entry and stop per
    unit of the asset. This function never sends orders to any venue.
    """

    balance = _positive_number(account_balance, "account_balance")
    risk = _positive_number(risk_percent, "risk_percent")
    stop = _positive_number(stop_distance, "stop_distance")

    if risk > 100:
        raise ValueError("risk_percent cannot exceed 100")

    risk_amount = balance * (risk / 100)
    position_size = risk_amount / stop

    result = {
        "account_balance": balance,
        "risk_percent": risk,
        "risk_amount": risk_amount,
        "stop_distance": stop,
        "position_size": position_size,
    }

    if entry_price is not None:
        entry = _positive_number(entry_price, "entry_price")
        result["entry_price"] = entry
        result["notional_value"] = position_size * entry

    return result
