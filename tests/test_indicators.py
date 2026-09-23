import pytest

from src.indicators import (
    average_true_range,
    exponential_moving_average,
    on_balance_volume,
    relative_strength_index,
    simple_moving_average,
)


def test_simple_moving_average_returns_none_until_window_is_complete():
    result = simple_moving_average([1, 2, 3, 4, 5], period=3)
    assert result == [None, None, 2.0, 3.0, 4.0]


def test_exponential_moving_average_uses_sma_seed():
    result = exponential_moving_average([1, 2, 3, 4, 5], period=3)
    assert result == [None, None, 2.0, 3.0, 4.0]


def test_relative_strength_index_handles_up_only_series():
    result = relative_strength_index([1, 2, 3, 4, 5, 6], period=3)
    assert result == [None, None, None, 100.0, 100.0, 100.0]


def test_average_true_range_uses_true_range_and_wilder_smoothing():
    result = average_true_range(
        highs=[10, 12, 13],
        lows=[8, 9, 11],
        closes=[9, 11, 12],
        period=2,
    )
    assert result == [None, 2.5, 2.25]


def test_on_balance_volume_accumulates_directional_volume():
    result = on_balance_volume(
        closes=[10, 11, 10, 10, 12],
        volumes=[100, 50, 25, 10, 40],
    )
    assert result == [0.0, 50.0, 25.0, 25.0, 65.0]


def test_indicator_period_must_be_positive():
    with pytest.raises(ValueError):
        simple_moving_average([1, 2, 3], period=0)
