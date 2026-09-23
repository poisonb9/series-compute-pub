from datetime import datetime

import pytest

from src.data_loader import OHLCVBar, load_ohlcv_csv, validate_ohlcv


def _write_csv(tmp_path, content: str):
    csv_path = tmp_path / "ohlcv.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def _bar(
    timestamp: datetime,
    open_price: float = 10.0,
    high: float = 12.0,
    low: float = 9.0,
    close: float = 11.0,
    volume: float = 100.0,
) -> OHLCVBar:
    return OHLCVBar(timestamp, open_price, high, low, close, volume)


def test_loads_valid_csv_with_iso_timestamp(tmp_path):
    path = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01T00:01:00,11,13,10,12,150\n"
        "2026-01-01T00:00:00,10,12,9,11,100\n",
    )

    bars = load_ohlcv_csv(path)

    assert len(bars) == 2
    assert bars[0] == _bar(datetime(2026, 1, 1))
    assert bars[1].timestamp == datetime(2026, 1, 1, 0, 1)


def test_loads_valid_csv_with_epoch_milliseconds(tmp_path):
    path = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "1767225600000,10,12,9,11,100\n",
    )

    bars = load_ohlcv_csv(path)

    assert bars[0].timestamp == datetime(2026, 1, 1)


def test_normalizes_headers_with_spaces_and_uppercase(tmp_path):
    path = _write_csv(
        tmp_path,
        " Timestamp , OPEN , High , LOW , Close , Volume \n"
        "2026-01-01T00:00:00Z,10,12,9,11,100\n",
    )

    bars = load_ohlcv_csv(path)

    assert bars == [_bar(datetime(2026, 1, 1))]


def test_raises_value_error_when_required_column_is_missing(tmp_path):
    path = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close\n"
        "2026-01-01T00:00:00,10,12,9,11\n",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_ohlcv_csv(path)


def test_raises_value_error_for_invalid_number(tmp_path):
    path = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01T00:00:00,invalid,12,9,11,100\n",
    )

    with pytest.raises(ValueError, match="invalid numeric value"):
        load_ohlcv_csv(path)


def test_raises_value_error_for_invalid_timestamp(tmp_path):
    path = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "not-a-timestamp,10,12,9,11,100\n",
    )

    with pytest.raises(ValueError, match="invalid timestamp"):
        load_ohlcv_csv(path)


def test_validate_ohlcv_approves_valid_list():
    bars = [
        _bar(datetime(2026, 1, 1)),
        _bar(datetime(2026, 1, 1, 0, 1), open_price=11, high=13, low=10, close=12),
    ]

    result = validate_ohlcv(bars)

    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []
    assert result.row_count == 2
    assert result.start_timestamp == datetime(2026, 1, 1)
    assert result.end_timestamp == datetime(2026, 1, 1, 0, 1)


def test_validate_ohlcv_rejects_empty_list():
    result = validate_ohlcv([])

    assert result.is_valid is False
    assert result.errors
    assert result.row_count == 0
    assert result.start_timestamp is None
    assert result.end_timestamp is None


def test_validate_ohlcv_detects_duplicate_timestamp():
    timestamp = datetime(2026, 1, 1)
    result = validate_ohlcv([_bar(timestamp), _bar(timestamp)])

    assert result.is_valid is False
    assert any("duplicate timestamp" in error for error in result.errors)


def test_validate_ohlcv_detects_out_of_order_timestamps():
    bars = [
        _bar(datetime(2026, 1, 1, 0, 1)),
        _bar(datetime(2026, 1, 1)),
    ]

    result = validate_ohlcv(bars)

    assert result.is_valid is False
    assert any("out of order" in error for error in result.errors)


def test_validate_ohlcv_detects_high_below_prices():
    result = validate_ohlcv(
        [_bar(datetime(2026, 1, 1), open_price=10, high=9, low=8, close=9.5)]
    )

    assert result.is_valid is False
    assert any("high must be" in error for error in result.errors)


def test_validate_ohlcv_detects_low_above_prices():
    result = validate_ohlcv(
        [_bar(datetime(2026, 1, 1), open_price=10, high=12, low=11, close=10.5)]
    )

    assert result.is_valid is False
    assert any("low must be" in error for error in result.errors)


def test_validate_ohlcv_detects_non_positive_price():
    result = validate_ohlcv(
        [_bar(datetime(2026, 1, 1), open_price=0, close=-1)]
    )

    assert result.is_valid is False
    assert any("greater than zero" in error for error in result.errors)


def test_validate_ohlcv_detects_negative_volume():
    result = validate_ohlcv([_bar(datetime(2026, 1, 1), volume=-1)])

    assert result.is_valid is False
    assert any("volume must be" in error for error in result.errors)
