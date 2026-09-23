from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
_EPOCH_MILLISECONDS_THRESHOLD = 100_000_000_000
# 1e14: acima disto o valor so' faz sentido como microssegundos. O piso fica bem
# acima de qualquer instante plausivel em milissegundos (1e14 ms seria o ano ~5138).
_EPOCH_MICROSECONDS_THRESHOLD = 100_000_000_000_000
_MIN_PLAUSIBLE_YEAR = 1970
_MAX_PLAUSIBLE_YEAR = 2200


@dataclass
class OHLCVBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    row_count: int
    start_timestamp: datetime | None
    end_timestamp: datetime | None


def parse_timestamp(value: str) -> datetime:
    """Parse supported timestamps and normalize them to UTC-naive datetimes."""
    if not isinstance(value, str):
        raise ValueError("Timestamp must be a string.")

    stripped = value.strip()
    if not stripped:
        raise ValueError("Timestamp cannot be empty.")

    try:
        epoch_value = float(stripped)
    except ValueError:
        normalized = stripped[:-1] + "+00:00" if stripped.endswith(("Z", "z")) else stripped
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp: {value!r}.") from exc

        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    if not math.isfinite(epoch_value):
        raise ValueError(f"Invalid timestamp: {value!r}.")

    # UNIDADE DETECTADA POR MAGNITUDE, nunca presumida.
    #
    # Measured on this project's own data on 2026-08-05: the epoch unit depends on the
    # FOLDER and the acquisition date, not on the period of the data.
    #
    #     data/external/binance_klines_raw  13 digits (ms) through 2024-12,
    #                                       16 digits (us) from 2025-01 onward
    #     data/fresh_frozen                 13 digits (ms) for that SAME 2025 period
    #
    # Before this repair the ladder had two rungs. A 16-digit microsecond value was
    # divided ONCE by 1000, landed around the year 56000, and `fromtimestamp` raised --
    # which this function turned into "Invalid timestamp". That FAILED CLOSED, which is
    # the right direction, but it meant the production loader could not read data that
    # is perfectly intact, and an operator meeting the error concludes the DATA is
    # corrupt when the READER is incomplete.
    if abs(epoch_value) >= _EPOCH_MICROSECONDS_THRESHOLD:
        epoch_value /= 1_000_000.0
    elif abs(epoch_value) >= _EPOCH_MILLISECONDS_THRESHOLD:
        epoch_value /= 1000.0

    try:
        parsed = datetime.fromtimestamp(epoch_value, tz=timezone.utc).replace(tzinfo=None)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp: {value!r}.") from exc

    # ⚠️ A JANELA DE PLAUSIBILIDADE nao e' redundante com a escada acima, e o motivo
    # importa: detectar por magnitude sem limite superior apenas empurra a data absurda
    # uma escala adiante. Um valor com 19 digitos (nanossegundos, que este projeto ainda
    # nao viu mas que a Binance ja emitiu noutros produtos) cairia no ramo de
    # microssegundos e produziria o ano ~33000 -- silenciosamente, porque
    # `fromtimestamp` aceita. Um carimbo fora da janela e' entrada corrompida, nao dado
    # a converter.
    if not (_MIN_PLAUSIBLE_YEAR <= parsed.year <= _MAX_PLAUSIBLE_YEAR):
        raise ValueError(
            f"Invalid timestamp: {value!r} resolves to year {parsed.year}, outside the "
            f"plausible window [{_MIN_PLAUSIBLE_YEAR}, {_MAX_PLAUSIBLE_YEAR}]."
        )
    return parsed


def load_ohlcv_csv(path: str | Path) -> list[OHLCVBar]:
    csv_path = Path(path)

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row.")

        normalized_headers = [header.strip().lower() for header in reader.fieldnames]
        if len(set(normalized_headers)) != len(normalized_headers):
            raise ValueError("CSV contains duplicate column names after normalization.")

        missing_columns = [
            column for column in _REQUIRED_COLUMNS if column not in normalized_headers
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"CSV is missing required columns: {missing}.")

        header_map = dict(zip(reader.fieldnames, normalized_headers))
        bars: list[OHLCVBar] = []

        for row_number, row in enumerate(reader, start=2):
            normalized_row = {
                normalized_name: row.get(original_name)
                for original_name, normalized_name in header_map.items()
            }

            try:
                timestamp = parse_timestamp(normalized_row["timestamp"])
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_number} has an invalid timestamp: "
                    f"{normalized_row['timestamp']!r}."
                ) from exc

            numeric_values: dict[str, float] = {}
            for column in _REQUIRED_COLUMNS[1:]:
                raw_value = normalized_row[column]
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Row {row_number} has an invalid numeric value "
                        f"for {column}: {raw_value!r}."
                    ) from exc

                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"Row {row_number} has an invalid numeric value "
                        f"for {column}: {raw_value!r}."
                    )
                numeric_values[column] = numeric_value

            bars.append(
                OHLCVBar(
                    timestamp=timestamp,
                    open=numeric_values["open"],
                    high=numeric_values["high"],
                    low=numeric_values["low"],
                    close=numeric_values["close"],
                    volume=numeric_values["volume"],
                )
            )

    bars.sort(key=lambda bar: bar.timestamp)
    return bars


def validate_ohlcv(bars: list[OHLCVBar]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    row_count = len(bars)

    if not bars:
        return ValidationResult(
            is_valid=False,
            errors=["OHLCV data is empty."],
            warnings=warnings,
            row_count=0,
            start_timestamp=None,
            end_timestamp=None,
        )

    timestamps = [bar.timestamp for bar in bars]
    start_timestamp = min(timestamps)
    end_timestamp = max(timestamps)

    seen_timestamps: set[datetime] = set()
    previous_timestamp: datetime | None = None

    for index, bar in enumerate(bars, start=1):
        if bar.timestamp in seen_timestamps:
            errors.append(
                f"Row {index} has a duplicate timestamp: {bar.timestamp.isoformat()}."
            )
        seen_timestamps.add(bar.timestamp)

        if previous_timestamp is not None and bar.timestamp < previous_timestamp:
            errors.append(
                f"Row {index} timestamp is out of order: "
                f"{bar.timestamp.isoformat()}."
            )
        previous_timestamp = bar.timestamp

        prices = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
        }
        finite_prices = True
        for name, price in prices.items():
            if not math.isfinite(price) or price <= 0:
                finite_prices = False
                errors.append(f"Row {index} {name} must be greater than zero.")

        if not math.isfinite(bar.volume) or bar.volume < 0:
            errors.append(f"Row {index} volume must be greater than or equal to zero.")

        if finite_prices:
            if bar.high < max(bar.open, bar.close, bar.low):
                errors.append(
                    f"Row {index} high must be greater than or equal to "
                    "open, close, and low."
                )
            if bar.low > min(bar.open, bar.close, bar.high):
                errors.append(
                    f"Row {index} low must be less than or equal to "
                    "open, close, and high."
                )

    return ValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        row_count=row_count,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
