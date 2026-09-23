from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.data_loader import OHLCVBar, parse_timestamp, validate_ohlcv


class HistoricalDataStatus(Enum):
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"


@dataclass(frozen=True)
class HistoricalDatasetMetadata:
    path: Path
    symbol: str
    timeframe: str
    rows_count: int
    bars_count: int
    start_timestamp: str | None
    end_timestamp: str | None
    status: HistoricalDataStatus
    warnings: list[str]
    errors: list[str]


@dataclass(frozen=True)
class HistoricalDataLoadResult:
    metadata: HistoricalDatasetMetadata
    bars: list[OHLCVBar]


_COLUMN_ALIASES = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "open_time": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}
_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
_TIMEFRAME_PATTERN = re.compile(r"^\d+(?:m|h|d|w)$", re.IGNORECASE)


def load_historical_csv(
    path: str | Path,
    *,
    symbol: str = "UNKNOWN",
    timeframe: str = "UNKNOWN",
) -> HistoricalDataLoadResult:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if symbol == "UNKNOWN" and timeframe == "UNKNOWN":
        inferred_symbol, inferred_timeframe = infer_symbol_timeframe_from_filename(csv_path)
        symbol = inferred_symbol
        timeframe = inferred_timeframe

    warnings: list[str] = []
    errors: list[str] = []
    bars: list[OHLCVBar] = []
    rows_count = 0

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            sample = csv_file.read(4096)
            csv_file.seek(0)
            reader = csv.DictReader(csv_file, delimiter=_detect_delimiter(sample))
            if reader.fieldnames is None:
                raise ValueError("CSV must include a header row.")

            header_map = _build_header_map(reader.fieldnames)
            missing_columns = [
                column for column in _REQUIRED_COLUMNS if column not in header_map.values()
            ]
            if missing_columns:
                raise ValueError(
                    "CSV is missing required columns: "
                    + ", ".join(missing_columns)
                    + "."
                )

            for row_number, row in enumerate(reader, start=2):
                if _is_empty_row(row):
                    continue
                rows_count += 1
                normalized_row = {
                    canonical_name: row.get(original_name, "")
                    for original_name, canonical_name in header_map.items()
                }
                bars.append(_row_to_bar(normalized_row, row_number))
    except ValueError as exc:
        errors.append(str(exc))
        return _build_result(
            path=csv_path,
            symbol=symbol,
            timeframe=timeframe,
            rows_count=rows_count,
            bars=[],
            warnings=warnings,
            errors=errors,
        )

    warnings.extend(_data_warnings(bars))
    validation = validate_ohlcv(bars)
    warnings.extend(validation.warnings)
    errors.extend(validation.errors)

    return _build_result(
        path=csv_path,
        symbol=symbol,
        timeframe=timeframe,
        rows_count=rows_count,
        bars=bars,
        warnings=warnings,
        errors=errors,
    )


def infer_symbol_timeframe_from_filename(path: str | Path) -> tuple[str, str]:
    stem = Path(path).stem
    tokens = [token for token in re.split(r"[_\-\s.]+", stem) if token]

    for index, token in enumerate(tokens):
        if _TIMEFRAME_PATTERN.match(token) and index > 0:
            return tokens[index - 1].upper(), token.lower()

    return "UNKNOWN", "UNKNOWN"


def summarize_historical_dataset(result: HistoricalDataLoadResult) -> str:
    metadata = result.metadata
    lines = [
        "# Historical Dataset Summary",
        "",
        "Offline historical data loaded from local CSV only. No API, broker, live trading, or order execution is used.",
        "",
        f"- Symbol: {metadata.symbol}",
        f"- Timeframe: {metadata.timeframe}",
        f"- Bars count: {metadata.bars_count}",
        f"- Start timestamp: {metadata.start_timestamp or 'n/a'}",
        f"- End timestamp: {metadata.end_timestamp or 'n/a'}",
        f"- Status: {metadata.status.value}",
        "",
        "## Warnings",
        "",
    ]

    if metadata.warnings:
        lines.extend(f"- {warning}" for warning in metadata.warnings)
    else:
        lines.append("No warnings.")

    lines.extend(["", "## Errors", ""])
    if metadata.errors:
        lines.extend(f"- {error}" for error in metadata.errors)
    else:
        lines.append("No errors.")

    return "\n".join(lines).rstrip() + "\n"


def _detect_delimiter(sample: str) -> str:
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    if first_line.count(";") > first_line.count(","):
        return ";"
    return ","


def _build_header_map(fieldnames: list[str]) -> dict[str, str]:
    header_map: dict[str, str] = {}
    normalized_seen: set[str] = set()

    for fieldname in fieldnames:
        normalized_name = fieldname.strip().lower()
        if normalized_name in normalized_seen:
            raise ValueError("CSV contains duplicate column names after normalization.")
        normalized_seen.add(normalized_name)
        canonical_name = _COLUMN_ALIASES.get(normalized_name)
        if canonical_name:
            header_map[fieldname] = canonical_name

    return header_map


def _is_empty_row(row: dict[str, str | None]) -> bool:
    return all((value is None or not str(value).strip()) for value in row.values())


def _row_to_bar(row: dict[str, str | None], row_number: int) -> OHLCVBar:
    try:
        timestamp = parse_timestamp(str(row["timestamp"]))
    except ValueError as exc:
        raise ValueError(f"Row {row_number} has an invalid timestamp.") from exc

    values: dict[str, float] = {}
    for column in _REQUIRED_COLUMNS[1:]:
        raw_value = row.get(column)
        try:
            value = float(str(raw_value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Row {row_number} has an invalid numeric value for {column}."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"Row {row_number} has an invalid numeric value for {column}."
            )
        values[column] = value

    return OHLCVBar(
        timestamp=timestamp,
        open=values["open"],
        high=values["high"],
        low=values["low"],
        close=values["close"],
        volume=values["volume"],
    )


def _data_warnings(bars: list[OHLCVBar]) -> list[str]:
    warnings: list[str] = []
    if len(bars) < 100:
        warnings.append("Dataset has fewer than 100 bars.")
    if any(bar.volume == 0 for bar in bars):
        warnings.append("Dataset contains zero volume bars.")
    if _has_out_of_order_timestamps(bars):
        warnings.append("Dataset contains timestamps out of order.")
    if _has_duplicate_timestamps(bars):
        warnings.append("Dataset contains duplicate timestamps.")
    if _has_suspicious_time_gap(bars):
        warnings.append("Dataset contains suspicious time gaps.")
    return warnings


def _has_out_of_order_timestamps(bars: list[OHLCVBar]) -> bool:
    return any(
        current.timestamp < previous.timestamp
        for previous, current in zip(bars, bars[1:])
    )


def _has_duplicate_timestamps(bars: list[OHLCVBar]) -> bool:
    timestamps = [bar.timestamp for bar in bars]
    return len(set(timestamps)) != len(timestamps)


def _has_suspicious_time_gap(bars: list[OHLCVBar]) -> bool:
    if len(bars) < 3 or _has_out_of_order_timestamps(bars):
        return False

    diffs = [
        (current.timestamp - previous.timestamp).total_seconds()
        for previous, current in zip(bars, bars[1:])
    ]
    positive_diffs = [diff for diff in diffs if diff > 0]
    if len(positive_diffs) < 2:
        return False

    expected = min(positive_diffs)
    return any(diff > expected * 3 for diff in positive_diffs)


def _build_result(
    *,
    path: Path,
    symbol: str,
    timeframe: str,
    rows_count: int,
    bars: list[OHLCVBar],
    warnings: list[str],
    errors: list[str],
) -> HistoricalDataLoadResult:
    status = _status_for(warnings, errors)
    start_timestamp, end_timestamp = _start_end_timestamps(bars)

    return HistoricalDataLoadResult(
        metadata=HistoricalDatasetMetadata(
            path=path,
            symbol=symbol,
            timeframe=timeframe,
            rows_count=rows_count,
            bars_count=len(bars),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            status=status,
            warnings=warnings,
            errors=errors,
        ),
        bars=bars,
    )


def _status_for(
    warnings: list[str],
    errors: list[str],
) -> HistoricalDataStatus:
    if errors:
        return HistoricalDataStatus.INVALID
    if warnings:
        return HistoricalDataStatus.VALID_WITH_WARNINGS
    return HistoricalDataStatus.VALID


def _start_end_timestamps(bars: list[OHLCVBar]) -> tuple[str | None, str | None]:
    if not bars:
        return None, None
    timestamps: list[datetime] = [bar.timestamp for bar in bars]
    return min(timestamps).isoformat(), max(timestamps).isoformat()
