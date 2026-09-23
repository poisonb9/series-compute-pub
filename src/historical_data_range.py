from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.historical_data import infer_symbol_timeframe_from_filename, load_historical_csv


class DataRangeStatus(Enum):
    MATCHES_FILENAME = "matches_filename"
    OUTSIDE_DECLARED_RANGE = "outside_declared_range"
    CANNOT_INFER_RANGE = "cannot_infer_range"
    INVALID_DATASET = "invalid_dataset"


@dataclass(frozen=True)
class DataRangeAudit:
    path: Path
    symbol: str
    timeframe: str
    declared_start_month: str | None
    declared_end_month: str | None
    actual_start_timestamp: str | None
    actual_end_timestamp: str | None
    bars_count: int
    status: DataRangeStatus
    warnings: list[str]
    errors: list[str]


def infer_declared_month_range_from_filename(
    path: str | Path,
) -> tuple[str | None, str | None]:
    markers = _month_markers(Path(path).stem)
    if not markers:
        return None, None
    if len(markers) == 1:
        return markers[0], markers[0]
    return markers[0], markers[-1]


def audit_canonical_data_range(path: str | Path) -> DataRangeAudit:
    csv_path = Path(path)
    symbol, timeframe = infer_symbol_timeframe_from_filename(csv_path)
    declared_start, declared_end = infer_declared_month_range_from_filename(csv_path)
    warnings: list[str] = []
    errors: list[str] = []

    load_result = load_historical_csv(csv_path, symbol=symbol, timeframe=timeframe)
    metadata = load_result.metadata
    warnings.extend(metadata.warnings)
    errors.extend(metadata.errors)

    if metadata.errors or not load_result.bars:
        return DataRangeAudit(
            path=csv_path,
            symbol=metadata.symbol,
            timeframe=metadata.timeframe,
            declared_start_month=declared_start,
            declared_end_month=declared_end,
            actual_start_timestamp=metadata.start_timestamp,
            actual_end_timestamp=metadata.end_timestamp,
            bars_count=metadata.bars_count,
            status=DataRangeStatus.INVALID_DATASET,
            warnings=warnings,
            errors=errors or ["Dataset has no valid bars."],
        )

    if declared_start is None or declared_end is None:
        warnings.append("Could not infer declared month range from filename.")
        return DataRangeAudit(
            path=csv_path,
            symbol=metadata.symbol,
            timeframe=metadata.timeframe,
            declared_start_month=declared_start,
            declared_end_month=declared_end,
            actual_start_timestamp=metadata.start_timestamp,
            actual_end_timestamp=metadata.end_timestamp,
            bars_count=metadata.bars_count,
            status=DataRangeStatus.CANNOT_INFER_RANGE,
            warnings=warnings,
            errors=errors,
        )

    actual_start = min(bar.timestamp for bar in load_result.bars)
    actual_end = max(bar.timestamp for bar in load_result.bars)
    allowed_start = _month_start(declared_start)
    allowed_end_exclusive = _next_month(_month_start(declared_end))

    if actual_start < allowed_start or actual_end >= allowed_end_exclusive:
        warnings.append("Actual data extends outside declared filename range.")
        status = DataRangeStatus.OUTSIDE_DECLARED_RANGE
    else:
        status = DataRangeStatus.MATCHES_FILENAME

    return DataRangeAudit(
        path=csv_path,
        symbol=metadata.symbol,
        timeframe=metadata.timeframe,
        declared_start_month=declared_start,
        declared_end_month=declared_end,
        actual_start_timestamp=metadata.start_timestamp,
        actual_end_timestamp=metadata.end_timestamp,
        bars_count=metadata.bars_count,
        status=status,
        warnings=warnings,
        errors=errors,
    )


def export_data_range_audits_csv(
    audits: list[DataRangeAudit],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "symbol",
        "timeframe",
        "declared_start_month",
        "declared_end_month",
        "actual_start_timestamp",
        "actual_end_timestamp",
        "bars_count",
        "status",
        "warnings_count",
        "errors_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for audit in audits:
            writer.writerow(
                {
                    "path": str(audit.path),
                    "symbol": audit.symbol,
                    "timeframe": audit.timeframe,
                    "declared_start_month": audit.declared_start_month or "",
                    "declared_end_month": audit.declared_end_month or "",
                    "actual_start_timestamp": audit.actual_start_timestamp or "",
                    "actual_end_timestamp": audit.actual_end_timestamp or "",
                    "bars_count": audit.bars_count,
                    "status": audit.status.value,
                    "warnings_count": len(audit.warnings),
                    "errors_count": len(audit.errors),
                }
            )
    return output_path


def generate_data_range_audit_markdown(
    audits: list[DataRangeAudit],
    *,
    title: str = "Canonical Data Range Audit",
) -> str:
    outside = [
        audit
        for audit in audits
        if audit.status is DataRangeStatus.OUTSIDE_DECLARED_RANGE
    ]
    lines = [
        f"# {title}",
        "",
        "> Research/backtest-only data audit. This is not live trading, not an order, and not investment advice.",
        "",
        "This audits local CSV data ranges only. It does not prove edge, and live trading remains prohibited.",
        "",
        f"- Total CSVs audited: {len(audits)}",
        f"- Outside declared range: {len(outside)}",
        "",
        "## Audit Results",
        "",
    ]

    if audits:
        lines.extend(
            [
                "| Path | Symbol | Timeframe | Declared Start | Declared End | Actual Start | Actual End | Bars | Status | Warnings | Errors |",
                "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for audit in audits:
            lines.append(
                "| "
                f"{audit.path} | "
                f"{audit.symbol} | "
                f"{audit.timeframe} | "
                f"{audit.declared_start_month or ''} | "
                f"{audit.declared_end_month or ''} | "
                f"{audit.actual_start_timestamp or ''} | "
                f"{audit.actual_end_timestamp or ''} | "
                f"{audit.bars_count} | "
                f"{audit.status.value} | "
                f"{len(audit.warnings)} | "
                f"{len(audit.errors)} |"
            )
    else:
        lines.append("No canonical CSV files were found.")

    lines.extend(["", "## Range Issues", ""])
    if outside:
        lines.append(
            "The following files are outside declared filename range and should be rebuilt before quantitative interpretation:"
        )
        lines.extend(f"- {audit.path}" for audit in outside)
    else:
        lines.append("No files were detected outside their declared filename range.")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Rebuild contaminated canonical CSVs from filtered raw files.",
            "- Run this audit before campaign, benchmark, or multi-asset analysis.",
            "- Do not delete or modify source data automatically from this report.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _month_markers(stem: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for match in re.finditer(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(?!\d)", stem):
        matches.append((match.start(), f"{match.group(1)}-{match.group(2)}"))
    for match in re.finditer(r"(?<!\d)(20\d{2})[_-](0[1-9]|1[0-2])(?!\d)", stem):
        matches.append((match.start(), f"{match.group(1)}-{match.group(2)}"))

    seen: set[str] = set()
    ordered: list[str] = []
    for _position, marker in sorted(matches, key=lambda item: item[0]):
        if marker not in seen:
            ordered.append(marker)
            seen.add(marker)
    return ordered


def _month_start(month: str) -> datetime:
    year, month_number = month.split("-")
    return datetime(int(year), int(month_number), 1)


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1)
    return datetime(value.year, value.month + 1, 1)
