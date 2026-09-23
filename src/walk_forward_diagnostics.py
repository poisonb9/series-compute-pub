from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WalkForwardDiagnosticItem:
    symbol: str
    timeframe: str
    research_decision: str | None
    normalized_score: float | None
    monte_carlo_status: str | None
    walk_forward_status: str | None
    action_bucket: str | None
    likely_failure_mode: str
    severity: str
    reason: str


@dataclass(frozen=True)
class WalkForwardDiagnosticsSummary:
    total_items: int
    failed_count: int
    warning_count: int
    mc_positive_wf_failed_count: int
    needs_more_data_count: int
    review_priority_count: int
    items: list[WalkForwardDiagnosticItem]
    warnings: list[str]
    errors: list[str]


def build_walk_forward_diagnostics(
    decision_matrix_csv: str | Path = Path("reports/backtests/sma_decision_matrix.csv"),
    multi_asset_review_csv: str | Path = Path("reports/backtests/multi_asset_research_review.csv"),
) -> WalkForwardDiagnosticsSummary:
    decision_rows = _read_required_csv_rows(decision_matrix_csv)
    warnings: list[str] = []
    errors: list[str] = []

    multi_asset_rows = _read_optional_csv_rows(multi_asset_review_csv)
    if not Path(multi_asset_review_csv).exists():
        warnings.append("Multi-asset review CSV was not found; diagnostics used the decision matrix only.")

    multi_asset_by_key = {
        _key(row): row for row in multi_asset_rows if _key(row) != ("", "")
    }
    items = [
        _build_item(row, multi_asset_by_key.get(_key(row), {}))
        for row in decision_rows
        if _key(row) != ("", "")
    ]
    if not items:
        warnings.append("No Walk-Forward decision rows were available.")

    return WalkForwardDiagnosticsSummary(
        total_items=len(items),
        failed_count=sum(1 for item in items if item.walk_forward_status == "failed"),
        warning_count=sum(1 for item in items if item.walk_forward_status == "warning"),
        mc_positive_wf_failed_count=sum(
            1
            for item in items
            if item.monte_carlo_status == "robust_candidate"
            and item.walk_forward_status == "failed"
        ),
        needs_more_data_count=sum(
            1 for item in items if item.research_decision == "needs_more_data"
        ),
        review_priority_count=sum(
            1
            for item in items
            if item.action_bucket == "wfa_bottleneck_review"
            or item.likely_failure_mode == "mc_positive_but_oos_unstable"
        ),
        items=items,
        warnings=warnings,
        errors=errors,
    )


def export_walk_forward_diagnostics_csv(
    summary: WalkForwardDiagnosticsSummary,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "timeframe",
        "research_decision",
        "normalized_score",
        "monte_carlo_status",
        "walk_forward_status",
        "action_bucket",
        "likely_failure_mode",
        "severity",
        "reason",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary.items:
            writer.writerow(
                {
                    "symbol": item.symbol,
                    "timeframe": item.timeframe,
                    "research_decision": item.research_decision or "",
                    "normalized_score": _optional_float(item.normalized_score),
                    "monte_carlo_status": item.monte_carlo_status or "",
                    "walk_forward_status": item.walk_forward_status or "",
                    "action_bucket": item.action_bucket or "",
                    "likely_failure_mode": item.likely_failure_mode,
                    "severity": item.severity,
                    "reason": item.reason,
                }
            )
    return output_path


def generate_walk_forward_diagnostics_markdown(
    summary: WalkForwardDiagnosticsSummary,
    *,
    title: str = "Walk-Forward Diagnostics",
) -> str:
    mc_positive_items = [
        item
        for item in summary.items
        if item.likely_failure_mode == "mc_positive_but_oos_unstable"
    ]
    lines = [
        f"# {title}",
        "",
        "> Research/backtest-only diagnostics. This is not live trading, not an order, and not investment advice.",
        "",
        "This report does not prove edge, does not change approval criteria, and live trading remains prohibited.",
        "",
        "## Overview",
        "",
        f"- Total items: {summary.total_items}",
        f"- Walk-Forward failed: {summary.failed_count}",
        f"- Walk-Forward warning: {summary.warning_count}",
        f"- Monte Carlo robust_candidate with Walk-Forward failed: {summary.mc_positive_wf_failed_count}",
        f"- Needs more data decisions: {summary.needs_more_data_count}",
        f"- Review priority items: {summary.review_priority_count}",
        "",
        "## Diagnostics Table",
        "",
    ]
    if summary.items:
        lines.extend(
            [
                "| Symbol | Timeframe | Decision | Score | MC | Walk-Forward | Mode | Severity | Reason |",
                "| --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for item in summary.items:
            lines.append(
                "| "
                f"{item.symbol} | "
                f"{item.timeframe} | "
                f"{item.research_decision or 'n/a'} | "
                f"{_format_optional_float(item.normalized_score)} | "
                f"{item.monte_carlo_status or 'n/a'} | "
                f"{item.walk_forward_status or 'n/a'} | "
                f"{item.likely_failure_mode} | "
                f"{item.severity} | "
                f"{item.reason} |"
            )
    else:
        lines.append("No Walk-Forward decision rows were available.")

    lines.extend(["", "## MC Positive But WF Failed", ""])
    if mc_positive_items:
        lines.extend(
            [
                "| Symbol | Timeframe | Decision | Score | MC | Walk-Forward |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for item in mc_positive_items:
            lines.append(
                "| "
                f"{item.symbol} | "
                f"{item.timeframe} | "
                f"{item.research_decision or 'n/a'} | "
                f"{_format_optional_float(item.normalized_score)} | "
                f"{item.monte_carlo_status or 'n/a'} | "
                f"{item.walk_forward_status or 'n/a'} |"
            )
    else:
        lines.append("No Monte Carlo robust_candidate item failed Walk-Forward in this input.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Monte Carlo robust_candidate without Walk-Forward confirmation is not enough.",
            "- Failed Walk-Forward points to out-of-sample instability under the current research settings.",
            "- Pipeline-only combinations validate infrastructure, not robust research interpretation.",
            "- This diagnostic preserves the poor results instead of improving scores or changing gates.",
            "",
            "## Next Safe Step",
            "",
            "- Review Walk-Forward configuration, available bars, OOS behavior and parameter stability.",
            "- Inspect priority cases before any future grid expansion.",
            "- Do not change the Walk-Forward engine in this diagnostic step.",
            "- Do not operate from this report.",
            "",
        ]
    )
    _append_messages(lines, "Warnings", summary.warnings)
    _append_messages(lines, "Errors", summary.errors)
    markdown = "\n".join(lines).rstrip() + "\n"
    _assert_no_forbidden_language(markdown)
    return markdown


def classify_wfa_failure_modes(
    summary: WalkForwardDiagnosticsSummary,
) -> dict[str, int]:
    counts = {
        "mc_positive_but_oos_unstable": 0,
        "broad_fragility": 0,
        "wfa_warning_needs_review": 0,
        "needs_more_data": 0,
        "pipeline_validation_only": 0,
        "ambiguous": 0,
    }
    for item in summary.items:
        if item.likely_failure_mode in counts:
            counts[item.likely_failure_mode] += 1
        else:
            counts["ambiguous"] += 1
        if item.research_decision == "needs_more_data":
            counts["needs_more_data"] += 1
        if (
            item.research_decision == "pipeline_only"
            or item.action_bucket == "pipeline_validation_only"
        ):
            counts["pipeline_validation_only"] += 1
    return counts


def generate_wfa_failure_classification_markdown(
    summary: WalkForwardDiagnosticsSummary,
    *,
    title: str = "WFA Failure Classification",
) -> str:
    counts = classify_wfa_failure_modes(summary)
    priority_items = [
        item
        for item in summary.items
        if item.likely_failure_mode == "mc_positive_but_oos_unstable"
    ]
    priority_label = _format_item_labels(priority_items)
    lines = [
        f"# {title}",
        "",
        "> Research/backtest-only classification. This is not live trading, not an order, and not investment advice.",
        "",
        "This classification does not prove edge, does not change approval criteria, and live trading remains prohibited.",
        "",
        "## Failure Mode Counts",
        "",
    ]
    for mode, count in counts.items():
        lines.append(f"- {mode}: {count}")

    lines.extend(
        [
            "",
            "## MC Positive But WF Failed",
            "",
            f"- Priority items: {priority_label}",
            "- Monte Carlo robust_candidate is not enough when Walk-Forward fails.",
            "- WFA failed suggests instability outside the training sample.",
            "- BTCUSDT 4h and SOLUSDT 1d should be reviewed before any grid expansion when present.",
            "- If WFA fails across multiple assets or timeframes, the simple SMA hypothesis should remain classified as fragile.",
            "- Do not alter criteria to rescue the result.",
            "",
        ]
    )
    markdown = "\n".join(lines).rstrip() + "\n"
    _assert_no_forbidden_language(markdown)
    return markdown


def generate_wfa_review_action_plan_markdown(
    summary: WalkForwardDiagnosticsSummary,
    *,
    title: str = "WFA Review Action Plan",
) -> str:
    priority_items = [
        item
        for item in summary.items
        if item.likely_failure_mode == "mc_positive_but_oos_unstable"
    ]
    priority_label = _format_item_labels(priority_items)
    lines = [
        f"# {title}",
        "",
        "> Research/backtest-only action plan. This is not live trading, not an order, and not investment advice.",
        "",
        "This plan does not prove edge, does not change approval criteria, and live trading remains prohibited.",
        "",
        "## Ordered Review Plan",
        "",
        "1. Review the current WFA parameters: number of windows, train size, test size and minimum bars.",
        "2. Verify whether 1d datasets have enough bars for robust WFA interpretation.",
        f"3. Inspect priority cases: {priority_label}.",
        "4. Preserve the current SMA grid as the baseline.",
        "5. Only after the WFA review, consider a small pre-registered grid expansion.",
        "6. Do not create a new strategy yet.",
        "7. Do not operate from these diagnostics.",
        "",
        "## Notes",
        "",
        "- This block is diagnostic only and does not modify Walk-Forward behavior.",
        "- BTCUSDT 4h and SOLUSDT 1d are the expected priority cases when they have Monte Carlo robust_candidate with failed Walk-Forward.",
        "- Expanding the grid before understanding the OOS failure would be premature.",
        "",
    ]
    markdown = "\n".join(lines).rstrip() + "\n"
    _assert_no_forbidden_language(markdown)
    return markdown


def _build_item(
    decision_row: dict[str, str],
    multi_asset_row: dict[str, str],
) -> WalkForwardDiagnosticItem:
    research_decision = _first_non_empty(
        decision_row.get("research_decision"),
        multi_asset_row.get("research_decision"),
    )
    normalized_score = _first_float(
        decision_row.get("normalized_score"),
        multi_asset_row.get("normalized_score"),
    )
    monte_carlo_status = _first_non_empty(
        decision_row.get("monte_carlo_status"),
        multi_asset_row.get("monte_carlo_status"),
    )
    walk_forward_status = _first_non_empty(
        decision_row.get("walk_forward_status"),
        multi_asset_row.get("walk_forward_status"),
    )
    action_bucket = _first_non_empty(decision_row.get("action_bucket"))
    likely_failure_mode, severity, reason, action_bucket = _classify_item(
        research_decision=research_decision,
        monte_carlo_status=monte_carlo_status,
        walk_forward_status=walk_forward_status,
        action_bucket=action_bucket,
    )
    return WalkForwardDiagnosticItem(
        symbol=decision_row.get("symbol", ""),
        timeframe=decision_row.get("timeframe", ""),
        research_decision=research_decision,
        normalized_score=normalized_score,
        monte_carlo_status=monte_carlo_status,
        walk_forward_status=walk_forward_status,
        action_bucket=action_bucket,
        likely_failure_mode=likely_failure_mode,
        severity=severity,
        reason=reason,
    )


def _classify_item(
    *,
    research_decision: str | None,
    monte_carlo_status: str | None,
    walk_forward_status: str | None,
    action_bucket: str | None,
) -> tuple[str, str, str, str | None]:
    reasons: list[str] = []
    mode = "ambiguous"
    severity = "low"
    bucket = action_bucket

    if monte_carlo_status == "robust_candidate" and walk_forward_status in {
        "failed",
        "warning",
    }:
        mode = "mc_positive_but_oos_unstable"
        severity = "high"
        bucket = "wfa_bottleneck_review"
        reasons.append(
            "Monte Carlo is robust_candidate, but Walk-Forward is unstable outside the training sample."
        )
    elif walk_forward_status == "failed" and monte_carlo_status == "fragile":
        mode = "broad_fragility"
        severity = "high"
        reasons.append("Walk-Forward failed and Monte Carlo is fragile.")
    elif walk_forward_status == "warning":
        mode = "wfa_warning_needs_review"
        severity = "medium"
        reasons.append("Walk-Forward returned warning and needs review.")
    else:
        reasons.append("Signals are mixed or incomplete and need conservative review.")

    if research_decision == "needs_more_data":
        reasons.append("Research decision is needs_more_data, so more history is required.")
    if research_decision == "pipeline_only":
        reasons.append("Research decision is pipeline_only, so this is pipeline validation only.")

    return mode, severity, " ".join(reasons), bucket


def _read_required_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return _read_csv_rows(csv_path)


def _read_optional_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    return _read_csv_rows(csv_path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def _key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("symbol", ""), row.get("timeframe", ""))


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _first_float(*values: str | None) -> float | None:
    for value in values:
        if value is None or not str(value).strip():
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _optional_float(value: float | None) -> float | str:
    if value is None:
        return ""
    return value


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_item_labels(items: list[WalkForwardDiagnosticItem]) -> str:
    if not items:
        return "none"
    return ", ".join(f"{item.symbol} {item.timeframe}" for item in items)


def _append_messages(lines: list[str], title: str, messages: list[str]) -> None:
    if not messages:
        return
    lines.extend([f"## {title}", ""])
    lines.extend(f"- {message}" for message in messages)
    lines.append("")


def _assert_no_forbidden_language(markdown: str) -> None:
    lower_markdown = markdown.lower()
    forbidden_phrases = (
        "guaranteed",
        "approved for live",
        "proven edge",
        "lucro garantido",
        "aprovado para operar",
        "edge comprovado",
        "estrat\u00e9gia vencedora",
    )
    for phrase in forbidden_phrases:
        if phrase in lower_markdown:
            raise ValueError(f"Markdown contains forbidden phrase: {phrase}")
