#!/usr/bin/env python3
"""Export compact benchmark trend data for CI dashboards.

The exporter consumes existing benchmark result artifacts, groups comparable
profile/model/quantization/prompt-suite points over time, and writes JSON,
CSV, Markdown, and JUnit dashboard artifacts. It is host-side only and never
launches QEMU.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_result_index


@dataclass(frozen=True)
class TrendPoint:
    key: str
    source: str
    artifact_type: str
    status: str
    generated_at: str
    profile: str
    model: str
    quantization: str
    prompt_suite_sha256: str
    command_sha256: str
    launch_plan_sha256: str
    environment_sha256: str
    commit: str
    command_airgap_status: str
    telemetry_status: str
    median_tok_per_s: float | None
    wall_tok_per_s_median: float | None
    us_per_token_median: float | None
    wall_us_per_token_median: float | None
    ttft_us_p95: float | None
    host_child_tok_per_cpu_s_median: float | None
    host_child_peak_rss_bytes_max: int | None
    max_memory_bytes: int | None


@dataclass(frozen=True)
class TrendFinding:
    gate: str
    key: str
    metric: str
    value: float | int | None
    threshold: float | int | None
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def trend_key(summary: bench_result_index.ArtifactSummary) -> str:
    parts = (
        summary.profile or "-",
        summary.model or "-",
        summary.quantization or "-",
        summary.prompt_suite_sha256 or "no-suite",
    )
    return "/".join(parts)


def to_point(summary: bench_result_index.ArtifactSummary) -> TrendPoint:
    return TrendPoint(
        key=trend_key(summary),
        source=summary.source,
        artifact_type=summary.artifact_type,
        status=summary.status,
        generated_at=summary.generated_at,
        profile=summary.profile,
        model=summary.model,
        quantization=summary.quantization,
        prompt_suite_sha256=summary.prompt_suite_sha256,
        command_sha256=summary.command_sha256,
        launch_plan_sha256=summary.launch_plan_sha256,
        environment_sha256=summary.environment_sha256,
        commit=summary.commit,
        command_airgap_status=summary.command_airgap_status,
        telemetry_status=summary.telemetry_status,
        median_tok_per_s=summary.median_tok_per_s,
        wall_tok_per_s_median=summary.wall_tok_per_s_median,
        us_per_token_median=summary.us_per_token_median,
        wall_us_per_token_median=summary.wall_us_per_token_median,
        ttft_us_p95=summary.ttft_us_p95,
        host_child_tok_per_cpu_s_median=summary.host_child_tok_per_cpu_s_median,
        host_child_peak_rss_bytes_max=summary.host_child_peak_rss_bytes_max,
        max_memory_bytes=summary.max_memory_bytes,
    )


def group_points(points: list[TrendPoint], max_points_per_key: int | None) -> dict[str, list[TrendPoint]]:
    grouped: dict[str, list[TrendPoint]] = {}
    for point in sorted(points, key=lambda item: (item.key, item.generated_at, item.source)):
        grouped.setdefault(point.key, []).append(point)
    if max_points_per_key is not None:
        grouped = {key: values[-max_points_per_key:] for key, values in grouped.items()}
    return grouped


def delta_pct(previous: float | None, latest: float | None) -> float | None:
    if previous is None or latest is None or previous == 0:
        return None
    return ((latest - previous) / previous) * 100.0


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def min_value(values: list[float]) -> float | None:
    return min(values) if values else None


def max_value(values: list[float]) -> float | None:
    return max(values) if values else None


def cv_pct(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    return float(statistics.pstdev(values) / abs(mean) * 100.0)


def latest_rows(grouped: dict[str, list[TrendPoint]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, points in sorted(grouped.items()):
        latest = points[-1]
        previous = points[-2] if len(points) >= 2 else None
        rows.append(
            {
                "key": key,
                "points": len(points),
                "latest_generated_at": latest.generated_at,
                "latest_commit": latest.commit,
                "latest_status": latest.status,
                "latest_airgap_status": latest.command_airgap_status,
                "latest_telemetry_status": latest.telemetry_status,
                "latest_median_tok_per_s": latest.median_tok_per_s,
                "previous_median_tok_per_s": previous.median_tok_per_s if previous else None,
                "median_tok_per_s_delta_pct": delta_pct(
                    previous.median_tok_per_s if previous else None,
                    latest.median_tok_per_s,
                ),
                "latest_wall_tok_per_s_median": latest.wall_tok_per_s_median,
                "previous_wall_tok_per_s_median": previous.wall_tok_per_s_median if previous else None,
                "wall_tok_per_s_delta_pct": delta_pct(
                    previous.wall_tok_per_s_median if previous else None,
                    latest.wall_tok_per_s_median,
                ),
                "latest_max_memory_bytes": latest.max_memory_bytes,
                "previous_max_memory_bytes": previous.max_memory_bytes if previous else None,
                "max_memory_delta_pct": delta_pct(
                    float(previous.max_memory_bytes) if previous and previous.max_memory_bytes is not None else None,
                    float(latest.max_memory_bytes) if latest.max_memory_bytes is not None else None,
                ),
                "latest_source": latest.source,
                "previous_source": previous.source if previous else "",
            }
        )
    return rows


def drift_rows(grouped: dict[str, list[TrendPoint]], field: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, points in sorted(grouped.items()):
        by_hash: dict[str, list[TrendPoint]] = {}
        for point in points:
            value = str(getattr(point, field) or "")
            by_hash.setdefault(value or "missing", []).append(point)
        if len(by_hash) <= 1:
            continue
        rows.append(
            {
                "key": key,
                "field": field,
                "hashes": sorted(by_hash),
                "points": len(points),
                "commits": sorted({point.commit for point in points}),
                "sources": sorted(point.source for point in points),
            }
        )
    return rows


def count_by(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "-"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def window_rows(grouped: dict[str, list[TrendPoint]], window_points: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, points in sorted(grouped.items()):
        window = points[-window_points:]
        if not window:
            continue
        first = window[0]
        latest = window[-1]
        guest_values = [
            float(point.median_tok_per_s)
            for point in window
            if point.median_tok_per_s is not None
        ]
        wall_values = [
            float(point.wall_tok_per_s_median)
            for point in window
            if point.wall_tok_per_s_median is not None
        ]
        memory_values = [
            float(point.max_memory_bytes)
            for point in window
            if point.max_memory_bytes is not None
        ]
        rows.append(
            {
                "key": key,
                "window_points_requested": window_points,
                "window_points": len(window),
                "window_start_generated_at": first.generated_at,
                "window_end_generated_at": latest.generated_at,
                "window_start_commit": first.commit,
                "window_end_commit": latest.commit,
                "guest_tok_per_s_points": len(guest_values),
                "guest_tok_per_s_min": min_value(guest_values),
                "guest_tok_per_s_median": median(guest_values),
                "guest_tok_per_s_max": max_value(guest_values),
                "guest_tok_per_s_cv_pct": cv_pct(guest_values),
                "guest_tok_per_s_delta_pct": delta_pct(first.median_tok_per_s, latest.median_tok_per_s),
                "wall_tok_per_s_points": len(wall_values),
                "wall_tok_per_s_min": min_value(wall_values),
                "wall_tok_per_s_median": median(wall_values),
                "wall_tok_per_s_max": max_value(wall_values),
                "wall_tok_per_s_cv_pct": cv_pct(wall_values),
                "wall_tok_per_s_delta_pct": delta_pct(
                    first.wall_tok_per_s_median,
                    latest.wall_tok_per_s_median,
                ),
                "max_memory_points": len(memory_values),
                "max_memory_bytes_min": min_value(memory_values),
                "max_memory_bytes_median": median(memory_values),
                "max_memory_bytes_max": max_value(memory_values),
                "max_memory_cv_pct": cv_pct(memory_values),
                "max_memory_delta_pct": delta_pct(
                    float(first.max_memory_bytes) if first.max_memory_bytes is not None else None,
                    float(latest.max_memory_bytes) if latest.max_memory_bytes is not None else None,
                ),
                "window_start_source": first.source,
                "window_end_source": latest.source,
            }
        )
    return rows


def build_report(
    grouped: dict[str, list[TrendPoint]],
    *,
    window_points: int,
    fail_on_empty: bool,
    fail_on_airgap: bool,
    fail_on_telemetry: bool,
    fail_on_command_drift: bool,
    fail_on_launch_plan_drift: bool,
    fail_on_environment_drift: bool,
    min_points_per_key: int | None,
    min_latest_tok_per_s: float | None,
    min_latest_wall_tok_per_s: float | None,
    fail_on_tok_regression_pct: float | None,
    fail_on_wall_tok_regression_pct: float | None,
    fail_on_memory_growth_pct: float | None,
    fail_on_window_tok_regression_pct: float | None,
    fail_on_window_wall_tok_regression_pct: float | None,
    fail_on_window_memory_growth_pct: float | None,
    max_window_tok_cv_pct: float | None,
    max_window_wall_tok_cv_pct: float | None,
    max_window_memory_cv_pct: float | None,
) -> dict[str, object]:
    all_points = [point for points in grouped.values() for point in points]
    latest = latest_rows(grouped)
    windows = window_rows(grouped, window_points)
    command_drift = drift_rows(grouped, "command_sha256")
    launch_plan_drift = drift_rows(grouped, "launch_plan_sha256")
    environment_drift = drift_rows(grouped, "environment_sha256")
    findings: list[str] = []
    finding_rows: list[TrendFinding] = []

    def add_finding(
        gate: str,
        message: str,
        *,
        key: str = "",
        metric: str = "",
        value: float | int | None = None,
        threshold: float | int | None = None,
    ) -> None:
        findings.append(message)
        finding_rows.append(
            TrendFinding(
                gate=gate,
                key=key,
                metric=metric,
                value=value,
                threshold=threshold,
                message=message,
            )
        )

    if not all_points:
        add_finding("empty", "no supported benchmark artifacts found")
    if fail_on_airgap:
        airgap_failures = [point for point in all_points if point.command_airgap_status == "fail"]
        if airgap_failures:
            add_finding(
                "airgap",
                f"{len(airgap_failures)} trend point(s) violate QEMU air-gap policy",
                metric="command_airgap_status",
                value=len(airgap_failures),
                threshold=0,
            )
    if fail_on_telemetry:
        telemetry_failures = [point for point in all_points if point.telemetry_status == "fail"]
        if telemetry_failures:
            add_finding(
                "telemetry",
                f"{len(telemetry_failures)} trend point(s) are missing telemetry",
                metric="telemetry_status",
                value=len(telemetry_failures),
                threshold=0,
            )
    if fail_on_command_drift and command_drift:
        add_finding(
            "command_drift",
            f"{len(command_drift)} trend key(s) have command_sha256 drift",
            metric="command_sha256",
            value=len(command_drift),
            threshold=0,
        )
    if fail_on_launch_plan_drift and launch_plan_drift:
        add_finding(
            "launch_plan_drift",
            f"{len(launch_plan_drift)} trend key(s) have launch_plan_sha256 drift",
            metric="launch_plan_sha256",
            value=len(launch_plan_drift),
            threshold=0,
        )
    if fail_on_environment_drift and environment_drift:
        add_finding(
            "environment_drift",
            f"{len(environment_drift)} trend key(s) have environment_sha256 drift",
            metric="environment_sha256",
            value=len(environment_drift),
            threshold=0,
        )
    if min_points_per_key is not None:
        for key, points in sorted(grouped.items()):
            if len(points) < min_points_per_key:
                add_finding(
                    "min_points_per_key",
                    f"{key} has {len(points)} trend point(s), below minimum {min_points_per_key}",
                    key=key,
                    metric="points",
                    value=len(points),
                    threshold=min_points_per_key,
                )

    if min_latest_tok_per_s is not None:
        for row in latest:
            latest_tok = row.get("latest_median_tok_per_s")
            if latest_tok is None:
                add_finding(
                    "min_latest_tok_per_s",
                    f"{row['key']} latest guest tok/s is missing",
                    key=str(row["key"]),
                    metric="latest_median_tok_per_s",
                    threshold=min_latest_tok_per_s,
                )
            elif isinstance(latest_tok, float) and latest_tok < min_latest_tok_per_s:
                add_finding(
                    "min_latest_tok_per_s",
                    f"{row['key']} latest guest tok/s {latest_tok:.3f} "
                    f"is below minimum {min_latest_tok_per_s:.3f}",
                    key=str(row["key"]),
                    metric="latest_median_tok_per_s",
                    value=latest_tok,
                    threshold=min_latest_tok_per_s,
                )
    if min_latest_wall_tok_per_s is not None:
        for row in latest:
            latest_wall_tok = row.get("latest_wall_tok_per_s_median")
            if latest_wall_tok is None:
                add_finding(
                    "min_latest_wall_tok_per_s",
                    f"{row['key']} latest wall tok/s is missing",
                    key=str(row["key"]),
                    metric="latest_wall_tok_per_s_median",
                    threshold=min_latest_wall_tok_per_s,
                )
            elif isinstance(latest_wall_tok, float) and latest_wall_tok < min_latest_wall_tok_per_s:
                add_finding(
                    "min_latest_wall_tok_per_s",
                    f"{row['key']} latest wall tok/s {latest_wall_tok:.3f} "
                    f"is below minimum {min_latest_wall_tok_per_s:.3f}",
                    key=str(row["key"]),
                    metric="latest_wall_tok_per_s_median",
                    value=latest_wall_tok,
                    threshold=min_latest_wall_tok_per_s,
                )

    if fail_on_tok_regression_pct is not None:
        threshold = -abs(fail_on_tok_regression_pct)
        for row in latest:
            delta = row.get("median_tok_per_s_delta_pct")
            if isinstance(delta, float) and delta < threshold:
                add_finding(
                    "tok_regression_pct",
                    f"{row['key']} guest tok/s regressed {delta:.3f}% "
                    f"(threshold {threshold:.3f}%)",
                    key=str(row["key"]),
                    metric="median_tok_per_s_delta_pct",
                    value=delta,
                    threshold=threshold,
                )
    if fail_on_wall_tok_regression_pct is not None:
        threshold = -abs(fail_on_wall_tok_regression_pct)
        for row in latest:
            delta = row.get("wall_tok_per_s_delta_pct")
            if isinstance(delta, float) and delta < threshold:
                add_finding(
                    "wall_tok_regression_pct",
                    f"{row['key']} wall tok/s regressed {delta:.3f}% "
                    f"(threshold {threshold:.3f}%)",
                    key=str(row["key"]),
                    metric="wall_tok_per_s_delta_pct",
                    value=delta,
                    threshold=threshold,
                )
    if fail_on_memory_growth_pct is not None:
        threshold = abs(fail_on_memory_growth_pct)
        for row in latest:
            delta = row.get("max_memory_delta_pct")
            if isinstance(delta, float) and delta > threshold:
                add_finding(
                    "memory_growth_pct",
                    f"{row['key']} max memory grew {delta:.3f}% "
                    f"(threshold {threshold:.3f}%)",
                    key=str(row["key"]),
                    metric="max_memory_delta_pct",
                    value=delta,
                    threshold=threshold,
                )
    if fail_on_window_tok_regression_pct is not None:
        threshold = -abs(fail_on_window_tok_regression_pct)
        for row in windows:
            delta = row.get("guest_tok_per_s_delta_pct")
            if isinstance(delta, float) and delta < threshold:
                add_finding(
                    "window_tok_regression_pct",
                    f"{row['key']} recent-window guest tok/s regressed {delta:.3f}% "
                    f"(threshold {threshold:.3f}%)",
                    key=str(row["key"]),
                    metric="guest_tok_per_s_delta_pct",
                    value=delta,
                    threshold=threshold,
                )
    if fail_on_window_wall_tok_regression_pct is not None:
        threshold = -abs(fail_on_window_wall_tok_regression_pct)
        for row in windows:
            delta = row.get("wall_tok_per_s_delta_pct")
            if isinstance(delta, float) and delta < threshold:
                add_finding(
                    "window_wall_tok_regression_pct",
                    f"{row['key']} recent-window wall tok/s regressed {delta:.3f}% "
                    f"(threshold {threshold:.3f}%)",
                    key=str(row["key"]),
                    metric="wall_tok_per_s_delta_pct",
                    value=delta,
                    threshold=threshold,
                )
    if fail_on_window_memory_growth_pct is not None:
        threshold = abs(fail_on_window_memory_growth_pct)
        for row in windows:
            delta = row.get("max_memory_delta_pct")
            if isinstance(delta, float) and delta > threshold:
                add_finding(
                    "window_memory_growth_pct",
                    f"{row['key']} recent-window max memory grew {delta:.3f}% "
                    f"(threshold {threshold:.3f}%)",
                    key=str(row["key"]),
                    metric="max_memory_delta_pct",
                    value=delta,
                    threshold=threshold,
                )
    if max_window_tok_cv_pct is not None:
        threshold = max_window_tok_cv_pct
        for row in windows:
            value = row.get("guest_tok_per_s_cv_pct")
            if isinstance(value, float) and value > threshold:
                add_finding(
                    "window_tok_cv_pct",
                    f"{row['key']} recent-window guest tok/s CV {value:.3f}% "
                    f"exceeds maximum {threshold:.3f}%",
                    key=str(row["key"]),
                    metric="guest_tok_per_s_cv_pct",
                    value=value,
                    threshold=threshold,
                )
    if max_window_wall_tok_cv_pct is not None:
        threshold = max_window_wall_tok_cv_pct
        for row in windows:
            value = row.get("wall_tok_per_s_cv_pct")
            if isinstance(value, float) and value > threshold:
                add_finding(
                    "window_wall_tok_cv_pct",
                    f"{row['key']} recent-window wall tok/s CV {value:.3f}% "
                    f"exceeds maximum {threshold:.3f}%",
                    key=str(row["key"]),
                    metric="wall_tok_per_s_cv_pct",
                    value=value,
                    threshold=threshold,
                )
    if max_window_memory_cv_pct is not None:
        threshold = max_window_memory_cv_pct
        for row in windows:
            value = row.get("max_memory_cv_pct")
            if isinstance(value, float) and value > threshold:
                add_finding(
                    "window_memory_cv_pct",
                    f"{row['key']} recent-window max memory CV {value:.3f}% "
                    f"exceeds maximum {threshold:.3f}%",
                    key=str(row["key"]),
                    metric="max_memory_cv_pct",
                    value=value,
                    threshold=threshold,
                )

    enabled_failure_gate = (
        fail_on_empty
        or fail_on_airgap
        or fail_on_telemetry
        or fail_on_command_drift
        or fail_on_launch_plan_drift
        or fail_on_environment_drift
        or min_points_per_key is not None
        or min_latest_tok_per_s is not None
        or min_latest_wall_tok_per_s is not None
        or fail_on_tok_regression_pct is not None
        or fail_on_wall_tok_regression_pct is not None
        or fail_on_memory_growth_pct is not None
        or fail_on_window_tok_regression_pct is not None
        or fail_on_window_wall_tok_regression_pct is not None
        or fail_on_window_memory_growth_pct is not None
        or max_window_tok_cv_pct is not None
        or max_window_wall_tok_cv_pct is not None
        or max_window_memory_cv_pct is not None
    )
    status = "fail" if findings and enabled_failure_gate else "pass"
    drift_summary = {
        "command_sha256": len(command_drift),
        "launch_plan_sha256": len(launch_plan_drift),
        "environment_sha256": len(environment_drift),
    }
    summary = {
        "trend_keys": len(grouped),
        "trend_points": len(all_points),
        "latest_rows": len(latest),
        "window_rows": len(windows),
        "findings": len(findings),
        "drift_keys": sum(drift_summary.values()),
        "status_counts": count_by([point.status for point in all_points]),
        "latest_status_counts": count_by([str(row.get("latest_status") or "-") for row in latest]),
        "command_airgap_status_counts": count_by([point.command_airgap_status for point in all_points]),
        "telemetry_status_counts": count_by([point.telemetry_status for point in all_points]),
        "drift": drift_summary,
    }
    return {
        "generated_at": iso_now(),
        "status": status,
        "summary": summary,
        "findings": findings,
        "finding_rows": [asdict(row) for row in finding_rows],
        "thresholds": {
            "fail_on_empty": fail_on_empty,
            "fail_on_airgap": fail_on_airgap,
            "fail_on_telemetry": fail_on_telemetry,
            "fail_on_command_drift": fail_on_command_drift,
            "fail_on_launch_plan_drift": fail_on_launch_plan_drift,
            "fail_on_environment_drift": fail_on_environment_drift,
            "min_points_per_key": min_points_per_key,
            "min_latest_tok_per_s": min_latest_tok_per_s,
            "min_latest_wall_tok_per_s": min_latest_wall_tok_per_s,
            "fail_on_tok_regression_pct": fail_on_tok_regression_pct,
            "fail_on_wall_tok_regression_pct": fail_on_wall_tok_regression_pct,
            "fail_on_memory_growth_pct": fail_on_memory_growth_pct,
            "fail_on_window_tok_regression_pct": fail_on_window_tok_regression_pct,
            "fail_on_window_wall_tok_regression_pct": fail_on_window_wall_tok_regression_pct,
            "fail_on_window_memory_growth_pct": fail_on_window_memory_growth_pct,
            "max_window_tok_cv_pct": max_window_tok_cv_pct,
            "max_window_wall_tok_cv_pct": max_window_wall_tok_cv_pct,
            "max_window_memory_cv_pct": max_window_memory_cv_pct,
            "window_points": window_points,
        },
        "trend_keys": len(grouped),
        "trend_points": len(all_points),
        "latest": latest,
        "windows": windows,
        "drift": {
            "command_sha256": command_drift,
            "launch_plan_sha256": launch_plan_drift,
            "environment_sha256": environment_drift,
        },
        "trends": {
            key: [asdict(point) for point in points]
            for key, points in sorted(grouped.items())
        },
    }


def format_value(value: object) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(report: dict[str, object]) -> str:
    latest = report["latest"]
    assert isinstance(latest, list)
    lines = [
        "# Benchmark Trend Export",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Trend keys: {report['trend_keys']}",
        f"Trend points: {report['trend_points']}",
        "",
    ]
    thresholds = report.get("thresholds")
    if isinstance(thresholds, dict):
        active_thresholds = [
            f"{key}={format_value(value)}"
            for key, value in thresholds.items()
            if value not in (None, False, "")
        ]
        if active_thresholds:
            lines.append("Thresholds: " + ", ".join(active_thresholds))
            lines.append("")
    findings = report["findings"]
    assert isinstance(findings, list)
    if findings:
        lines.append("## Findings")
        lines.extend(f"- {finding}" for finding in findings)
        lines.append("")
    if latest:
        lines.extend(
            [
                "| Key | Points | Latest commit | Status | Air-gap | Telemetry | Guest tok/s | Guest delta % | Wall tok/s | Wall delta % | Max memory bytes | Memory delta % | Source |",
                "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in latest:
            values = {key: format_value(value) for key, value in row.items()}
            lines.append(
                "| {key} | {points} | {latest_commit} | {latest_status} | {latest_airgap_status} | "
                "{latest_telemetry_status} | {latest_median_tok_per_s} | {median_tok_per_s_delta_pct} | "
                "{latest_wall_tok_per_s_median} | {wall_tok_per_s_delta_pct} | "
                "{latest_max_memory_bytes} | {max_memory_delta_pct} | {latest_source} |".format(**values)
            )
    else:
        lines.append("No supported benchmark artifacts found.")
    drift = report.get("drift")
    if isinstance(drift, dict):
        drift_lines: list[str] = []
        for field in ("command_sha256", "launch_plan_sha256", "environment_sha256"):
            rows = drift.get(field)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                hashes = row.get("hashes")
                hash_count = len(hashes) if isinstance(hashes, list) else 0
                drift_lines.append(f"- {row.get('key', '-')} {field}: {hash_count} value(s)")
        if drift_lines:
            lines.extend(["", "## Drift", *drift_lines])
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "key",
        "points",
        "latest_generated_at",
        "latest_commit",
        "latest_status",
        "latest_airgap_status",
        "latest_telemetry_status",
        "latest_median_tok_per_s",
        "previous_median_tok_per_s",
        "median_tok_per_s_delta_pct",
        "latest_wall_tok_per_s_median",
        "previous_wall_tok_per_s_median",
        "wall_tok_per_s_delta_pct",
        "latest_max_memory_bytes",
        "previous_max_memory_bytes",
        "max_memory_delta_pct",
        "latest_source",
        "previous_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_points_csv(grouped: dict[str, list[TrendPoint]], path: Path) -> None:
    fields = list(TrendPoint.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for points in grouped.values():
            for point in points:
                writer.writerow(asdict(point))


def write_windows_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "key",
        "window_points_requested",
        "window_points",
        "window_start_generated_at",
        "window_end_generated_at",
        "window_start_commit",
        "window_end_commit",
        "guest_tok_per_s_points",
        "guest_tok_per_s_min",
        "guest_tok_per_s_median",
        "guest_tok_per_s_max",
        "guest_tok_per_s_cv_pct",
        "guest_tok_per_s_delta_pct",
        "wall_tok_per_s_points",
        "wall_tok_per_s_min",
        "wall_tok_per_s_median",
        "wall_tok_per_s_max",
        "wall_tok_per_s_cv_pct",
        "wall_tok_per_s_delta_pct",
        "max_memory_points",
        "max_memory_bytes_min",
        "max_memory_bytes_median",
        "max_memory_bytes_max",
        "max_memory_cv_pct",
        "max_memory_delta_pct",
        "window_start_source",
        "window_end_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_drift_csv(report: dict[str, object], path: Path) -> None:
    fields = ["key", "field", "hashes", "points", "commits", "sources"]
    rows: list[dict[str, object]] = []
    drift = report.get("drift")
    if isinstance(drift, dict):
        for entries in drift.values():
            if isinstance(entries, list):
                rows.extend(row for row in entries if isinstance(row, dict))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: ",".join(str(item) for item in value)
                    if isinstance(value, list)
                    else value
                    for field, value in ((field, row.get(field, "")) for field in fields)
                }
            )


def write_findings_csv(report: dict[str, object], path: Path) -> None:
    fields = ["gate", "key", "metric", "value", "threshold", "message"]
    rows = report.get("finding_rows")
    if not isinstance(rows, list):
        rows = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if not isinstance(row, dict):
                continue
            writer.writerow({field: row.get(field, "") for field in fields})


def junit_report(report: dict[str, object]) -> str:
    findings = report["findings"]
    assert isinstance(findings, list)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_bench_trend_export",
            "tests": "1",
            "failures": "1" if report["status"] == "fail" else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "trend_export"})
    if report["status"] == "fail":
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "benchmark_trend_export_failure",
                "message": "; ".join(str(finding) for finding in findings),
            },
        )
        failure.text = "\n".join(str(finding) for finding in findings)
    return ET.tostring(suite, encoding="unicode") + "\n"


def write_outputs(report: dict[str, object], grouped: dict[str, list[TrendPoint]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bench_trend_export_latest.json"
    markdown_path = output_dir / "bench_trend_export_latest.md"
    latest_csv_path = output_dir / "bench_trend_export_latest.csv"
    points_csv_path = output_dir / "bench_trend_export_points_latest.csv"
    windows_csv_path = output_dir / "bench_trend_export_windows_latest.csv"
    drift_csv_path = output_dir / "bench_trend_export_drift_latest.csv"
    findings_csv_path = output_dir / "bench_trend_export_findings_latest.csv"
    junit_path = output_dir / "bench_trend_export_junit_latest.xml"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    latest = report["latest"]
    assert isinstance(latest, list)
    write_csv([row for row in latest if isinstance(row, dict)], latest_csv_path)
    write_points_csv(grouped, points_csv_path)
    windows = report["windows"]
    assert isinstance(windows, list)
    write_windows_csv([row for row in windows if isinstance(row, dict)], windows_csv_path)
    write_drift_csv(report, drift_csv_path)
    write_findings_csv(report, findings_csv_path)
    junit_path.write_text(junit_report(report), encoding="utf-8")
    return json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Benchmark report file or directory; defaults to bench/results",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/dashboards"))
    parser.add_argument(
        "--max-points-per-key",
        type=int,
        help="Keep only the latest N points per comparable trend key",
    )
    parser.add_argument(
        "--window-points",
        type=int,
        default=5,
        help="Write recent-window stats using the latest N retained points per comparable key",
    )
    parser.add_argument("--fail-on-empty", action="store_true")
    parser.add_argument("--fail-on-airgap", action="store_true")
    parser.add_argument("--fail-on-telemetry", action="store_true")
    parser.add_argument("--fail-on-command-drift", action="store_true")
    parser.add_argument("--fail-on-launch-plan-drift", action="store_true")
    parser.add_argument("--fail-on-environment-drift", action="store_true")
    parser.add_argument(
        "--min-points-per-key",
        type=int,
        help="Fail when any comparable trend key has fewer than this many retained points",
    )
    parser.add_argument(
        "--min-latest-tok-per-s",
        type=float,
        help="Fail when any latest comparable key's guest tok/s is below this absolute floor",
    )
    parser.add_argument(
        "--min-latest-wall-tok-per-s",
        type=float,
        help="Fail when any latest comparable key's host wall-clock tok/s is below this absolute floor",
    )
    parser.add_argument(
        "--fail-on-tok-regression-pct",
        type=float,
        help="Fail when latest guest tok/s falls more than this percent versus the previous point",
    )
    parser.add_argument(
        "--fail-on-wall-tok-regression-pct",
        type=float,
        help="Fail when latest host wall-clock tok/s falls more than this percent versus the previous point",
    )
    parser.add_argument(
        "--fail-on-memory-growth-pct",
        type=float,
        help="Fail when latest max memory grows more than this percent versus the previous point",
    )
    parser.add_argument(
        "--fail-on-window-tok-regression-pct",
        type=float,
        help="Fail when recent-window guest tok/s falls more than this percent from window start to latest point",
    )
    parser.add_argument(
        "--fail-on-window-wall-tok-regression-pct",
        type=float,
        help="Fail when recent-window host wall-clock tok/s falls more than this percent from window start to latest point",
    )
    parser.add_argument(
        "--fail-on-window-memory-growth-pct",
        type=float,
        help="Fail when recent-window max memory grows more than this percent from window start to latest point",
    )
    parser.add_argument(
        "--max-window-tok-cv-pct",
        type=float,
        help="Fail when recent-window guest tok/s coefficient of variation exceeds this percent",
    )
    parser.add_argument(
        "--max-window-wall-tok-cv-pct",
        type=float,
        help="Fail when recent-window host wall-clock tok/s coefficient of variation exceeds this percent",
    )
    parser.add_argument(
        "--max-window-memory-cv-pct",
        type=float,
        help="Fail when recent-window max memory coefficient of variation exceeds this percent",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_points_per_key is not None and args.max_points_per_key <= 0:
        print("error: --max-points-per-key must be positive", file=sys.stderr)
        return 2
    if args.window_points <= 0:
        print("error: --window-points must be positive", file=sys.stderr)
        return 2
    if args.min_points_per_key is not None and args.min_points_per_key <= 0:
        print("error: --min-points-per-key must be positive", file=sys.stderr)
        return 2
    for name in ("min_latest_tok_per_s", "min_latest_wall_tok_per_s"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            option = "--" + name.replace("_", "-")
            print(f"error: {option} must be positive", file=sys.stderr)
            return 2
    for name in (
        "fail_on_tok_regression_pct",
        "fail_on_wall_tok_regression_pct",
        "fail_on_memory_growth_pct",
        "fail_on_window_tok_regression_pct",
        "fail_on_window_wall_tok_regression_pct",
        "fail_on_window_memory_growth_pct",
        "max_window_tok_cv_pct",
        "max_window_wall_tok_cv_pct",
        "max_window_memory_cv_pct",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            option = "--" + name.replace("_", "-")
            print(f"error: {option} must be non-negative", file=sys.stderr)
            return 2

    inputs = args.input or [Path("bench/results")]
    summaries = bench_result_index.load_summaries(inputs)
    points = [to_point(summary) for summary in summaries]
    grouped = group_points(points, args.max_points_per_key)
    report = build_report(
        grouped,
        window_points=args.window_points,
        fail_on_empty=args.fail_on_empty,
        fail_on_airgap=args.fail_on_airgap,
        fail_on_telemetry=args.fail_on_telemetry,
        fail_on_command_drift=args.fail_on_command_drift,
        fail_on_launch_plan_drift=args.fail_on_launch_plan_drift,
        fail_on_environment_drift=args.fail_on_environment_drift,
        min_points_per_key=args.min_points_per_key,
        min_latest_tok_per_s=args.min_latest_tok_per_s,
        min_latest_wall_tok_per_s=args.min_latest_wall_tok_per_s,
        fail_on_tok_regression_pct=args.fail_on_tok_regression_pct,
        fail_on_wall_tok_regression_pct=args.fail_on_wall_tok_regression_pct,
        fail_on_memory_growth_pct=args.fail_on_memory_growth_pct,
        fail_on_window_tok_regression_pct=args.fail_on_window_tok_regression_pct,
        fail_on_window_wall_tok_regression_pct=args.fail_on_window_wall_tok_regression_pct,
        fail_on_window_memory_growth_pct=args.fail_on_window_memory_growth_pct,
        max_window_tok_cv_pct=args.max_window_tok_cv_pct,
        max_window_wall_tok_cv_pct=args.max_window_wall_tok_cv_pct,
        max_window_memory_cv_pct=args.max_window_memory_cv_pct,
    )
    json_path = write_outputs(report, grouped, args.output_dir)
    print(json_path)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
