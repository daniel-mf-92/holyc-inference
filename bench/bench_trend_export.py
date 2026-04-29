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


def build_report(
    grouped: dict[str, list[TrendPoint]],
    *,
    fail_on_empty: bool,
    fail_on_airgap: bool,
    fail_on_telemetry: bool,
) -> dict[str, object]:
    all_points = [point for points in grouped.values() for point in points]
    findings: list[str] = []
    if not all_points:
        findings.append("no supported benchmark artifacts found")
    if fail_on_airgap:
        airgap_failures = [point for point in all_points if point.command_airgap_status == "fail"]
        if airgap_failures:
            findings.append(f"{len(airgap_failures)} trend point(s) violate QEMU air-gap policy")
    if fail_on_telemetry:
        telemetry_failures = [point for point in all_points if point.telemetry_status == "fail"]
        if telemetry_failures:
            findings.append(f"{len(telemetry_failures)} trend point(s) are missing telemetry")

    status = "fail" if findings and (fail_on_empty or fail_on_airgap or fail_on_telemetry) else "pass"
    return {
        "generated_at": iso_now(),
        "status": status,
        "findings": findings,
        "trend_keys": len(grouped),
        "trend_points": len(all_points),
        "latest": latest_rows(grouped),
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
    junit_path = output_dir / "bench_trend_export_junit_latest.xml"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    latest = report["latest"]
    assert isinstance(latest, list)
    write_csv([row for row in latest if isinstance(row, dict)], latest_csv_path)
    write_points_csv(grouped, points_csv_path)
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
    parser.add_argument("--fail-on-empty", action="store_true")
    parser.add_argument("--fail-on-airgap", action="store_true")
    parser.add_argument("--fail-on-telemetry", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_points_per_key is not None and args.max_points_per_key <= 0:
        print("error: --max-points-per-key must be positive", file=sys.stderr)
        return 2

    inputs = args.input or [Path("bench/results")]
    summaries = bench_result_index.load_summaries(inputs)
    points = [to_point(summary) for summary in summaries]
    grouped = group_points(points, args.max_points_per_key)
    report = build_report(
        grouped,
        fail_on_empty=args.fail_on_empty,
        fail_on_airgap=args.fail_on_airgap,
        fail_on_telemetry=args.fail_on_telemetry,
    )
    json_path = write_outputs(report, grouped, args.output_dir)
    print(json_path)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
