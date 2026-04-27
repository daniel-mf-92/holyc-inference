#!/usr/bin/env python3
"""Host-side performance regression dashboard builder.

The tool consumes benchmark records from JSON, JSONL, or CSV files under
bench/results and writes machine-readable plus Markdown dashboards under
bench/dashboards. It is host-side only and does not launch QEMU.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TOK_REGRESSION_PCT = 5.0
DEFAULT_MEMORY_REGRESSION_PCT = 10.0
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class PerfRecord:
    source: str
    commit: str
    timestamp: str
    benchmark: str
    profile: str
    model: str
    quantization: str
    prompt: str
    tok_per_s: float | None
    memory_bytes: int | None

    @property
    def key(self) -> str:
        parts = (self.benchmark, self.profile, self.model, self.quantization, self.prompt)
        return "/".join(part or "-" for part in parts)


@dataclass(frozen=True)
class Regression:
    key: str
    metric: str
    baseline_commit: str
    candidate_commit: str
    baseline_value: float
    candidate_value: float
    delta_pct: float
    threshold_pct: float


@dataclass(frozen=True)
class CommitPoint:
    key: str
    commit: str
    latest_timestamp: str
    records: int
    median_tok_per_s: float | None
    max_memory_bytes: int | None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def first_present(row: dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return str(value)
    return default


def normalize_record(row: dict[str, Any], source: Path, fallback_timestamp: str) -> PerfRecord | None:
    tok_per_s = parse_float(row.get("tok_per_s"))
    tok_per_s_milli = parse_float(row.get("tok_per_s_milli"))
    if tok_per_s is None and tok_per_s_milli is not None:
        tok_per_s = tok_per_s_milli / 1000.0

    memory_bytes = parse_int(
        row.get("memory_bytes")
        or row.get("max_rss_bytes")
        or row.get("rss_bytes")
        or row.get("peak_memory_bytes")
    )

    if tok_per_s is None and memory_bytes is None:
        return None

    return PerfRecord(
        source=str(source),
        commit=first_present(row, ("commit", "git_commit", "sha"), "unknown"),
        timestamp=first_present(row, ("timestamp", "generated_at", "time"), fallback_timestamp),
        benchmark=first_present(row, ("benchmark", "bench", "name", "suite"), source.stem),
        profile=first_present(row, ("profile", "mode"), "default"),
        model=first_present(row, ("model", "model_name"), ""),
        quantization=first_present(row, ("quantization", "quant", "format"), ""),
        prompt=first_present(row, ("prompt", "prompt_id", "case", "scenario"), ""),
        tok_per_s=tok_per_s,
        memory_bytes=memory_bytes,
    )


def flatten_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(payload, dict):
        return

    yielded = False
    for key in RESULT_KEYS:
        nested = payload.get(key)
        if isinstance(nested, list):
            yielded = True
            for item in nested:
                if isinstance(item, dict):
                    merged = {k: v for k, v in payload.items() if k not in RESULT_KEYS}
                    merged.update(item)
                    yield merged

    if not yielded:
        yield payload


def load_json_records(path: Path) -> Iterable[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    yield from flatten_json_payload(payload)


def load_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            yield from flatten_json_payload(payload)


def load_csv_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def load_records(paths: Iterable[Path]) -> list[PerfRecord]:
    records: list[PerfRecord] = []
    for path in sorted(paths):
        if path.is_dir():
            children = [
                child
                for child in path.rglob("*")
                if child.suffix.lower() in {".json", ".jsonl", ".csv"} and child.is_file()
            ]
            records.extend(load_records(children))
            continue

        suffix = path.suffix.lower()
        fallback_timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        if suffix == ".json":
            raw_rows = load_json_records(path)
        elif suffix == ".jsonl":
            raw_rows = load_jsonl_records(path)
        elif suffix == ".csv":
            raw_rows = load_csv_records(path)
        else:
            continue

        for row in raw_rows:
            normalized = normalize_record(row, path, fallback_timestamp)
            if normalized is not None:
                records.append(normalized)
    return records


def record_sort_key(record: PerfRecord) -> tuple[str, str]:
    return (record.timestamp, record.commit)


def summarize(records: list[PerfRecord]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, list[PerfRecord]] = {}
    for record in records:
        by_key.setdefault(record.key, []).append(record)

    summaries: dict[str, dict[str, Any]] = {}
    for key, key_records in sorted(by_key.items()):
        tps_values = [record.tok_per_s for record in key_records if record.tok_per_s is not None]
        memory_values = [record.memory_bytes for record in key_records if record.memory_bytes is not None]
        summaries[key] = {
            "records": len(key_records),
            "latest_commit": sorted(key_records, key=record_sort_key)[-1].commit,
            "median_tok_per_s": statistics.median(tps_values) if tps_values else None,
            "max_memory_bytes": max(memory_values) if memory_values else None,
        }
    return summaries


def commit_points(records: list[PerfRecord]) -> list[CommitPoint]:
    by_key_commit: dict[tuple[str, str], list[PerfRecord]] = {}
    for record in records:
        by_key_commit.setdefault((record.key, record.commit), []).append(record)

    points: list[CommitPoint] = []
    for (key, commit), commit_records in sorted(by_key_commit.items()):
        tps_values = [record.tok_per_s for record in commit_records if record.tok_per_s is not None]
        memory_values = [record.memory_bytes for record in commit_records if record.memory_bytes is not None]
        points.append(
            CommitPoint(
                key=key,
                commit=commit,
                latest_timestamp=max(record.timestamp for record in commit_records),
                records=len(commit_records),
                median_tok_per_s=statistics.median(tps_values) if tps_values else None,
                max_memory_bytes=max(memory_values) if memory_values else None,
            )
        )
    return sorted(points, key=lambda point: (point.key, point.latest_timestamp, point.commit))


def select_comparison_points(
    points: list[CommitPoint], baseline_commit: str | None, candidate_commit: str | None
) -> tuple[CommitPoint, CommitPoint] | None:
    if candidate_commit is not None:
        candidates = [point for point in points if point.commit == candidate_commit]
        if not candidates:
            return None
        candidate = candidates[-1]
    else:
        candidate = points[-1]

    eligible_baselines = [point for point in points if point.commit != candidate.commit]
    if baseline_commit is not None:
        eligible_baselines = [point for point in eligible_baselines if point.commit == baseline_commit]
    if not eligible_baselines:
        return None

    return eligible_baselines[-1], candidate


def detect_regressions(
    records: list[PerfRecord],
    tok_threshold_pct: float,
    memory_threshold_pct: float,
    baseline_commit: str | None = None,
    candidate_commit: str | None = None,
) -> list[Regression]:
    regressions: list[Regression] = []
    by_key: dict[str, list[CommitPoint]] = {}
    for point in commit_points(records):
        by_key.setdefault(point.key, []).append(point)

    for key, key_points in sorted(by_key.items()):
        comparison = select_comparison_points(key_points, baseline_commit, candidate_commit)
        if comparison is None:
            continue
        baseline, candidate = comparison

        if baseline.median_tok_per_s and candidate.median_tok_per_s is not None:
            delta_pct = (
                (baseline.median_tok_per_s - candidate.median_tok_per_s)
                * 100.0
                / baseline.median_tok_per_s
            )
            if delta_pct > tok_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="tok_per_s",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=baseline.median_tok_per_s,
                        candidate_value=candidate.median_tok_per_s,
                        delta_pct=delta_pct,
                        threshold_pct=tok_threshold_pct,
                    )
                )

        if baseline.max_memory_bytes and candidate.max_memory_bytes is not None:
            delta_pct = (
                (candidate.max_memory_bytes - baseline.max_memory_bytes)
                * 100.0
                / baseline.max_memory_bytes
            )
            if delta_pct > memory_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="memory_bytes",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=float(baseline.max_memory_bytes),
                        candidate_value=float(candidate.max_memory_bytes),
                        delta_pct=delta_pct,
                        threshold_pct=memory_threshold_pct,
                    )
                )
    return regressions


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Perf Regression Dashboard",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Records: {report['record_count']}",
        f"Regressions: {len(report['regressions'])}",
        "",
        "## Regressions",
        "",
    ]

    if report["regressions"]:
        lines.append("| Key | Metric | Baseline | Candidate | Delta | Threshold |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for regression in report["regressions"]:
            lines.append(
                "| {key} | {metric} | {baseline_value:.3f} | {candidate_value:.3f} | "
                "{delta_pct:.2f}% | {threshold_pct:.2f}% |".format(**regression)
            )
    else:
        lines.append("No regressions detected.")

    lines.extend(["", "## Commit Points", ""])
    if report["commit_points"]:
        lines.append("| Key | Commit | Records | Median tok/s | Max Memory Bytes |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for point in report["commit_points"]:
            tps = point["median_tok_per_s"]
            memory = point["max_memory_bytes"]
            tps_cell = f"{tps:.3f}" if tps is not None else "-"
            memory_cell = str(memory) if memory is not None else "-"
            lines.append(
                f"| {point['key']} | {point['commit']} | {point['records']} | "
                f"{tps_cell} | {memory_cell} |"
            )
    else:
        lines.append("No commit-level performance points found.")

    lines.extend(["", "## Latest Summary", ""])
    if report["summaries"]:
        lines.append("| Key | Records | Latest Commit | Median tok/s | Max Memory Bytes |")
        lines.append("| --- | ---: | --- | ---: | ---: |")
        for key, summary in report["summaries"].items():
            tps = summary["median_tok_per_s"]
            memory = summary["max_memory_bytes"]
            tps_cell = f"{tps:.3f}" if tps is not None else "-"
            memory_cell = str(memory) if memory is not None else "-"
            lines.append(
                f"| {key} | {summary['records']} | {summary['latest_commit']} | "
                f"{tps_cell} | {memory_cell} |"
            )
    else:
        lines.append("No performance records found.")

    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_dashboard_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "perf_regression_latest.json"
    md_path = output_dir / "perf_regression_latest.md"
    commit_points_csv = output_dir / "perf_regression_commit_points_latest.csv"
    regressions_csv = output_dir / "perf_regression_regressions_latest.csv"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(
        commit_points_csv,
        report["commit_points"],
        [
            "key",
            "commit",
            "latest_timestamp",
            "records",
            "median_tok_per_s",
            "max_memory_bytes",
        ],
    )
    write_csv(
        regressions_csv,
        report["regressions"],
        [
            "key",
            "metric",
            "baseline_commit",
            "candidate_commit",
            "baseline_value",
            "candidate_value",
            "delta_pct",
            "threshold_pct",
        ],
    )

    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_commit_points_csv={commit_points_csv}")
    print(f"wrote_regressions_csv={regressions_csv}")


def build_report(
    records: list[PerfRecord],
    tok_threshold_pct: float,
    memory_threshold_pct: float,
    baseline_commit: str | None = None,
    candidate_commit: str | None = None,
) -> dict[str, Any]:
    regressions = detect_regressions(
        records,
        tok_threshold_pct,
        memory_threshold_pct,
        baseline_commit=baseline_commit,
        candidate_commit=candidate_commit,
    )
    return {
        "generated_at": iso_now(),
        "record_count": len(records),
        "status": "fail" if regressions else "pass",
        "comparison": {
            "baseline_commit": baseline_commit,
            "candidate_commit": candidate_commit,
            "mode": "explicit" if baseline_commit or candidate_commit else "latest-distinct-commits",
        },
        "thresholds": {
            "tok_regression_pct": tok_threshold_pct,
            "memory_regression_pct": memory_threshold_pct,
        },
        "summaries": summarize(records),
        "commit_points": [asdict(point) for point in commit_points(records)],
        "regressions": [asdict(regression) for regression in regressions],
        "records": [asdict(record) for record in sorted(records, key=lambda item: (item.key, item.timestamp))],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Result file or directory to scan; defaults to bench/results",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/dashboards"))
    parser.add_argument("--tok-regression-pct", type=float, default=DEFAULT_TOK_REGRESSION_PCT)
    parser.add_argument("--memory-regression-pct", type=float, default=DEFAULT_MEMORY_REGRESSION_PCT)
    parser.add_argument("--baseline-commit", help="Commit SHA/name to use as the comparison baseline")
    parser.add_argument("--candidate-commit", help="Commit SHA/name to compare against the baseline")
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or [Path("bench/results")]
    records = load_records(inputs)
    report = build_report(
        records,
        args.tok_regression_pct,
        args.memory_regression_pct,
        baseline_commit=args.baseline_commit,
        candidate_commit=args.candidate_commit,
    )

    write_dashboard_outputs(report, args.output_dir)
    print(f"status={report['status']}")

    if args.fail_on_regression and report["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
