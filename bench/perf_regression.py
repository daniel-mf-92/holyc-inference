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


def detect_regressions(
    records: list[PerfRecord], tok_threshold_pct: float, memory_threshold_pct: float
) -> list[Regression]:
    regressions: list[Regression] = []
    by_key: dict[str, list[PerfRecord]] = {}
    for record in records:
        by_key.setdefault(record.key, []).append(record)

    for key, key_records in sorted(by_key.items()):
        ordered = sorted(key_records, key=record_sort_key)
        if len(ordered) < 2:
            continue
        baseline = ordered[-2]
        candidate = ordered[-1]

        if baseline.tok_per_s and candidate.tok_per_s is not None:
            delta_pct = (baseline.tok_per_s - candidate.tok_per_s) * 100.0 / baseline.tok_per_s
            if delta_pct > tok_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="tok_per_s",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=baseline.tok_per_s,
                        candidate_value=candidate.tok_per_s,
                        delta_pct=delta_pct,
                        threshold_pct=tok_threshold_pct,
                    )
                )

        if baseline.memory_bytes and candidate.memory_bytes is not None:
            delta_pct = (candidate.memory_bytes - baseline.memory_bytes) * 100.0 / baseline.memory_bytes
            if delta_pct > memory_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="memory_bytes",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=float(baseline.memory_bytes),
                        candidate_value=float(candidate.memory_bytes),
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


def build_report(
    records: list[PerfRecord], tok_threshold_pct: float, memory_threshold_pct: float
) -> dict[str, Any]:
    regressions = detect_regressions(records, tok_threshold_pct, memory_threshold_pct)
    return {
        "generated_at": iso_now(),
        "record_count": len(records),
        "status": "fail" if regressions else "pass",
        "thresholds": {
            "tok_regression_pct": tok_threshold_pct,
            "memory_regression_pct": memory_threshold_pct,
        },
        "summaries": summarize(records),
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
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or [Path("bench/results")]
    records = load_records(inputs)
    report = build_report(records, args.tok_regression_pct, args.memory_regression_pct)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "perf_regression_latest.json"
    md_path = args.output_dir / "perf_regression_latest.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")

    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"status={report['status']}")

    if args.fail_on_regression and report["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
