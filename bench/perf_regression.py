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
import xml.etree.ElementTree as ET
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
    wall_tok_per_s: float | None
    us_per_token: float | None
    wall_us_per_token: float | None
    memory_bytes: int | None
    ttft_us: int | None = None
    host_overhead_pct: float | None = None
    prompt_suite_sha256: str = ""

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
    tok_per_s_records: int
    wall_tok_per_s_records: int
    us_per_token_records: int
    wall_us_per_token_records: int
    memory_records: int
    ttft_us_records: int
    host_overhead_records: int
    median_tok_per_s: float | None
    p05_tok_per_s: float | None
    median_wall_tok_per_s: float | None
    median_us_per_token: float | None
    p95_us_per_token: float | None
    median_wall_us_per_token: float | None
    p95_wall_us_per_token: float | None
    median_ttft_us: float | None
    p95_ttft_us: float | None
    median_host_overhead_pct: float | None
    tok_per_s_cv_pct: float | None
    max_memory_bytes: int | None
    prompt_suite_sha256: str


@dataclass(frozen=True)
class SampleViolation:
    key: str
    commit: str
    records: int
    minimum_records: int


@dataclass(frozen=True)
class VariabilityViolation:
    key: str
    commit: str
    records: int
    tok_per_s_cv_pct: float
    threshold_pct: float


@dataclass(frozen=True)
class CommitCoverageViolation:
    key: str
    commits: int
    minimum_commits: int
    latest_commit: str


@dataclass(frozen=True)
class ComparisonCoverageViolation:
    key: str
    baseline_commit: str | None
    candidate_commit: str | None
    missing_commits: str


@dataclass(frozen=True)
class PromptSuiteDriftViolation:
    key: str
    hashes: list[str]
    commits: list[str]
    sources: list[str]


@dataclass(frozen=True)
class TelemetryCoverageViolation:
    key: str
    commit: str
    metric: str
    records: int
    present_records: int


@dataclass(frozen=True)
class ComparisonRow:
    key: str
    baseline_commit: str
    candidate_commit: str
    baseline_latest_timestamp: str
    candidate_latest_timestamp: str
    baseline_records: int
    candidate_records: int
    median_tok_per_s_baseline: float | None
    median_tok_per_s_candidate: float | None
    median_tok_per_s_delta_pct: float | None
    p05_tok_per_s_baseline: float | None
    p05_tok_per_s_candidate: float | None
    p05_tok_per_s_delta_pct: float | None
    median_wall_tok_per_s_baseline: float | None
    median_wall_tok_per_s_candidate: float | None
    median_wall_tok_per_s_delta_pct: float | None
    median_us_per_token_baseline: float | None
    median_us_per_token_candidate: float | None
    median_us_per_token_delta_pct: float | None
    median_wall_us_per_token_baseline: float | None
    median_wall_us_per_token_candidate: float | None
    median_wall_us_per_token_delta_pct: float | None
    max_memory_bytes_baseline: int | None
    max_memory_bytes_candidate: int | None
    max_memory_bytes_delta_pct: float | None
    median_ttft_us_baseline: float | None
    median_ttft_us_candidate: float | None
    median_ttft_us_delta_pct: float | None
    p95_ttft_us_baseline: float | None
    p95_ttft_us_candidate: float | None
    p95_ttft_us_delta_pct: float | None
    median_host_overhead_pct_baseline: float | None
    median_host_overhead_pct_candidate: float | None
    median_host_overhead_pct_delta_pct: float | None


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


def parse_duration_us(row: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = parse_float(row.get(name))
        if value is not None:
            return int(value)
        value = parse_float(row.get(f"{name}_ms"))
        if value is not None:
            return int(value * 1000.0)
        value = parse_float(row.get(f"{name}_s"))
        if value is not None:
            return int(value * 1_000_000.0)
    return None


def first_present(row: dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return str(value)
    return default


def prompt_suite_sha256(row: dict[str, Any]) -> str:
    value = first_present(row, ("prompt_suite_sha256", "suite_sha256", "prompt_suite_hash"))
    if value:
        return value
    prompt_suite = row.get("prompt_suite")
    if isinstance(prompt_suite, dict):
        nested = prompt_suite.get("suite_sha256") or prompt_suite.get("sha256")
        if nested:
            return str(nested)
    return ""


def normalize_record(row: dict[str, Any], source: Path, fallback_timestamp: str) -> PerfRecord | None:
    tok_per_s = parse_float(row.get("tok_per_s"))
    tok_per_s_milli = parse_float(row.get("tok_per_s_milli"))
    if tok_per_s is None and tok_per_s_milli is not None:
        tok_per_s = tok_per_s_milli / 1000.0
    wall_tok_per_s = parse_float(
        row.get("wall_tok_per_s") or row.get("host_tok_per_s") or row.get("host_wall_tok_per_s")
    )
    wall_tok_per_s_milli = parse_float(
        row.get("wall_tok_per_s_milli")
        or row.get("host_tok_per_s_milli")
        or row.get("host_wall_tok_per_s_milli")
    )
    if wall_tok_per_s is None and wall_tok_per_s_milli is not None:
        wall_tok_per_s = wall_tok_per_s_milli / 1000.0
    us_per_token = parse_float(row.get("us_per_token") or row.get("us_per_token_median"))
    wall_us_per_token = parse_float(
        row.get("wall_us_per_token")
        or row.get("wall_us_per_token_median")
        or row.get("host_us_per_token")
        or row.get("host_wall_us_per_token")
    )

    memory_bytes = parse_int(
        row.get("memory_bytes")
        or row.get("max_rss_bytes")
        or row.get("rss_bytes")
        or row.get("peak_memory_bytes")
    )

    ttft_us = parse_duration_us(
        row,
        "ttft_us",
        "ttft",
        "time_to_first_token_us",
        "time_to_first_token",
        "first_token_us",
        "first_token",
    )

    host_overhead_pct = parse_float(row.get("host_overhead_pct") or row.get("overhead_pct"))

    if (
        tok_per_s is None
        and wall_tok_per_s is None
        and us_per_token is None
        and wall_us_per_token is None
        and memory_bytes is None
        and ttft_us is None
        and host_overhead_pct is None
    ):
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
        wall_tok_per_s=wall_tok_per_s,
        us_per_token=us_per_token,
        wall_us_per_token=wall_us_per_token,
        memory_bytes=memory_bytes,
        ttft_us=ttft_us,
        host_overhead_pct=host_overhead_pct,
        prompt_suite_sha256=prompt_suite_sha256(row),
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
        wall_tps_values = [
            record.wall_tok_per_s for record in key_records if record.wall_tok_per_s is not None
        ]
        us_per_token_values = [
            record.us_per_token for record in key_records if record.us_per_token is not None
        ]
        wall_us_per_token_values = [
            record.wall_us_per_token
            for record in key_records
            if record.wall_us_per_token is not None
        ]
        memory_values = [record.memory_bytes for record in key_records if record.memory_bytes is not None]
        ttft_values = [record.ttft_us for record in key_records if record.ttft_us is not None]
        host_overhead_values = [
            record.host_overhead_pct
            for record in key_records
            if record.host_overhead_pct is not None
        ]
        summaries[key] = {
            "records": len(key_records),
            "latest_commit": sorted(key_records, key=record_sort_key)[-1].commit,
            "median_tok_per_s": statistics.median(tps_values) if tps_values else None,
            "p05_tok_per_s": percentile(tps_values, 5.0),
            "median_wall_tok_per_s": statistics.median(wall_tps_values) if wall_tps_values else None,
            "median_us_per_token": (
                statistics.median(us_per_token_values) if us_per_token_values else None
            ),
            "p95_us_per_token": percentile(us_per_token_values, 95.0),
            "median_wall_us_per_token": (
                statistics.median(wall_us_per_token_values) if wall_us_per_token_values else None
            ),
            "p95_wall_us_per_token": percentile(wall_us_per_token_values, 95.0),
            "median_ttft_us": statistics.median(ttft_values) if ttft_values else None,
            "p95_ttft_us": percentile([float(value) for value in ttft_values], 95.0),
            "median_host_overhead_pct": (
                statistics.median(host_overhead_values) if host_overhead_values else None
            ),
            "max_memory_bytes": max(memory_values) if memory_values else None,
        }
    return summaries


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def commit_points(records: list[PerfRecord]) -> list[CommitPoint]:
    by_key_commit: dict[tuple[str, str], list[PerfRecord]] = {}
    for record in records:
        by_key_commit.setdefault((record.key, record.commit), []).append(record)

    points: list[CommitPoint] = []
    for (key, commit), commit_records in sorted(by_key_commit.items()):
        tps_values = [record.tok_per_s for record in commit_records if record.tok_per_s is not None]
        wall_tps_values = [
            record.wall_tok_per_s for record in commit_records if record.wall_tok_per_s is not None
        ]
        us_per_token_values = [
            record.us_per_token for record in commit_records if record.us_per_token is not None
        ]
        wall_us_per_token_values = [
            record.wall_us_per_token
            for record in commit_records
            if record.wall_us_per_token is not None
        ]
        memory_values = [record.memory_bytes for record in commit_records if record.memory_bytes is not None]
        ttft_values = [record.ttft_us for record in commit_records if record.ttft_us is not None]
        host_overhead_values = [
            record.host_overhead_pct
            for record in commit_records
            if record.host_overhead_pct is not None
        ]
        tps_cv_pct = coefficient_of_variation_pct(tps_values)
        prompt_hashes = sorted(
            {record.prompt_suite_sha256 for record in commit_records if record.prompt_suite_sha256}
        )
        points.append(
            CommitPoint(
                key=key,
                commit=commit,
                latest_timestamp=max(record.timestamp for record in commit_records),
                records=len(commit_records),
                tok_per_s_records=len(tps_values),
                wall_tok_per_s_records=len(wall_tps_values),
                us_per_token_records=len(us_per_token_values),
                wall_us_per_token_records=len(wall_us_per_token_values),
                memory_records=len(memory_values),
                ttft_us_records=len(ttft_values),
                host_overhead_records=len(host_overhead_values),
                median_tok_per_s=statistics.median(tps_values) if tps_values else None,
                p05_tok_per_s=percentile(tps_values, 5.0),
                median_wall_tok_per_s=statistics.median(wall_tps_values) if wall_tps_values else None,
                median_us_per_token=(
                    statistics.median(us_per_token_values) if us_per_token_values else None
                ),
                p95_us_per_token=percentile(us_per_token_values, 95.0),
                median_wall_us_per_token=(
                    statistics.median(wall_us_per_token_values)
                    if wall_us_per_token_values
                    else None
                ),
                p95_wall_us_per_token=percentile(wall_us_per_token_values, 95.0),
                median_ttft_us=statistics.median(ttft_values) if ttft_values else None,
                p95_ttft_us=percentile([float(value) for value in ttft_values], 95.0),
                median_host_overhead_pct=(
                    statistics.median(host_overhead_values) if host_overhead_values else None
                ),
                tok_per_s_cv_pct=tps_cv_pct,
                max_memory_bytes=max(memory_values) if memory_values else None,
                prompt_suite_sha256=";".join(prompt_hashes),
            )
        )
    return sorted(points, key=lambda point: (point.key, point.latest_timestamp, point.commit))


def coefficient_of_variation_pct(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean <= 0.0:
        return None
    return statistics.stdev(values) * 100.0 / mean


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


def throughput_delta_pct(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0.0:
        return None
    return (baseline - candidate) * 100.0 / baseline


def growth_delta_pct(baseline: float | int | None, candidate: float | int | None) -> float | None:
    if baseline is None or candidate is None or float(baseline) <= 0.0:
        return None
    return (float(candidate) - float(baseline)) * 100.0 / float(baseline)


def comparison_rows(
    points: list[CommitPoint], baseline_commit: str | None, candidate_commit: str | None
) -> list[ComparisonRow]:
    by_key: dict[str, list[CommitPoint]] = {}
    for point in points:
        by_key.setdefault(point.key, []).append(point)

    rows: list[ComparisonRow] = []
    for key, key_points in sorted(by_key.items()):
        comparison = select_comparison_points(key_points, baseline_commit, candidate_commit)
        if comparison is None:
            continue
        baseline, candidate = comparison
        rows.append(
            ComparisonRow(
                key=key,
                baseline_commit=baseline.commit,
                candidate_commit=candidate.commit,
                baseline_latest_timestamp=baseline.latest_timestamp,
                candidate_latest_timestamp=candidate.latest_timestamp,
                baseline_records=baseline.records,
                candidate_records=candidate.records,
                median_tok_per_s_baseline=baseline.median_tok_per_s,
                median_tok_per_s_candidate=candidate.median_tok_per_s,
                median_tok_per_s_delta_pct=throughput_delta_pct(
                    baseline.median_tok_per_s, candidate.median_tok_per_s
                ),
                p05_tok_per_s_baseline=baseline.p05_tok_per_s,
                p05_tok_per_s_candidate=candidate.p05_tok_per_s,
                p05_tok_per_s_delta_pct=throughput_delta_pct(
                    baseline.p05_tok_per_s, candidate.p05_tok_per_s
                ),
                median_wall_tok_per_s_baseline=baseline.median_wall_tok_per_s,
                median_wall_tok_per_s_candidate=candidate.median_wall_tok_per_s,
                median_wall_tok_per_s_delta_pct=throughput_delta_pct(
                    baseline.median_wall_tok_per_s, candidate.median_wall_tok_per_s
                ),
                median_us_per_token_baseline=baseline.median_us_per_token,
                median_us_per_token_candidate=candidate.median_us_per_token,
                median_us_per_token_delta_pct=growth_delta_pct(
                    baseline.median_us_per_token, candidate.median_us_per_token
                ),
                median_wall_us_per_token_baseline=baseline.median_wall_us_per_token,
                median_wall_us_per_token_candidate=candidate.median_wall_us_per_token,
                median_wall_us_per_token_delta_pct=growth_delta_pct(
                    baseline.median_wall_us_per_token, candidate.median_wall_us_per_token
                ),
                max_memory_bytes_baseline=baseline.max_memory_bytes,
                max_memory_bytes_candidate=candidate.max_memory_bytes,
                max_memory_bytes_delta_pct=growth_delta_pct(
                    baseline.max_memory_bytes, candidate.max_memory_bytes
                ),
                median_ttft_us_baseline=baseline.median_ttft_us,
                median_ttft_us_candidate=candidate.median_ttft_us,
                median_ttft_us_delta_pct=growth_delta_pct(
                    baseline.median_ttft_us, candidate.median_ttft_us
                ),
                p95_ttft_us_baseline=baseline.p95_ttft_us,
                p95_ttft_us_candidate=candidate.p95_ttft_us,
                p95_ttft_us_delta_pct=growth_delta_pct(baseline.p95_ttft_us, candidate.p95_ttft_us),
                median_host_overhead_pct_baseline=baseline.median_host_overhead_pct,
                median_host_overhead_pct_candidate=candidate.median_host_overhead_pct,
                median_host_overhead_pct_delta_pct=growth_delta_pct(
                    baseline.median_host_overhead_pct, candidate.median_host_overhead_pct
                ),
            )
        )
    return rows


def detect_regressions(
    records: list[PerfRecord],
    tok_threshold_pct: float,
    memory_threshold_pct: float,
    wall_tok_threshold_pct: float | None = None,
    ttft_threshold_pct: float | None = None,
    p95_ttft_threshold_pct: float | None = None,
    host_overhead_threshold_pct: float | None = None,
    p05_tok_threshold_pct: float | None = None,
    us_per_token_threshold_pct: float | None = None,
    wall_us_per_token_threshold_pct: float | None = None,
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

        if (
            p05_tok_threshold_pct is not None
            and baseline.p05_tok_per_s
            and candidate.p05_tok_per_s is not None
        ):
            delta_pct = (
                (baseline.p05_tok_per_s - candidate.p05_tok_per_s)
                * 100.0
                / baseline.p05_tok_per_s
            )
            if delta_pct > p05_tok_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="tok_per_s_p05",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=baseline.p05_tok_per_s,
                        candidate_value=candidate.p05_tok_per_s,
                        delta_pct=delta_pct,
                        threshold_pct=p05_tok_threshold_pct,
                    )
                )

        if (
            wall_tok_threshold_pct is not None
            and baseline.median_wall_tok_per_s
            and candidate.median_wall_tok_per_s is not None
        ):
            delta_pct = (
                (baseline.median_wall_tok_per_s - candidate.median_wall_tok_per_s)
                * 100.0
                / baseline.median_wall_tok_per_s
            )
            if delta_pct > wall_tok_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="wall_tok_per_s",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=baseline.median_wall_tok_per_s,
                        candidate_value=candidate.median_wall_tok_per_s,
                        delta_pct=delta_pct,
                        threshold_pct=wall_tok_threshold_pct,
                    )
                )

        if (
            us_per_token_threshold_pct is not None
            and baseline.median_us_per_token
            and candidate.median_us_per_token is not None
        ):
            delta_pct = (
                (candidate.median_us_per_token - baseline.median_us_per_token)
                * 100.0
                / baseline.median_us_per_token
            )
            if delta_pct > us_per_token_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="us_per_token",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=baseline.median_us_per_token,
                        candidate_value=candidate.median_us_per_token,
                        delta_pct=delta_pct,
                        threshold_pct=us_per_token_threshold_pct,
                    )
                )

        if (
            wall_us_per_token_threshold_pct is not None
            and baseline.median_wall_us_per_token
            and candidate.median_wall_us_per_token is not None
        ):
            delta_pct = (
                (candidate.median_wall_us_per_token - baseline.median_wall_us_per_token)
                * 100.0
                / baseline.median_wall_us_per_token
            )
            if delta_pct > wall_us_per_token_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="wall_us_per_token",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=baseline.median_wall_us_per_token,
                        candidate_value=candidate.median_wall_us_per_token,
                        delta_pct=delta_pct,
                        threshold_pct=wall_us_per_token_threshold_pct,
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

        if (
            ttft_threshold_pct is not None
            and baseline.median_ttft_us
            and candidate.median_ttft_us is not None
        ):
            delta_pct = (
                (candidate.median_ttft_us - baseline.median_ttft_us)
                * 100.0
                / baseline.median_ttft_us
            )
            if delta_pct > ttft_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="ttft_us",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=baseline.median_ttft_us,
                        candidate_value=candidate.median_ttft_us,
                        delta_pct=delta_pct,
                        threshold_pct=ttft_threshold_pct,
                )
            )

        if (
            p95_ttft_threshold_pct is not None
            and baseline.p95_ttft_us
            and candidate.p95_ttft_us is not None
        ):
            delta_pct = (
                (candidate.p95_ttft_us - baseline.p95_ttft_us)
                * 100.0
                / baseline.p95_ttft_us
            )
            if delta_pct > p95_ttft_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="ttft_us_p95",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=baseline.p95_ttft_us,
                        candidate_value=candidate.p95_ttft_us,
                        delta_pct=delta_pct,
                        threshold_pct=p95_ttft_threshold_pct,
                    )
                )

        if (
            host_overhead_threshold_pct is not None
            and baseline.median_host_overhead_pct
            and candidate.median_host_overhead_pct is not None
        ):
            delta_pct = (
                (candidate.median_host_overhead_pct - baseline.median_host_overhead_pct)
                * 100.0
                / baseline.median_host_overhead_pct
            )
            if delta_pct > host_overhead_threshold_pct:
                regressions.append(
                    Regression(
                        key=key,
                        metric="host_overhead_pct",
                        baseline_commit=baseline.commit,
                        candidate_commit=candidate.commit,
                        baseline_value=baseline.median_host_overhead_pct,
                        candidate_value=candidate.median_host_overhead_pct,
                        delta_pct=delta_pct,
                        threshold_pct=host_overhead_threshold_pct,
                    )
                )
    return regressions


def detect_sample_violations(points: list[CommitPoint], minimum_records: int) -> list[SampleViolation]:
    if minimum_records <= 1:
        return []
    violations = []
    for point in points:
        if point.records < minimum_records:
            violations.append(
                SampleViolation(
                    key=point.key,
                    commit=point.commit,
                    records=point.records,
                    minimum_records=minimum_records,
                )
            )
    return violations


def detect_variability_violations(
    points: list[CommitPoint], max_tok_cv_pct: float | None
) -> list[VariabilityViolation]:
    if max_tok_cv_pct is None:
        return []
    violations: list[VariabilityViolation] = []
    for point in points:
        if point.tok_per_s_cv_pct is None:
            continue
        if point.tok_per_s_cv_pct > max_tok_cv_pct:
            violations.append(
                VariabilityViolation(
                    key=point.key,
                    commit=point.commit,
                    records=point.records,
                    tok_per_s_cv_pct=point.tok_per_s_cv_pct,
                    threshold_pct=max_tok_cv_pct,
                )
            )
    return violations


def detect_commit_coverage_violations(
    points: list[CommitPoint], minimum_commits: int
) -> list[CommitCoverageViolation]:
    if minimum_commits <= 1:
        return []

    by_key: dict[str, list[CommitPoint]] = {}
    for point in points:
        by_key.setdefault(point.key, []).append(point)

    violations: list[CommitCoverageViolation] = []
    for key, key_points in sorted(by_key.items()):
        commits = {point.commit for point in key_points}
        if len(commits) >= minimum_commits:
            continue
        latest = sorted(key_points, key=lambda point: (point.latest_timestamp, point.commit))[-1]
        violations.append(
            CommitCoverageViolation(
                key=key,
                commits=len(commits),
                minimum_commits=minimum_commits,
                latest_commit=latest.commit,
            )
        )
    return violations


def detect_comparison_coverage_violations(
    points: list[CommitPoint], baseline_commit: str | None, candidate_commit: str | None
) -> list[ComparisonCoverageViolation]:
    if baseline_commit is None and candidate_commit is None:
        return []

    by_key: dict[str, set[str]] = {}
    for point in points:
        by_key.setdefault(point.key, set()).add(point.commit)

    violations: list[ComparisonCoverageViolation] = []
    for key, commits in sorted(by_key.items()):
        missing = []
        if baseline_commit is not None and baseline_commit not in commits:
            missing.append(f"baseline:{baseline_commit}")
        if candidate_commit is not None and candidate_commit not in commits:
            missing.append(f"candidate:{candidate_commit}")
        if missing:
            violations.append(
                ComparisonCoverageViolation(
                    key=key,
                    baseline_commit=baseline_commit,
                    candidate_commit=candidate_commit,
                    missing_commits=";".join(missing),
                )
            )
    return violations


def detect_prompt_suite_drift(records: list[PerfRecord]) -> list[PromptSuiteDriftViolation]:
    by_key: dict[str, list[PerfRecord]] = {}
    for record in records:
        if record.prompt_suite_sha256:
            by_key.setdefault(record.key, []).append(record)

    violations: list[PromptSuiteDriftViolation] = []
    for key, key_records in sorted(by_key.items()):
        hashes = sorted({record.prompt_suite_sha256 for record in key_records})
        if len(hashes) <= 1:
            continue
        violations.append(
            PromptSuiteDriftViolation(
                key=key,
                hashes=hashes,
                commits=sorted({record.commit for record in key_records}),
                sources=sorted({record.source for record in key_records}),
            )
        )
    return violations


def detect_telemetry_coverage_violations(
    points: list[CommitPoint],
    *,
    require_tok_per_s: bool = False,
    require_wall_tok_per_s: bool = False,
    require_us_per_token: bool = False,
    require_wall_us_per_token: bool = False,
    require_memory: bool = False,
    require_ttft_us: bool = False,
    require_host_overhead_pct: bool = False,
) -> list[TelemetryCoverageViolation]:
    required = [
        ("tok_per_s", require_tok_per_s, "tok_per_s_records"),
        ("wall_tok_per_s", require_wall_tok_per_s, "wall_tok_per_s_records"),
        ("us_per_token", require_us_per_token, "us_per_token_records"),
        ("wall_us_per_token", require_wall_us_per_token, "wall_us_per_token_records"),
        ("memory_bytes", require_memory, "memory_records"),
        ("ttft_us", require_ttft_us, "ttft_us_records"),
        ("host_overhead_pct", require_host_overhead_pct, "host_overhead_records"),
    ]
    violations: list[TelemetryCoverageViolation] = []
    for point in points:
        for metric, enabled, field in required:
            if not enabled:
                continue
            present_records = int(getattr(point, field))
            if present_records == 0:
                violations.append(
                    TelemetryCoverageViolation(
                        key=point.key,
                        commit=point.commit,
                        metric=metric,
                        records=point.records,
                        present_records=present_records,
                    )
                )
    return violations


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Perf Regression Dashboard",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Records: {report['record_count']}",
        f"Regressions: {len(report['regressions'])}",
        f"P05 throughput regressions: {len([row for row in report['regressions'] if row['metric'] == 'tok_per_s_p05'])}",
        f"Token latency regressions: {len([row for row in report['regressions'] if row['metric'] in {'us_per_token', 'wall_us_per_token'}])}",
        f"P95 TTFT regressions: {len([row for row in report['regressions'] if row['metric'] == 'ttft_us_p95'])}",
        f"Host overhead regressions: {len([row for row in report['regressions'] if row['metric'] == 'host_overhead_pct'])}",
        f"Sample violations: {len(report['sample_violations'])}",
        f"Variability violations: {len(report['variability_violations'])}",
        f"Commit coverage violations: {len(report['commit_coverage_violations'])}",
        f"Comparison coverage violations: {len(report['comparison_coverage_violations'])}",
        f"Prompt-suite drift violations: {len(report['prompt_suite_drift_violations'])}",
        f"Telemetry coverage violations: {len(report['telemetry_coverage_violations'])}",
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

    lines.extend(["", "## Sample Coverage", ""])
    if report["sample_violations"]:
        lines.append("| Key | Commit | Records | Minimum Records |")
        lines.append("| --- | --- | ---: | ---: |")
        for violation in report["sample_violations"]:
            lines.append(
                "| {key} | {commit} | {records} | {minimum_records} |".format(**violation)
            )
    else:
        lines.append("Sample coverage requirements satisfied.")

    lines.extend(["", "## Variability", ""])
    if report["variability_violations"]:
        lines.append("| Key | Commit | Records | Tok/s CV | Threshold |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for violation in report["variability_violations"]:
            lines.append(
                "| {key} | {commit} | {records} | {tok_per_s_cv_pct:.2f}% | "
                "{threshold_pct:.2f}% |".format(**violation)
            )
    else:
        lines.append("Variability requirements satisfied.")

    lines.extend(["", "## Commit Coverage", ""])
    if report["commit_coverage_violations"]:
        lines.append("| Key | Commits | Minimum Commits | Latest Commit |")
        lines.append("| --- | ---: | ---: | --- |")
        for violation in report["commit_coverage_violations"]:
            lines.append(
                "| {key} | {commits} | {minimum_commits} | {latest_commit} |".format(
                    **violation
                )
            )
    else:
        lines.append("Commit coverage requirements satisfied.")

    lines.extend(["", "## Comparison Coverage", ""])
    if report["comparison_coverage_violations"]:
        lines.append("| Key | Baseline Commit | Candidate Commit | Missing Commits |")
        lines.append("| --- | --- | --- | --- |")
        for violation in report["comparison_coverage_violations"]:
            baseline = violation["baseline_commit"] or "-"
            candidate = violation["candidate_commit"] or "-"
            lines.append(
                f"| {violation['key']} | {baseline} | {candidate} | "
                f"{violation['missing_commits']} |"
            )
    else:
        lines.append("Explicit comparison commits were present for all benchmark keys.")

    lines.extend(["", "## Prompt Suite Drift", ""])
    if report["prompt_suite_drift_violations"]:
        lines.append("| Key | Hashes | Commits | Sources |")
        lines.append("| --- | ---: | ---: | ---: |")
        for violation in report["prompt_suite_drift_violations"]:
            lines.append(
                f"| {violation['key']} | {len(violation['hashes'])} | "
                f"{len(violation['commits'])} | {len(violation['sources'])} |"
            )
    else:
        lines.append("Prompt-suite hashes are consistent for comparable benchmark keys.")

    lines.extend(["", "## Telemetry Coverage", ""])
    if report["telemetry_coverage_violations"]:
        lines.append("| Key | Commit | Metric | Records | Present Records |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for violation in report["telemetry_coverage_violations"]:
            lines.append(
                "| {key} | {commit} | {metric} | {records} | {present_records} |".format(
                    **violation
                )
            )
    else:
        lines.append("Required telemetry fields are present for every commit point.")

    lines.extend(["", "## Comparisons", ""])
    if report["comparisons"]:
        lines.append(
            "| Key | Baseline | Candidate | Median tok/s Delta | P05 tok/s Delta | Wall tok/s Delta | us/token Delta | Wall us/token Delta | Memory Delta | Median TTFT Delta | P95 TTFT Delta | Host Overhead Delta |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for comparison in report["comparisons"]:
            median_tps_delta = comparison["median_tok_per_s_delta_pct"]
            p05_tps_delta = comparison["p05_tok_per_s_delta_pct"]
            wall_tps_delta = comparison["median_wall_tok_per_s_delta_pct"]
            us_per_token_delta = comparison["median_us_per_token_delta_pct"]
            wall_us_per_token_delta = comparison["median_wall_us_per_token_delta_pct"]
            memory_delta = comparison["max_memory_bytes_delta_pct"]
            ttft_delta = comparison["median_ttft_us_delta_pct"]
            p95_ttft_delta = comparison["p95_ttft_us_delta_pct"]
            overhead_delta = comparison["median_host_overhead_pct_delta_pct"]
            median_tps_cell = f"{median_tps_delta:.2f}%" if median_tps_delta is not None else "-"
            p05_tps_cell = f"{p05_tps_delta:.2f}%" if p05_tps_delta is not None else "-"
            wall_tps_cell = f"{wall_tps_delta:.2f}%" if wall_tps_delta is not None else "-"
            us_per_token_cell = (
                f"{us_per_token_delta:.2f}%" if us_per_token_delta is not None else "-"
            )
            wall_us_per_token_cell = (
                f"{wall_us_per_token_delta:.2f}%"
                if wall_us_per_token_delta is not None
                else "-"
            )
            memory_cell = f"{memory_delta:.2f}%" if memory_delta is not None else "-"
            ttft_cell = f"{ttft_delta:.2f}%" if ttft_delta is not None else "-"
            p95_ttft_cell = f"{p95_ttft_delta:.2f}%" if p95_ttft_delta is not None else "-"
            overhead_cell = f"{overhead_delta:.2f}%" if overhead_delta is not None else "-"
            lines.append(
                f"| {comparison['key']} | {comparison['baseline_commit']} | "
                f"{comparison['candidate_commit']} | {median_tps_cell} | {p05_tps_cell} | "
                f"{wall_tps_cell} | {us_per_token_cell} | {wall_us_per_token_cell} | "
                f"{memory_cell} | {ttft_cell} | {p95_ttft_cell} | {overhead_cell} |"
            )
    else:
        lines.append("No comparable baseline/candidate commit points found.")

    lines.extend(["", "## Commit Points", ""])
    if report["commit_points"]:
        lines.append(
            "| Key | Commit | Records | Tok/s Records | Wall Tok/s Records | us/token Records | Wall us/token Records | Memory Records | TTFT Records | Host Overhead Records | P05 tok/s | Median tok/s | Median wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Median TTFT us | P95 TTFT us | Median Host Overhead % | Tok/s CV | Max Memory Bytes | Prompt Suite |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for point in report["commit_points"]:
            p05_tps = point["p05_tok_per_s"]
            tps = point["median_tok_per_s"]
            wall_tps = point["median_wall_tok_per_s"]
            us_per_token = point["median_us_per_token"]
            p95_us_per_token = point["p95_us_per_token"]
            wall_us_per_token = point["median_wall_us_per_token"]
            p95_wall_us_per_token = point["p95_wall_us_per_token"]
            ttft = point["median_ttft_us"]
            p95_ttft = point["p95_ttft_us"]
            overhead = point["median_host_overhead_pct"]
            tps_cv = point["tok_per_s_cv_pct"]
            memory = point["max_memory_bytes"]
            prompt_suite = point["prompt_suite_sha256"] or "-"
            p05_tps_cell = f"{p05_tps:.3f}" if p05_tps is not None else "-"
            tps_cell = f"{tps:.3f}" if tps is not None else "-"
            wall_tps_cell = f"{wall_tps:.3f}" if wall_tps is not None else "-"
            us_per_token_cell = f"{us_per_token:.3f}" if us_per_token is not None else "-"
            p95_us_per_token_cell = (
                f"{p95_us_per_token:.3f}" if p95_us_per_token is not None else "-"
            )
            wall_us_per_token_cell = (
                f"{wall_us_per_token:.3f}" if wall_us_per_token is not None else "-"
            )
            p95_wall_us_per_token_cell = (
                f"{p95_wall_us_per_token:.3f}" if p95_wall_us_per_token is not None else "-"
            )
            ttft_cell = f"{ttft:.1f}" if ttft is not None else "-"
            p95_ttft_cell = f"{p95_ttft:.1f}" if p95_ttft is not None else "-"
            overhead_cell = f"{overhead:.3f}" if overhead is not None else "-"
            tps_cv_cell = f"{tps_cv:.2f}%" if tps_cv is not None else "-"
            memory_cell = str(memory) if memory is not None else "-"
            lines.append(
                f"| {point['key']} | {point['commit']} | {point['records']} | "
                f"{point['tok_per_s_records']} | {point['wall_tok_per_s_records']} | "
                f"{point['us_per_token_records']} | {point['wall_us_per_token_records']} | "
                f"{point['memory_records']} | {point['ttft_us_records']} | "
                f"{point['host_overhead_records']} | {p05_tps_cell} | {tps_cell} | "
                f"{wall_tps_cell} | {us_per_token_cell} | {p95_us_per_token_cell} | "
                f"{wall_us_per_token_cell} | {p95_wall_us_per_token_cell} | "
                f"{ttft_cell} | {p95_ttft_cell} | {overhead_cell} | "
                f"{tps_cv_cell} | {memory_cell} | {prompt_suite} |"
            )
    else:
        lines.append("No commit-level performance points found.")

    lines.extend(["", "## Latest Summary", ""])
    if report["summaries"]:
        lines.append("| Key | Records | Latest Commit | P05 tok/s | Median tok/s | Median wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Median TTFT us | P95 TTFT us | Median Host Overhead % | Max Memory Bytes |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key, summary in report["summaries"].items():
            p05_tps = summary["p05_tok_per_s"]
            tps = summary["median_tok_per_s"]
            wall_tps = summary["median_wall_tok_per_s"]
            us_per_token = summary["median_us_per_token"]
            p95_us_per_token = summary["p95_us_per_token"]
            wall_us_per_token = summary["median_wall_us_per_token"]
            p95_wall_us_per_token = summary["p95_wall_us_per_token"]
            ttft = summary["median_ttft_us"]
            p95_ttft = summary["p95_ttft_us"]
            overhead = summary["median_host_overhead_pct"]
            memory = summary["max_memory_bytes"]
            p05_tps_cell = f"{p05_tps:.3f}" if p05_tps is not None else "-"
            tps_cell = f"{tps:.3f}" if tps is not None else "-"
            wall_tps_cell = f"{wall_tps:.3f}" if wall_tps is not None else "-"
            us_per_token_cell = f"{us_per_token:.3f}" if us_per_token is not None else "-"
            p95_us_per_token_cell = (
                f"{p95_us_per_token:.3f}" if p95_us_per_token is not None else "-"
            )
            wall_us_per_token_cell = (
                f"{wall_us_per_token:.3f}" if wall_us_per_token is not None else "-"
            )
            p95_wall_us_per_token_cell = (
                f"{p95_wall_us_per_token:.3f}" if p95_wall_us_per_token is not None else "-"
            )
            ttft_cell = f"{ttft:.1f}" if ttft is not None else "-"
            p95_ttft_cell = f"{p95_ttft:.1f}" if p95_ttft is not None else "-"
            overhead_cell = f"{overhead:.3f}" if overhead is not None else "-"
            memory_cell = str(memory) if memory is not None else "-"
            lines.append(
                f"| {key} | {summary['records']} | {summary['latest_commit']} | "
                f"{p05_tps_cell} | {tps_cell} | {wall_tps_cell} | "
                f"{us_per_token_cell} | {p95_us_per_token_cell} | "
                f"{wall_us_per_token_cell} | {p95_wall_us_per_token_cell} | "
                f"{ttft_cell} | {p95_ttft_cell} | {overhead_cell} | {memory_cell} |"
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


def junit_report(report: dict[str, Any]) -> str:
    variability_violations = report.get("variability_violations", [])
    commit_coverage_violations = report.get("commit_coverage_violations", [])
    comparison_coverage_violations = report.get("comparison_coverage_violations", [])
    prompt_suite_drift_violations = report.get("prompt_suite_drift_violations", [])
    telemetry_coverage_violations = report.get("telemetry_coverage_violations", [])
    failures = (
        len(report["regressions"])
        + len(report["sample_violations"])
        + len(variability_violations)
        + len(commit_coverage_violations)
        + len(comparison_coverage_violations)
        + len(prompt_suite_drift_violations)
        + len(telemetry_coverage_violations)
    )
    tests = failures or 1
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_perf_regression",
            "tests": str(tests),
            "failures": str(failures),
            "errors": "0",
            "timestamp": report["generated_at"],
        },
    )

    if failures == 0:
        ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "perf_regression",
                "name": "dashboard_pass",
            },
        )
    else:
        for regression in report["regressions"]:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": "perf_regression.regression",
                    "name": f"{regression['metric']}:{regression['key']}",
                },
            )
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "perf_regression",
                    "message": (
                        f"{regression['metric']} changed {regression['delta_pct']:.2f}% "
                        f"over {regression['threshold_pct']:.2f}% threshold"
                    ),
                },
            )
            failure.text = (
                f"baseline={regression['baseline_commit']} value={regression['baseline_value']:.3f}\n"
                f"candidate={regression['candidate_commit']} value={regression['candidate_value']:.3f}\n"
                f"key={regression['key']}"
            )

        for violation in report["sample_violations"]:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": "perf_regression.sample_coverage",
                    "name": f"{violation['commit']}:{violation['key']}",
                },
            )
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "sample_coverage",
                    "message": (
                        f"{violation['records']} samples below minimum "
                        f"{violation['minimum_records']}"
                    ),
                },
            )
            failure.text = (
                f"commit={violation['commit']}\n"
                f"records={violation['records']}\n"
                f"minimum_records={violation['minimum_records']}\n"
                f"key={violation['key']}"
            )

        for violation in variability_violations:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": "perf_regression.variability",
                    "name": f"{violation['commit']}:{violation['key']}",
                },
            )
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "tok_per_s_variability",
                    "message": (
                        f"tok/s CV {violation['tok_per_s_cv_pct']:.2f}% exceeds "
                        f"{violation['threshold_pct']:.2f}% threshold"
                    ),
                },
            )
            failure.text = (
                f"commit={violation['commit']}\n"
                f"records={violation['records']}\n"
                f"tok_per_s_cv_pct={violation['tok_per_s_cv_pct']:.3f}\n"
                f"threshold_pct={violation['threshold_pct']:.3f}\n"
                f"key={violation['key']}"
            )

        for violation in commit_coverage_violations:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": "perf_regression.commit_coverage",
                    "name": violation["key"],
                },
            )
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "commit_coverage",
                    "message": (
                        f"{violation['commits']} commits below minimum "
                        f"{violation['minimum_commits']}"
                    ),
                },
            )
            failure.text = (
                f"commits={violation['commits']}\n"
                f"minimum_commits={violation['minimum_commits']}\n"
                f"latest_commit={violation['latest_commit']}\n"
                f"key={violation['key']}"
            )

        for violation in comparison_coverage_violations:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": "perf_regression.comparison_coverage",
                    "name": violation["key"],
                },
            )
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "comparison_coverage",
                    "message": f"missing explicit comparison commits: {violation['missing_commits']}",
                },
            )
            baseline = violation["baseline_commit"] or ""
            candidate = violation["candidate_commit"] or ""
            failure.text = (
                f"baseline_commit={baseline}\n"
                f"candidate_commit={candidate}\n"
                f"missing_commits={violation['missing_commits']}\n"
                f"key={violation['key']}"
            )

        for violation in prompt_suite_drift_violations:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": "perf_regression.prompt_suite_drift",
                    "name": violation["key"],
                },
            )
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "prompt_suite_drift",
                    "message": f"{len(violation['hashes'])} prompt-suite hashes for comparable key",
                },
            )
            failure.text = (
                f"hashes={json.dumps(violation['hashes'], separators=(',', ':'))}\n"
                f"commits={json.dumps(violation['commits'], separators=(',', ':'))}\n"
                f"sources={json.dumps(violation['sources'], separators=(',', ':'))}\n"
                f"key={violation['key']}"
            )

        for violation in telemetry_coverage_violations:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": "perf_regression.telemetry_coverage",
                    "name": f"{violation['metric']}:{violation['commit']}:{violation['key']}",
                },
            )
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "telemetry_coverage",
                    "message": f"missing required {violation['metric']} telemetry",
                },
            )
            failure.text = (
                f"commit={violation['commit']}\n"
                f"metric={violation['metric']}\n"
                f"records={violation['records']}\n"
                f"present_records={violation['present_records']}\n"
                f"key={violation['key']}"
            )

    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def write_dashboard_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "perf_regression_latest.json"
    md_path = output_dir / "perf_regression_latest.md"
    junit_path = output_dir / "perf_regression_junit_latest.xml"
    commit_points_csv = output_dir / "perf_regression_commit_points_latest.csv"
    regressions_csv = output_dir / "perf_regression_regressions_latest.csv"
    comparisons_csv = output_dir / "perf_regression_comparisons_latest.csv"
    sample_violations_csv = output_dir / "perf_regression_sample_violations_latest.csv"
    variability_violations_csv = output_dir / "perf_regression_variability_violations_latest.csv"
    commit_coverage_violations_csv = (
        output_dir / "perf_regression_commit_coverage_violations_latest.csv"
    )
    comparison_coverage_violations_csv = (
        output_dir / "perf_regression_comparison_coverage_violations_latest.csv"
    )
    prompt_suite_drift_csv = output_dir / "perf_regression_prompt_suite_drift_latest.csv"
    telemetry_coverage_csv = output_dir / "perf_regression_telemetry_coverage_violations_latest.csv"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    junit_path.write_text(junit_report(report), encoding="utf-8")
    write_csv(
        commit_points_csv,
        report["commit_points"],
        [
            "key",
            "commit",
            "latest_timestamp",
            "records",
            "tok_per_s_records",
            "wall_tok_per_s_records",
            "us_per_token_records",
            "wall_us_per_token_records",
            "memory_records",
            "ttft_us_records",
            "host_overhead_records",
            "p05_tok_per_s",
            "median_tok_per_s",
            "median_wall_tok_per_s",
            "median_us_per_token",
            "p95_us_per_token",
            "median_wall_us_per_token",
            "p95_wall_us_per_token",
            "median_ttft_us",
            "p95_ttft_us",
            "median_host_overhead_pct",
            "tok_per_s_cv_pct",
            "max_memory_bytes",
            "prompt_suite_sha256",
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
    write_csv(
        comparisons_csv,
        report["comparisons"],
        [
            "key",
            "baseline_commit",
            "candidate_commit",
            "baseline_latest_timestamp",
            "candidate_latest_timestamp",
            "baseline_records",
            "candidate_records",
            "median_tok_per_s_baseline",
            "median_tok_per_s_candidate",
            "median_tok_per_s_delta_pct",
            "p05_tok_per_s_baseline",
            "p05_tok_per_s_candidate",
            "p05_tok_per_s_delta_pct",
            "median_wall_tok_per_s_baseline",
            "median_wall_tok_per_s_candidate",
            "median_wall_tok_per_s_delta_pct",
            "median_us_per_token_baseline",
            "median_us_per_token_candidate",
            "median_us_per_token_delta_pct",
            "median_wall_us_per_token_baseline",
            "median_wall_us_per_token_candidate",
            "median_wall_us_per_token_delta_pct",
            "max_memory_bytes_baseline",
            "max_memory_bytes_candidate",
            "max_memory_bytes_delta_pct",
            "median_ttft_us_baseline",
            "median_ttft_us_candidate",
            "median_ttft_us_delta_pct",
            "p95_ttft_us_baseline",
            "p95_ttft_us_candidate",
            "p95_ttft_us_delta_pct",
            "median_host_overhead_pct_baseline",
            "median_host_overhead_pct_candidate",
            "median_host_overhead_pct_delta_pct",
        ],
    )
    write_csv(
        sample_violations_csv,
        report["sample_violations"],
        ["key", "commit", "records", "minimum_records"],
    )
    write_csv(
        variability_violations_csv,
        report["variability_violations"],
        ["key", "commit", "records", "tok_per_s_cv_pct", "threshold_pct"],
    )
    write_csv(
        commit_coverage_violations_csv,
        report["commit_coverage_violations"],
        ["key", "commits", "minimum_commits", "latest_commit"],
    )
    write_csv(
        comparison_coverage_violations_csv,
        report["comparison_coverage_violations"],
        ["key", "baseline_commit", "candidate_commit", "missing_commits"],
    )
    write_csv(
        prompt_suite_drift_csv,
        report["prompt_suite_drift_violations"],
        ["key", "hashes", "commits", "sources"],
    )
    write_csv(
        telemetry_coverage_csv,
        report["telemetry_coverage_violations"],
        ["key", "commit", "metric", "records", "present_records"],
    )

    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_junit={junit_path}")
    print(f"wrote_commit_points_csv={commit_points_csv}")
    print(f"wrote_regressions_csv={regressions_csv}")
    print(f"wrote_comparisons_csv={comparisons_csv}")
    print(f"wrote_sample_violations_csv={sample_violations_csv}")
    print(f"wrote_variability_violations_csv={variability_violations_csv}")
    print(f"wrote_commit_coverage_violations_csv={commit_coverage_violations_csv}")
    print(f"wrote_comparison_coverage_violations_csv={comparison_coverage_violations_csv}")
    print(f"wrote_prompt_suite_drift_csv={prompt_suite_drift_csv}")
    print(f"wrote_telemetry_coverage_csv={telemetry_coverage_csv}")


def build_report(
    records: list[PerfRecord],
    tok_threshold_pct: float,
    memory_threshold_pct: float,
    baseline_commit: str | None = None,
    candidate_commit: str | None = None,
    min_records_per_point: int = 1,
    max_tok_cv_pct: float | None = None,
    wall_tok_threshold_pct: float | None = None,
    ttft_threshold_pct: float | None = None,
    p95_ttft_threshold_pct: float | None = None,
    host_overhead_threshold_pct: float | None = None,
    p05_tok_threshold_pct: float | None = None,
    us_per_token_threshold_pct: float | None = None,
    wall_us_per_token_threshold_pct: float | None = None,
    min_commits_per_key: int = 1,
    require_tok_per_s: bool = False,
    require_wall_tok_per_s: bool = False,
    require_us_per_token: bool = False,
    require_wall_us_per_token: bool = False,
    require_memory: bool = False,
    require_ttft_us: bool = False,
    require_host_overhead_pct: bool = False,
) -> dict[str, Any]:
    points = commit_points(records)
    regressions = detect_regressions(
        records,
        tok_threshold_pct,
        memory_threshold_pct,
        wall_tok_threshold_pct=wall_tok_threshold_pct,
        ttft_threshold_pct=ttft_threshold_pct,
        p95_ttft_threshold_pct=p95_ttft_threshold_pct,
        host_overhead_threshold_pct=host_overhead_threshold_pct,
        p05_tok_threshold_pct=p05_tok_threshold_pct,
        us_per_token_threshold_pct=us_per_token_threshold_pct,
        wall_us_per_token_threshold_pct=wall_us_per_token_threshold_pct,
        baseline_commit=baseline_commit,
        candidate_commit=candidate_commit,
    )
    sample_violations = detect_sample_violations(points, min_records_per_point)
    variability_violations = detect_variability_violations(points, max_tok_cv_pct)
    commit_coverage_violations = detect_commit_coverage_violations(points, min_commits_per_key)
    comparisons = comparison_rows(points, baseline_commit, candidate_commit)
    comparison_coverage_violations = detect_comparison_coverage_violations(
        points, baseline_commit, candidate_commit
    )
    prompt_suite_drift_violations = detect_prompt_suite_drift(records)
    telemetry_coverage_violations = detect_telemetry_coverage_violations(
        points,
        require_tok_per_s=require_tok_per_s,
        require_wall_tok_per_s=require_wall_tok_per_s,
        require_us_per_token=require_us_per_token,
        require_wall_us_per_token=require_wall_us_per_token,
        require_memory=require_memory,
        require_ttft_us=require_ttft_us,
        require_host_overhead_pct=require_host_overhead_pct,
    )
    return {
        "generated_at": iso_now(),
        "record_count": len(records),
        "status": (
            "fail"
            if regressions
            or sample_violations
            or variability_violations
            or commit_coverage_violations
            or comparison_coverage_violations
            or prompt_suite_drift_violations
            or telemetry_coverage_violations
            else "pass"
        ),
        "comparison": {
            "baseline_commit": baseline_commit,
            "candidate_commit": candidate_commit,
            "mode": "explicit" if baseline_commit or candidate_commit else "latest-distinct-commits",
        },
        "thresholds": {
            "tok_regression_pct": tok_threshold_pct,
            "memory_regression_pct": memory_threshold_pct,
            "wall_tok_regression_pct": wall_tok_threshold_pct,
            "ttft_regression_pct": ttft_threshold_pct,
            "p95_ttft_regression_pct": p95_ttft_threshold_pct,
            "host_overhead_regression_pct": host_overhead_threshold_pct,
            "p05_tok_regression_pct": p05_tok_threshold_pct,
            "us_per_token_regression_pct": us_per_token_threshold_pct,
            "wall_us_per_token_regression_pct": wall_us_per_token_threshold_pct,
            "min_records_per_point": min_records_per_point,
            "max_tok_cv_pct": max_tok_cv_pct,
            "min_commits_per_key": min_commits_per_key,
            "require_tok_per_s": require_tok_per_s,
            "require_wall_tok_per_s": require_wall_tok_per_s,
            "require_us_per_token": require_us_per_token,
            "require_wall_us_per_token": require_wall_us_per_token,
            "require_memory": require_memory,
            "require_ttft_us": require_ttft_us,
            "require_host_overhead_pct": require_host_overhead_pct,
        },
        "summaries": summarize(records),
        "commit_points": [asdict(point) for point in points],
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "regressions": [asdict(regression) for regression in regressions],
        "sample_violations": [asdict(violation) for violation in sample_violations],
        "variability_violations": [asdict(violation) for violation in variability_violations],
        "commit_coverage_violations": [
            asdict(violation) for violation in commit_coverage_violations
        ],
        "comparison_coverage_violations": [
            asdict(violation) for violation in comparison_coverage_violations
        ],
        "prompt_suite_drift_violations": [
            asdict(violation) for violation in prompt_suite_drift_violations
        ],
        "telemetry_coverage_violations": [
            asdict(violation) for violation in telemetry_coverage_violations
        ],
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
    parser.add_argument(
        "--wall-tok-regression-pct",
        type=float,
        help="Fail when host wall-clock median tok/s drops by more than this percent",
    )
    parser.add_argument(
        "--ttft-regression-pct",
        type=float,
        help="Fail when median first-token latency increases by more than this percent",
    )
    parser.add_argument(
        "--p95-ttft-regression-pct",
        type=float,
        help="Fail when P95 first-token latency increases by more than this percent",
    )
    parser.add_argument(
        "--host-overhead-regression-pct",
        type=float,
        help="Fail when median QEMU host overhead percentage increases by more than this percent",
    )
    parser.add_argument(
        "--p05-tok-regression-pct",
        type=float,
        help="Fail when P05 guest tok/s drops by more than this percent",
    )
    parser.add_argument(
        "--us-per-token-regression-pct",
        type=float,
        help="Fail when median guest microseconds/token increases by more than this percent",
    )
    parser.add_argument(
        "--wall-us-per-token-regression-pct",
        type=float,
        help="Fail when median host wall-clock microseconds/token increases by more than this percent",
    )
    parser.add_argument("--baseline-commit", help="Commit SHA/name to use as the comparison baseline")
    parser.add_argument("--candidate-commit", help="Commit SHA/name to compare against the baseline")
    parser.add_argument(
        "--min-records-per-point",
        type=int,
        default=1,
        help="Minimum samples required for each benchmark key/commit point before the dashboard passes",
    )
    parser.add_argument(
        "--max-tok-cv-pct",
        type=float,
        help="Fail when a benchmark key/commit point has tok/s coefficient of variation above this percent",
    )
    parser.add_argument(
        "--min-commits-per-key",
        type=int,
        default=1,
        help="Minimum distinct commits required for each benchmark key before the dashboard passes",
    )
    parser.add_argument(
        "--require-tok-per-s",
        action="store_true",
        help="Fail when any benchmark key/commit point lacks guest tok/s telemetry",
    )
    parser.add_argument(
        "--require-wall-tok-per-s",
        action="store_true",
        help="Fail when any benchmark key/commit point lacks host wall-clock tok/s telemetry",
    )
    parser.add_argument(
        "--require-us-per-token",
        action="store_true",
        help="Fail when any benchmark key/commit point lacks guest microseconds/token telemetry",
    )
    parser.add_argument(
        "--require-wall-us-per-token",
        action="store_true",
        help="Fail when any benchmark key/commit point lacks host wall-clock microseconds/token telemetry",
    )
    parser.add_argument(
        "--require-memory",
        action="store_true",
        help="Fail when any benchmark key/commit point lacks memory telemetry",
    )
    parser.add_argument(
        "--require-ttft-us",
        action="store_true",
        help="Fail when any benchmark key/commit point lacks first-token latency telemetry",
    )
    parser.add_argument(
        "--require-host-overhead-pct",
        action="store_true",
        help="Fail when any benchmark key/commit point lacks QEMU host overhead telemetry",
    )
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
        min_records_per_point=args.min_records_per_point,
        max_tok_cv_pct=args.max_tok_cv_pct,
        wall_tok_threshold_pct=args.wall_tok_regression_pct,
        ttft_threshold_pct=args.ttft_regression_pct,
        p95_ttft_threshold_pct=args.p95_ttft_regression_pct,
        host_overhead_threshold_pct=args.host_overhead_regression_pct,
        p05_tok_threshold_pct=args.p05_tok_regression_pct,
        us_per_token_threshold_pct=args.us_per_token_regression_pct,
        wall_us_per_token_threshold_pct=args.wall_us_per_token_regression_pct,
        min_commits_per_key=args.min_commits_per_key,
        require_tok_per_s=args.require_tok_per_s,
        require_wall_tok_per_s=args.require_wall_tok_per_s,
        require_us_per_token=args.require_us_per_token,
        require_wall_us_per_token=args.require_wall_us_per_token,
        require_memory=args.require_memory,
        require_ttft_us=args.require_ttft_us,
        require_host_overhead_pct=args.require_host_overhead_pct,
    )

    write_dashboard_outputs(report, args.output_dir)
    print(f"status={report['status']}")

    if args.fail_on_regression and report["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
