#!/usr/bin/env python3
"""Compare host-side QEMU prompt benchmark reports across builds.

The comparator consumes JSON reports produced by ``qemu_prompt_bench.py`` and
writes normalized build-to-build deltas under ``bench/results``. It is offline
host-side tooling only; it never launches QEMU.
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


RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class BuildMetric:
    build: str
    source: str
    commit: str
    benchmark: str
    profile: str
    model: str
    quantization: str
    prompt: str
    prompt_suite_sha256: str
    runs: int
    ok_runs: int
    median_tokens: float | None
    median_elapsed_us: float | None
    median_tok_per_s: float | None
    p05_tok_per_s: float | None
    median_wall_tok_per_s: float | None
    p05_wall_tok_per_s: float | None
    median_ttft_us: float | None
    median_us_per_token: float | None
    median_wall_us_per_token: float | None
    median_host_child_cpu_us: float | None
    median_host_child_cpu_pct: float | None
    median_host_child_tok_per_cpu_s: float | None
    max_host_child_peak_rss_bytes: int | None
    max_memory_bytes: int | None

    @property
    def key(self) -> str:
        parts = (self.benchmark, self.profile, self.model, self.quantization, self.prompt)
        return "/".join(part or "-" for part in parts)


@dataclass(frozen=True)
class BuildDelta:
    key: str
    baseline_build: str
    candidate_build: str
    baseline_commit: str
    candidate_commit: str
    baseline_prompt_suite_sha256: str
    candidate_prompt_suite_sha256: str
    baseline_tok_per_s: float | None
    candidate_tok_per_s: float | None
    tok_per_s_delta_pct: float | None
    baseline_tok_per_s_p05: float | None
    candidate_tok_per_s_p05: float | None
    tok_per_s_p05_delta_pct: float | None
    baseline_wall_tok_per_s: float | None
    candidate_wall_tok_per_s: float | None
    wall_tok_per_s_delta_pct: float | None
    baseline_wall_tok_per_s_p05: float | None
    candidate_wall_tok_per_s_p05: float | None
    wall_tok_per_s_p05_delta_pct: float | None
    baseline_elapsed_us: float | None
    candidate_elapsed_us: float | None
    elapsed_delta_pct: float | None
    baseline_ttft_us: float | None
    candidate_ttft_us: float | None
    ttft_delta_pct: float | None
    baseline_us_per_token: float | None
    candidate_us_per_token: float | None
    us_per_token_delta_pct: float | None
    baseline_wall_us_per_token: float | None
    candidate_wall_us_per_token: float | None
    wall_us_per_token_delta_pct: float | None
    baseline_host_child_cpu_us: float | None
    candidate_host_child_cpu_us: float | None
    host_child_cpu_us_delta_pct: float | None
    baseline_host_child_cpu_pct: float | None
    candidate_host_child_cpu_pct: float | None
    host_child_cpu_pct_delta_pct: float | None
    baseline_host_child_tok_per_cpu_s: float | None
    candidate_host_child_tok_per_cpu_s: float | None
    host_child_tok_per_cpu_s_delta_pct: float | None
    baseline_host_child_peak_rss_bytes: int | None
    candidate_host_child_peak_rss_bytes: int | None
    host_child_peak_rss_delta_pct: float | None
    baseline_memory_bytes: int | None
    candidate_memory_bytes: int | None
    memory_delta_pct: float | None
    baseline_ok_runs: int
    candidate_ok_runs: int


@dataclass(frozen=True)
class BuildRegression:
    key: str
    candidate_build: str
    metric: str
    delta_pct: float
    allowed_pct: float


@dataclass(frozen=True)
class BuildCoverageViolation:
    key: str
    build: str
    role: str
    ok_runs: int
    minimum_ok_runs: int


@dataclass(frozen=True)
class BuildPromptSuiteDrift:
    key: str
    baseline_build: str
    candidate_build: str
    baseline_prompt_suite_sha256: str
    candidate_prompt_suite_sha256: str


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


def first_float(row: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = parse_float(row.get(name))
        if value is not None:
            return value
    return None


def prompt_suite_sha256(row: dict[str, Any]) -> str:
    suite = row.get("prompt_suite")
    if isinstance(suite, dict):
        value = suite.get("suite_sha256") or suite.get("sha256")
        if value is not None:
            return str(value)
    return first_present(row, ("prompt_suite_sha256", "suite_sha256", "prompt_set_sha256"), "")


def duration_us(row: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = parse_float(row.get(name))
        if value is not None:
            return value
        value = parse_float(row.get(f"{name}_ms"))
        if value is not None:
            return value * 1000.0
        value = parse_float(row.get(f"{name}_s"))
        if value is not None:
            return value * 1_000_000.0
    return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def flatten_payload(payload: Any) -> Iterable[dict[str, Any]]:
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


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(flatten_payload(payload))


def metric_from_rows(build: str, source: Path, rows: list[dict[str, Any]]) -> list[BuildMetric]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        tok_per_s = parse_float(row.get("tok_per_s"))
        tok_per_s_milli = parse_float(row.get("tok_per_s_milli"))
        if tok_per_s is None and tok_per_s_milli is not None:
            row = dict(row)
            row["tok_per_s"] = tok_per_s_milli / 1000.0
        wall_tok_per_s = first_float(
            row,
            ("wall_tok_per_s", "host_tok_per_s", "host_wall_tok_per_s"),
        )
        wall_tok_per_s_milli = first_float(
            row,
            ("wall_tok_per_s_milli", "host_tok_per_s_milli", "host_wall_tok_per_s_milli"),
        )
        if wall_tok_per_s is None and wall_tok_per_s_milli is not None:
            row = dict(row)
            row["wall_tok_per_s"] = wall_tok_per_s_milli / 1000.0

        key_parts = (
            first_present(row, ("benchmark", "bench", "name", "suite"), source.stem),
            first_present(row, ("profile", "mode"), "default"),
            first_present(row, ("model", "model_name"), ""),
            first_present(row, ("quantization", "quant", "format"), ""),
            first_present(row, ("prompt", "prompt_id", "case", "scenario"), ""),
        )
        grouped.setdefault("/".join(part or "-" for part in key_parts), []).append(row)

    metrics: list[BuildMetric] = []
    for key_rows in grouped.values():
        first = key_rows[0]
        token_values = [value for row in key_rows if (value := parse_float(row.get("tokens"))) is not None]
        elapsed_values = [
            value
            for row in key_rows
            if (value := parse_float(row.get("elapsed_us") or row.get("duration_us") or row.get("total_us")))
            is not None
        ]
        tok_values = [value for row in key_rows if (value := parse_float(row.get("tok_per_s"))) is not None]
        wall_tok_values = [
            value
            for row in key_rows
            if (value := first_float(row, ("wall_tok_per_s", "host_tok_per_s", "host_wall_tok_per_s")))
            is not None
        ]
        ttft_values = [
            value
            for row in key_rows
            if (
                value := duration_us(
                    row,
                    (
                        "ttft_us",
                        "ttft",
                        "time_to_first_token_us",
                        "time_to_first_token",
                        "first_token_us",
                        "first_token",
                    ),
                )
            )
            is not None
        ]
        memory_values = [
            value for row in key_rows if (value := parse_int(row.get("memory_bytes"))) is not None and value >= 0
        ]
        us_per_token_values = [
            value
            for row in key_rows
            if (value := first_float(row, ("us_per_token", "us_per_token_median"))) is not None
        ]
        wall_us_per_token_values = [
            value
            for row in key_rows
            if (
                value := first_float(
                    row,
                    (
                        "wall_us_per_token",
                        "wall_us_per_token_median",
                        "host_us_per_token",
                        "host_wall_us_per_token",
                    ),
                )
            )
            is not None
        ]
        host_child_cpu_values = [
            value
            for row in key_rows
            if (value := first_float(row, ("host_child_cpu_us", "host_child_cpu_us_median"))) is not None
        ]
        host_child_cpu_pct_values = [
            value
            for row in key_rows
            if (value := first_float(row, ("host_child_cpu_pct", "host_child_cpu_pct_median"))) is not None
        ]
        host_child_tok_per_cpu_s_values = [
            value
            for row in key_rows
            if (
                value := first_float(
                    row,
                    ("host_child_tok_per_cpu_s", "host_child_tok_per_cpu_s_median"),
                )
            )
            is not None
        ]
        host_child_peak_rss_values = [
            value
            for row in key_rows
            if (
                value := parse_int(
                    row.get("host_child_peak_rss_bytes")
                    if row.get("host_child_peak_rss_bytes") is not None
                    else row.get("host_child_peak_rss_bytes_max")
                )
            )
            is not None
            and value >= 0
        ]
        ok_runs = sum(
            1
            for row in key_rows
            if parse_int(row.get("returncode", 0)) == 0 and str(row.get("timed_out", "false")).lower() != "true"
        )
        metrics.append(
            BuildMetric(
                build=build,
                source=str(source),
                commit=first_present(first, ("commit", "git_commit", "sha"), "unknown"),
                benchmark=first_present(first, ("benchmark", "bench", "name", "suite"), source.stem),
                profile=first_present(first, ("profile", "mode"), "default"),
                model=first_present(first, ("model", "model_name"), ""),
                quantization=first_present(first, ("quantization", "quant", "format"), ""),
                prompt=first_present(first, ("prompt", "prompt_id", "case", "scenario"), ""),
                prompt_suite_sha256=prompt_suite_sha256(first),
                runs=len(key_rows),
                ok_runs=ok_runs,
                median_tokens=statistics.median(token_values) if token_values else None,
                median_elapsed_us=statistics.median(elapsed_values) if elapsed_values else None,
                median_tok_per_s=statistics.median(tok_values) if tok_values else None,
                p05_tok_per_s=percentile(tok_values, 5.0),
                median_wall_tok_per_s=statistics.median(wall_tok_values) if wall_tok_values else None,
                p05_wall_tok_per_s=percentile(wall_tok_values, 5.0),
                median_ttft_us=statistics.median(ttft_values) if ttft_values else None,
                median_us_per_token=statistics.median(us_per_token_values) if us_per_token_values else None,
                median_wall_us_per_token=(
                    statistics.median(wall_us_per_token_values) if wall_us_per_token_values else None
                ),
                median_host_child_cpu_us=(
                    statistics.median(host_child_cpu_values) if host_child_cpu_values else None
                ),
                median_host_child_cpu_pct=(
                    statistics.median(host_child_cpu_pct_values) if host_child_cpu_pct_values else None
                ),
                median_host_child_tok_per_cpu_s=(
                    statistics.median(host_child_tok_per_cpu_s_values)
                    if host_child_tok_per_cpu_s_values
                    else None
                ),
                max_host_child_peak_rss_bytes=(
                    max(host_child_peak_rss_values) if host_child_peak_rss_values else None
                ),
                max_memory_bytes=max(memory_values) if memory_values else None,
            )
        )
    return sorted(metrics, key=lambda metric: metric.key)


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        build, path = spec.split("=", 1)
        build = build.strip()
        if not build:
            raise ValueError(f"empty build name in input spec: {spec!r}")
        return build, Path(path)
    path = Path(spec)
    return path.stem, path


def load_build_metrics(specs: list[str]) -> list[BuildMetric]:
    metrics: list[BuildMetric] = []
    for spec in specs:
        build, path = parse_input_spec(spec)
        rows = load_rows(path)
        metrics.extend(metric_from_rows(build, path, rows))
    return metrics


def pct_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (candidate - baseline) * 100.0 / baseline


def compare_builds(metrics: list[BuildMetric], baseline_build: str) -> list[BuildDelta]:
    by_build_key = {(metric.build, metric.key): metric for metric in metrics}
    baseline_metrics = [metric for metric in metrics if metric.build == baseline_build]
    candidate_builds = sorted({metric.build for metric in metrics if metric.build != baseline_build})

    deltas: list[BuildDelta] = []
    for baseline in baseline_metrics:
        for candidate_build in candidate_builds:
            candidate = by_build_key.get((candidate_build, baseline.key))
            if candidate is None:
                continue
            deltas.append(
                BuildDelta(
                    key=baseline.key,
                    baseline_build=baseline.build,
                    candidate_build=candidate.build,
                    baseline_commit=baseline.commit,
                    candidate_commit=candidate.commit,
                    baseline_prompt_suite_sha256=baseline.prompt_suite_sha256,
                    candidate_prompt_suite_sha256=candidate.prompt_suite_sha256,
                    baseline_tok_per_s=baseline.median_tok_per_s,
                    candidate_tok_per_s=candidate.median_tok_per_s,
                    tok_per_s_delta_pct=pct_delta(candidate.median_tok_per_s, baseline.median_tok_per_s),
                    baseline_tok_per_s_p05=baseline.p05_tok_per_s,
                    candidate_tok_per_s_p05=candidate.p05_tok_per_s,
                    tok_per_s_p05_delta_pct=pct_delta(candidate.p05_tok_per_s, baseline.p05_tok_per_s),
                    baseline_wall_tok_per_s=baseline.median_wall_tok_per_s,
                    candidate_wall_tok_per_s=candidate.median_wall_tok_per_s,
                    wall_tok_per_s_delta_pct=pct_delta(
                        candidate.median_wall_tok_per_s,
                        baseline.median_wall_tok_per_s,
                    ),
                    baseline_wall_tok_per_s_p05=baseline.p05_wall_tok_per_s,
                    candidate_wall_tok_per_s_p05=candidate.p05_wall_tok_per_s,
                    wall_tok_per_s_p05_delta_pct=pct_delta(
                        candidate.p05_wall_tok_per_s,
                        baseline.p05_wall_tok_per_s,
                    ),
                    baseline_elapsed_us=baseline.median_elapsed_us,
                    candidate_elapsed_us=candidate.median_elapsed_us,
                    elapsed_delta_pct=pct_delta(candidate.median_elapsed_us, baseline.median_elapsed_us),
                    baseline_ttft_us=baseline.median_ttft_us,
                    candidate_ttft_us=candidate.median_ttft_us,
                    ttft_delta_pct=pct_delta(candidate.median_ttft_us, baseline.median_ttft_us),
                    baseline_us_per_token=baseline.median_us_per_token,
                    candidate_us_per_token=candidate.median_us_per_token,
                    us_per_token_delta_pct=pct_delta(
                        candidate.median_us_per_token,
                        baseline.median_us_per_token,
                    ),
                    baseline_wall_us_per_token=baseline.median_wall_us_per_token,
                    candidate_wall_us_per_token=candidate.median_wall_us_per_token,
                    wall_us_per_token_delta_pct=pct_delta(
                        candidate.median_wall_us_per_token,
                        baseline.median_wall_us_per_token,
                    ),
                    baseline_host_child_cpu_us=baseline.median_host_child_cpu_us,
                    candidate_host_child_cpu_us=candidate.median_host_child_cpu_us,
                    host_child_cpu_us_delta_pct=pct_delta(
                        candidate.median_host_child_cpu_us,
                        baseline.median_host_child_cpu_us,
                    ),
                    baseline_host_child_cpu_pct=baseline.median_host_child_cpu_pct,
                    candidate_host_child_cpu_pct=candidate.median_host_child_cpu_pct,
                    host_child_cpu_pct_delta_pct=pct_delta(
                        candidate.median_host_child_cpu_pct,
                        baseline.median_host_child_cpu_pct,
                    ),
                    baseline_host_child_tok_per_cpu_s=baseline.median_host_child_tok_per_cpu_s,
                    candidate_host_child_tok_per_cpu_s=candidate.median_host_child_tok_per_cpu_s,
                    host_child_tok_per_cpu_s_delta_pct=pct_delta(
                        candidate.median_host_child_tok_per_cpu_s,
                        baseline.median_host_child_tok_per_cpu_s,
                    ),
                    baseline_host_child_peak_rss_bytes=baseline.max_host_child_peak_rss_bytes,
                    candidate_host_child_peak_rss_bytes=candidate.max_host_child_peak_rss_bytes,
                    host_child_peak_rss_delta_pct=pct_delta(
                        candidate.max_host_child_peak_rss_bytes,
                        baseline.max_host_child_peak_rss_bytes,
                    ),
                    baseline_memory_bytes=baseline.max_memory_bytes,
                    candidate_memory_bytes=candidate.max_memory_bytes,
                    memory_delta_pct=pct_delta(candidate.max_memory_bytes, baseline.max_memory_bytes),
                    baseline_ok_runs=baseline.ok_runs,
                    candidate_ok_runs=candidate.ok_runs,
                )
            )
    return sorted(deltas, key=lambda delta: (delta.candidate_build, delta.key))


def find_regressions(
    deltas: list[BuildDelta],
    max_tok_regression_pct: float,
    max_memory_growth_pct: float | None = None,
    *,
    max_wall_tok_regression_pct: float | None = None,
    max_ttft_growth_pct: float | None = None,
    max_p05_tok_regression_pct: float | None = None,
    max_p05_wall_tok_regression_pct: float | None = None,
    max_us_per_token_growth_pct: float | None = None,
    max_wall_us_per_token_growth_pct: float | None = None,
    max_host_child_cpu_growth_pct: float | None = None,
    max_host_child_cpu_pct_growth_pct: float | None = None,
    max_host_child_tok_per_cpu_s_regression_pct: float | None = None,
    max_host_child_rss_growth_pct: float | None = None,
) -> list[BuildRegression]:
    threshold = -abs(max_tok_regression_pct)
    wall_threshold = -abs(max_wall_tok_regression_pct) if max_wall_tok_regression_pct is not None else None
    p05_threshold = -abs(max_p05_tok_regression_pct) if max_p05_tok_regression_pct is not None else None
    p05_wall_threshold = (
        -abs(max_p05_wall_tok_regression_pct) if max_p05_wall_tok_regression_pct is not None else None
    )
    regressions: list[BuildRegression] = []
    for delta in deltas:
        if delta.tok_per_s_delta_pct is not None and delta.tok_per_s_delta_pct <= threshold:
            regressions.append(
                BuildRegression(
                    key=delta.key,
                    candidate_build=delta.candidate_build,
                    metric="tok_per_s",
                    delta_pct=delta.tok_per_s_delta_pct,
                    allowed_pct=abs(max_tok_regression_pct),
                )
            )
        if (
            p05_threshold is not None
            and delta.tok_per_s_p05_delta_pct is not None
            and delta.tok_per_s_p05_delta_pct <= p05_threshold
        ):
            regressions.append(
                BuildRegression(
                    key=delta.key,
                    candidate_build=delta.candidate_build,
                    metric="tok_per_s_p05",
                    delta_pct=delta.tok_per_s_p05_delta_pct,
                    allowed_pct=abs(max_p05_tok_regression_pct),
                )
            )
        if (
            wall_threshold is not None
            and delta.wall_tok_per_s_delta_pct is not None
            and delta.wall_tok_per_s_delta_pct <= wall_threshold
        ):
            regressions.append(
                BuildRegression(
                    key=delta.key,
                    candidate_build=delta.candidate_build,
                    metric="wall_tok_per_s",
                    delta_pct=delta.wall_tok_per_s_delta_pct,
                    allowed_pct=abs(max_wall_tok_regression_pct),
                )
            )
        if (
            p05_wall_threshold is not None
            and delta.wall_tok_per_s_p05_delta_pct is not None
            and delta.wall_tok_per_s_p05_delta_pct <= p05_wall_threshold
        ):
            regressions.append(
                BuildRegression(
                    key=delta.key,
                    candidate_build=delta.candidate_build,
                    metric="wall_tok_per_s_p05",
                    delta_pct=delta.wall_tok_per_s_p05_delta_pct,
                    allowed_pct=abs(max_p05_wall_tok_regression_pct),
                )
            )
        if (
            max_memory_growth_pct is not None
            and delta.memory_delta_pct is not None
            and delta.memory_delta_pct >= abs(max_memory_growth_pct)
        ):
            regressions.append(
                BuildRegression(
                    key=delta.key,
                    candidate_build=delta.candidate_build,
                    metric="memory_bytes",
                    delta_pct=delta.memory_delta_pct,
                    allowed_pct=abs(max_memory_growth_pct),
                )
            )
        if (
            max_ttft_growth_pct is not None
            and delta.ttft_delta_pct is not None
            and delta.ttft_delta_pct >= abs(max_ttft_growth_pct)
        ):
            regressions.append(
                BuildRegression(
                    key=delta.key,
                    candidate_build=delta.candidate_build,
                    metric="ttft_us",
                    delta_pct=delta.ttft_delta_pct,
                    allowed_pct=abs(max_ttft_growth_pct),
                )
            )
        growth_checks = (
            ("us_per_token", delta.us_per_token_delta_pct, max_us_per_token_growth_pct),
            ("wall_us_per_token", delta.wall_us_per_token_delta_pct, max_wall_us_per_token_growth_pct),
            ("host_child_cpu_us", delta.host_child_cpu_us_delta_pct, max_host_child_cpu_growth_pct),
            ("host_child_cpu_pct", delta.host_child_cpu_pct_delta_pct, max_host_child_cpu_pct_growth_pct),
            ("host_child_peak_rss_bytes", delta.host_child_peak_rss_delta_pct, max_host_child_rss_growth_pct),
        )
        for metric, value, limit in growth_checks:
            if limit is not None and value is not None and value >= abs(limit):
                regressions.append(
                    BuildRegression(
                        key=delta.key,
                        candidate_build=delta.candidate_build,
                        metric=metric,
                        delta_pct=value,
                        allowed_pct=abs(limit),
                    )
                )
        if (
            max_host_child_tok_per_cpu_s_regression_pct is not None
            and delta.host_child_tok_per_cpu_s_delta_pct is not None
            and delta.host_child_tok_per_cpu_s_delta_pct
            <= -abs(max_host_child_tok_per_cpu_s_regression_pct)
        ):
            regressions.append(
                BuildRegression(
                    key=delta.key,
                    candidate_build=delta.candidate_build,
                    metric="host_child_tok_per_cpu_s",
                    delta_pct=delta.host_child_tok_per_cpu_s_delta_pct,
                    allowed_pct=abs(max_host_child_tok_per_cpu_s_regression_pct),
                )
            )
    return regressions


def throughput_regressions(deltas: list[BuildDelta], max_tok_regression_pct: float) -> list[BuildRegression]:
    return [
        regression
        for regression in find_regressions(deltas, max_tok_regression_pct)
        if regression.metric == "tok_per_s"
    ]


def find_coverage_violations(deltas: list[BuildDelta], minimum_ok_runs: int) -> list[BuildCoverageViolation]:
    if minimum_ok_runs <= 0:
        return []
    violations: list[BuildCoverageViolation] = []
    seen: set[tuple[str, str, str]] = set()
    for delta in deltas:
        checks = (
            (delta.baseline_build, "baseline", delta.baseline_ok_runs),
            (delta.candidate_build, "candidate", delta.candidate_ok_runs),
        )
        for build, role, ok_runs in checks:
            dedupe_key = (delta.key, build, role)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            if ok_runs < minimum_ok_runs:
                violations.append(
                    BuildCoverageViolation(
                        key=delta.key,
                        build=build,
                        role=role,
                        ok_runs=ok_runs,
                        minimum_ok_runs=minimum_ok_runs,
                    )
                )
    return violations


def find_prompt_suite_drift(deltas: list[BuildDelta]) -> list[BuildPromptSuiteDrift]:
    drift: list[BuildPromptSuiteDrift] = []
    for delta in deltas:
        if not delta.baseline_prompt_suite_sha256 or not delta.candidate_prompt_suite_sha256:
            continue
        if delta.baseline_prompt_suite_sha256 == delta.candidate_prompt_suite_sha256:
            continue
        drift.append(
            BuildPromptSuiteDrift(
                key=delta.key,
                baseline_build=delta.baseline_build,
                candidate_build=delta.candidate_build,
                baseline_prompt_suite_sha256=delta.baseline_prompt_suite_sha256,
                candidate_prompt_suite_sha256=delta.candidate_prompt_suite_sha256,
            )
        )
    return drift


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Build Benchmark Compare",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Baseline: {report['baseline_build']}",
        f"Builds: {', '.join(report['builds'])}",
        f"Throughput regressions: {len([row for row in report['regressions'] if row['metric'] == 'tok_per_s'])}",
        f"P05 throughput regressions: {len([row for row in report['regressions'] if row['metric'] == 'tok_per_s_p05'])}",
        f"Wall throughput regressions: {len([row for row in report['regressions'] if row['metric'] == 'wall_tok_per_s'])}",
        f"P05 wall throughput regressions: {len([row for row in report['regressions'] if row['metric'] == 'wall_tok_per_s_p05'])}",
        f"TTFT regressions: {len([row for row in report['regressions'] if row['metric'] == 'ttft_us'])}",
        f"Token latency regressions: {len([row for row in report['regressions'] if row['metric'] in {'us_per_token', 'wall_us_per_token'}])}",
        f"Host child CPU/RSS regressions: {len([row for row in report['regressions'] if row['metric'] in {'host_child_cpu_us', 'host_child_cpu_pct', 'host_child_tok_per_cpu_s', 'host_child_peak_rss_bytes'}])}",
        f"Memory regressions: {len([row for row in report['regressions'] if row['metric'] == 'memory_bytes'])}",
        f"Coverage violations: {len(report['coverage_violations'])}",
        f"Prompt-suite drift: {len(report['prompt_suite_drift'])}",
        "",
        "## Deltas",
        "",
    ]
    if report["deltas"]:
        lines.append(
            "| Candidate | Prompt key | Base tok/s | Candidate tok/s | Tok/s delta % | Base P05 tok/s | Candidate P05 tok/s | P05 tok/s delta % | Base wall tok/s | Candidate wall tok/s | Wall tok/s delta % | Base P05 wall tok/s | Candidate P05 wall tok/s | P05 wall tok/s delta % | Base elapsed us | Candidate elapsed us | Elapsed delta % | Base TTFT us | Candidate TTFT us | TTFT delta % | Base us/token | Candidate us/token | us/token delta % | Base wall us/token | Candidate wall us/token | Wall us/token delta % | Base child CPU us | Candidate child CPU us | Child CPU us delta % | Base child CPU % | Candidate child CPU % | Child CPU % delta % | Base child tok/CPU s | Candidate child tok/CPU s | Child tok/CPU s delta % | Base child RSS bytes | Candidate child RSS bytes | Child RSS delta % | Base memory bytes | Candidate memory bytes | Memory delta % |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for delta in report["deltas"]:
            lines.append(
                "| {candidate_build} | {key} | {baseline_tok_per_s} | {candidate_tok_per_s} | "
                "{tok_per_s_delta_pct} | {baseline_tok_per_s_p05} | {candidate_tok_per_s_p05} | "
                "{tok_per_s_p05_delta_pct} | {baseline_wall_tok_per_s} | {candidate_wall_tok_per_s} | "
                "{wall_tok_per_s_delta_pct} | {baseline_wall_tok_per_s_p05} | {candidate_wall_tok_per_s_p05} | "
                "{wall_tok_per_s_p05_delta_pct} | {baseline_elapsed_us} | {candidate_elapsed_us} | {elapsed_delta_pct} | "
                "{baseline_ttft_us} | {candidate_ttft_us} | {ttft_delta_pct} | "
                "{baseline_us_per_token} | {candidate_us_per_token} | {us_per_token_delta_pct} | "
                "{baseline_wall_us_per_token} | {candidate_wall_us_per_token} | {wall_us_per_token_delta_pct} | "
                "{baseline_host_child_cpu_us} | {candidate_host_child_cpu_us} | {host_child_cpu_us_delta_pct} | "
                "{baseline_host_child_cpu_pct} | {candidate_host_child_cpu_pct} | {host_child_cpu_pct_delta_pct} | "
                "{baseline_host_child_tok_per_cpu_s} | {candidate_host_child_tok_per_cpu_s} | {host_child_tok_per_cpu_s_delta_pct} | "
                "{baseline_host_child_peak_rss_bytes} | {candidate_host_child_peak_rss_bytes} | {host_child_peak_rss_delta_pct} | "
                "{baseline_memory_bytes} | {candidate_memory_bytes} | {memory_delta_pct} |".format(
                    **{key: format_value(value) for key, value in delta.items()}
                )
            )
    else:
        lines.append("No comparable prompt metrics found.")
    if report["prompt_suite_drift"]:
        lines.extend(["", "## Prompt-Suite Drift", ""])
        lines.append("| Candidate | Prompt key | Base suite | Candidate suite |")
        lines.append("| --- | --- | --- | --- |")
        for drift in report["prompt_suite_drift"]:
            lines.append(
                "| {candidate_build} | {key} | {baseline_prompt_suite_sha256} | {candidate_prompt_suite_sha256} |".format(
                    **drift
                )
            )
    if report["coverage_violations"]:
        lines.extend(["", "## Coverage Violations", ""])
        lines.append("| Build | Role | Prompt key | OK runs | Minimum OK runs |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for violation in report["coverage_violations"]:
            lines.append(
                "| {build} | {role} | {key} | {ok_runs} | {minimum_ok_runs} |".format(
                    **violation
                )
            )
    return "\n".join(lines) + "\n"


def write_csv(deltas: list[BuildDelta], path: Path) -> None:
    fields = [
        "key",
        "baseline_build",
        "candidate_build",
        "baseline_commit",
        "candidate_commit",
        "baseline_prompt_suite_sha256",
        "candidate_prompt_suite_sha256",
        "baseline_tok_per_s",
        "candidate_tok_per_s",
        "tok_per_s_delta_pct",
        "baseline_tok_per_s_p05",
        "candidate_tok_per_s_p05",
        "tok_per_s_p05_delta_pct",
        "baseline_wall_tok_per_s",
        "candidate_wall_tok_per_s",
        "wall_tok_per_s_delta_pct",
        "baseline_wall_tok_per_s_p05",
        "candidate_wall_tok_per_s_p05",
        "wall_tok_per_s_p05_delta_pct",
        "baseline_elapsed_us",
        "candidate_elapsed_us",
        "elapsed_delta_pct",
        "baseline_ttft_us",
        "candidate_ttft_us",
        "ttft_delta_pct",
        "baseline_us_per_token",
        "candidate_us_per_token",
        "us_per_token_delta_pct",
        "baseline_wall_us_per_token",
        "candidate_wall_us_per_token",
        "wall_us_per_token_delta_pct",
        "baseline_host_child_cpu_us",
        "candidate_host_child_cpu_us",
        "host_child_cpu_us_delta_pct",
        "baseline_host_child_cpu_pct",
        "candidate_host_child_cpu_pct",
        "host_child_cpu_pct_delta_pct",
        "baseline_host_child_tok_per_cpu_s",
        "candidate_host_child_tok_per_cpu_s",
        "host_child_tok_per_cpu_s_delta_pct",
        "baseline_host_child_peak_rss_bytes",
        "candidate_host_child_peak_rss_bytes",
        "host_child_peak_rss_delta_pct",
        "baseline_memory_bytes",
        "candidate_memory_bytes",
        "memory_delta_pct",
        "baseline_ok_runs",
        "candidate_ok_runs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for delta in deltas:
            writer.writerow({field: getattr(delta, field) for field in fields})


def write_coverage_csv(violations: list[BuildCoverageViolation], path: Path) -> None:
    fields = ["key", "build", "role", "ok_runs", "minimum_ok_runs"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for violation in violations:
            writer.writerow({field: getattr(violation, field) for field in fields})


def write_prompt_suite_drift_csv(violations: list[BuildPromptSuiteDrift], path: Path) -> None:
    fields = [
        "key",
        "baseline_build",
        "candidate_build",
        "baseline_prompt_suite_sha256",
        "candidate_prompt_suite_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for violation in violations:
            writer.writerow({field: getattr(violation, field) for field in fields})


def write_junit(
    deltas: list[BuildDelta],
    regressions: list[BuildRegression],
    coverage_violations: list[BuildCoverageViolation],
    prompt_suite_drift: list[BuildPromptSuiteDrift],
    path: Path,
) -> None:
    regression_by_key: dict[tuple[str, str], list[BuildRegression]] = {}
    for regression in regressions:
        regression_by_key.setdefault((regression.candidate_build, regression.key), []).append(regression)
    coverage_by_key: dict[tuple[str, str], list[BuildCoverageViolation]] = {}
    for violation in coverage_violations:
        coverage_by_key.setdefault((violation.build, violation.key), []).append(violation)
    prompt_suite_drift_by_key = {
        (violation.candidate_build, violation.key): violation for violation in prompt_suite_drift
    }
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_build_compare",
            "tests": str(len(deltas)),
            "failures": str(len(regressions) + len(coverage_violations) + len(prompt_suite_drift)),
            "errors": "0",
        },
    )
    for delta in deltas:
        case_name = f"{delta.candidate_build}:{delta.key}"
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "build_compare",
                "name": case_name,
            },
        )
        case_regressions = regression_by_key.get((delta.candidate_build, delta.key), [])
        if case_regressions:
            message = "; ".join(
                f"{regression.metric} changed by {regression.delta_pct:.3f}% "
                f"with allowed {regression.allowed_pct:.3f}%"
                for regression in case_regressions
            )
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "benchmark_regression",
                    "message": message,
                },
            )
            failure.text = (
                f"candidate={delta.candidate_build}\n"
                f"key={delta.key}\n"
                f"baseline_tok_per_s={format_value(delta.baseline_tok_per_s)}\n"
                f"candidate_tok_per_s={format_value(delta.candidate_tok_per_s)}\n"
                f"tok_per_s_delta_pct={format_value(delta.tok_per_s_delta_pct)}\n"
                f"baseline_tok_per_s_p05={format_value(delta.baseline_tok_per_s_p05)}\n"
                f"candidate_tok_per_s_p05={format_value(delta.candidate_tok_per_s_p05)}\n"
                f"tok_per_s_p05_delta_pct={format_value(delta.tok_per_s_p05_delta_pct)}\n"
                f"baseline_wall_tok_per_s={format_value(delta.baseline_wall_tok_per_s)}\n"
                f"candidate_wall_tok_per_s={format_value(delta.candidate_wall_tok_per_s)}\n"
                f"wall_tok_per_s_delta_pct={format_value(delta.wall_tok_per_s_delta_pct)}\n"
                f"baseline_wall_tok_per_s_p05={format_value(delta.baseline_wall_tok_per_s_p05)}\n"
                f"candidate_wall_tok_per_s_p05={format_value(delta.candidate_wall_tok_per_s_p05)}\n"
                f"wall_tok_per_s_p05_delta_pct={format_value(delta.wall_tok_per_s_p05_delta_pct)}\n"
                f"baseline_ttft_us={format_value(delta.baseline_ttft_us)}\n"
                f"candidate_ttft_us={format_value(delta.candidate_ttft_us)}\n"
                f"ttft_delta_pct={format_value(delta.ttft_delta_pct)}\n"
                f"baseline_us_per_token={format_value(delta.baseline_us_per_token)}\n"
                f"candidate_us_per_token={format_value(delta.candidate_us_per_token)}\n"
                f"us_per_token_delta_pct={format_value(delta.us_per_token_delta_pct)}\n"
                f"baseline_wall_us_per_token={format_value(delta.baseline_wall_us_per_token)}\n"
                f"candidate_wall_us_per_token={format_value(delta.candidate_wall_us_per_token)}\n"
                f"wall_us_per_token_delta_pct={format_value(delta.wall_us_per_token_delta_pct)}\n"
                f"baseline_host_child_cpu_us={format_value(delta.baseline_host_child_cpu_us)}\n"
                f"candidate_host_child_cpu_us={format_value(delta.candidate_host_child_cpu_us)}\n"
                f"host_child_cpu_us_delta_pct={format_value(delta.host_child_cpu_us_delta_pct)}\n"
                f"baseline_host_child_cpu_pct={format_value(delta.baseline_host_child_cpu_pct)}\n"
                f"candidate_host_child_cpu_pct={format_value(delta.candidate_host_child_cpu_pct)}\n"
                f"host_child_cpu_pct_delta_pct={format_value(delta.host_child_cpu_pct_delta_pct)}\n"
                f"baseline_host_child_tok_per_cpu_s={format_value(delta.baseline_host_child_tok_per_cpu_s)}\n"
                f"candidate_host_child_tok_per_cpu_s={format_value(delta.candidate_host_child_tok_per_cpu_s)}\n"
                f"host_child_tok_per_cpu_s_delta_pct={format_value(delta.host_child_tok_per_cpu_s_delta_pct)}\n"
                f"baseline_host_child_peak_rss_bytes={format_value(delta.baseline_host_child_peak_rss_bytes)}\n"
                f"candidate_host_child_peak_rss_bytes={format_value(delta.candidate_host_child_peak_rss_bytes)}\n"
                f"host_child_peak_rss_delta_pct={format_value(delta.host_child_peak_rss_delta_pct)}\n"
                f"baseline_memory_bytes={format_value(delta.baseline_memory_bytes)}\n"
                f"candidate_memory_bytes={format_value(delta.candidate_memory_bytes)}\n"
                f"memory_delta_pct={format_value(delta.memory_delta_pct)}\n"
                f"baseline_commit={delta.baseline_commit}\n"
                f"candidate_commit={delta.candidate_commit}\n"
            )
        drift = prompt_suite_drift_by_key.get((delta.candidate_build, delta.key))
        if drift is not None:
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "build_compare_prompt_suite_drift",
                    "message": "baseline and candidate prompt-suite hashes differ",
                },
            )
            failure.text = (
                f"candidate={delta.candidate_build}\n"
                f"key={delta.key}\n"
                f"baseline_prompt_suite_sha256={drift.baseline_prompt_suite_sha256}\n"
                f"candidate_prompt_suite_sha256={drift.candidate_prompt_suite_sha256}\n"
            )
        case_coverage = coverage_by_key.get((delta.baseline_build, delta.key), []) + coverage_by_key.get(
            (delta.candidate_build, delta.key),
            [],
        )
        for violation in case_coverage:
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "build_compare_sample_coverage",
                    "message": (
                        f"{violation.build} {violation.role} has {violation.ok_runs} OK runs; "
                        f"minimum is {violation.minimum_ok_runs}"
                    ),
                },
            )
            failure.text = (
                f"build={violation.build}\n"
                f"role={violation.role}\n"
                f"key={violation.key}\n"
                f"ok_runs={violation.ok_runs}\n"
                f"minimum_ok_runs={violation.minimum_ok_runs}\n"
            )
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def write_report(
    metrics: list[BuildMetric],
    deltas: list[BuildDelta],
    baseline_build: str,
    output_dir: Path,
    *,
    max_tok_regression_pct: float,
    max_wall_tok_regression_pct: float | None = None,
    max_memory_growth_pct: float | None = None,
    max_ttft_growth_pct: float | None = None,
    max_p05_tok_regression_pct: float | None = None,
    max_p05_wall_tok_regression_pct: float | None = None,
    max_us_per_token_growth_pct: float | None = None,
    max_wall_us_per_token_growth_pct: float | None = None,
    max_host_child_cpu_growth_pct: float | None = None,
    max_host_child_cpu_pct_growth_pct: float | None = None,
    max_host_child_tok_per_cpu_s_regression_pct: float | None = None,
    max_host_child_rss_growth_pct: float | None = None,
    min_ok_runs_per_build: int = 0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    regressions = find_regressions(
        deltas,
        max_tok_regression_pct,
        max_memory_growth_pct,
        max_wall_tok_regression_pct=max_wall_tok_regression_pct,
        max_ttft_growth_pct=max_ttft_growth_pct,
        max_p05_tok_regression_pct=max_p05_tok_regression_pct,
        max_p05_wall_tok_regression_pct=max_p05_wall_tok_regression_pct,
        max_us_per_token_growth_pct=max_us_per_token_growth_pct,
        max_wall_us_per_token_growth_pct=max_wall_us_per_token_growth_pct,
        max_host_child_cpu_growth_pct=max_host_child_cpu_growth_pct,
        max_host_child_cpu_pct_growth_pct=max_host_child_cpu_pct_growth_pct,
        max_host_child_tok_per_cpu_s_regression_pct=max_host_child_tok_per_cpu_s_regression_pct,
        max_host_child_rss_growth_pct=max_host_child_rss_growth_pct,
    )
    coverage_violations = find_coverage_violations(deltas, min_ok_runs_per_build)
    prompt_suite_drift = find_prompt_suite_drift(deltas)
    report = {
        "generated_at": iso_now(),
        "status": "fail" if regressions or coverage_violations or prompt_suite_drift else "pass",
        "baseline_build": baseline_build,
        "builds": sorted({metric.build for metric in metrics}),
        "max_tok_regression_pct": abs(max_tok_regression_pct),
        "max_wall_tok_regression_pct": abs(max_wall_tok_regression_pct)
        if max_wall_tok_regression_pct is not None
        else None,
        "max_p05_tok_regression_pct": abs(max_p05_tok_regression_pct)
        if max_p05_tok_regression_pct is not None
        else None,
        "max_p05_wall_tok_regression_pct": abs(max_p05_wall_tok_regression_pct)
        if max_p05_wall_tok_regression_pct is not None
        else None,
        "max_memory_growth_pct": abs(max_memory_growth_pct) if max_memory_growth_pct is not None else None,
        "max_ttft_growth_pct": abs(max_ttft_growth_pct) if max_ttft_growth_pct is not None else None,
        "max_us_per_token_growth_pct": abs(max_us_per_token_growth_pct)
        if max_us_per_token_growth_pct is not None
        else None,
        "max_wall_us_per_token_growth_pct": abs(max_wall_us_per_token_growth_pct)
        if max_wall_us_per_token_growth_pct is not None
        else None,
        "max_host_child_cpu_growth_pct": abs(max_host_child_cpu_growth_pct)
        if max_host_child_cpu_growth_pct is not None
        else None,
        "max_host_child_cpu_pct_growth_pct": abs(max_host_child_cpu_pct_growth_pct)
        if max_host_child_cpu_pct_growth_pct is not None
        else None,
        "max_host_child_tok_per_cpu_s_regression_pct": abs(max_host_child_tok_per_cpu_s_regression_pct)
        if max_host_child_tok_per_cpu_s_regression_pct is not None
        else None,
        "max_host_child_rss_growth_pct": abs(max_host_child_rss_growth_pct)
        if max_host_child_rss_growth_pct is not None
        else None,
        "min_ok_runs_per_build": max(0, min_ok_runs_per_build),
        "metrics": [asdict(metric) for metric in metrics],
        "deltas": [asdict(delta) for delta in deltas],
        "regressions": [asdict(regression) for regression in regressions],
        "coverage_violations": [asdict(violation) for violation in coverage_violations],
        "prompt_suite_drift": [asdict(violation) for violation in prompt_suite_drift],
    }
    json_path = output_dir / "build_compare_latest.json"
    md_path = output_dir / "build_compare_latest.md"
    csv_path = output_dir / "build_compare_latest.csv"
    coverage_csv_path = output_dir / "build_compare_coverage_violations_latest.csv"
    prompt_suite_drift_csv_path = output_dir / "build_compare_prompt_suite_drift_latest.csv"
    junit_path = output_dir / "build_compare_junit_latest.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(deltas, csv_path)
    write_coverage_csv(coverage_violations, coverage_csv_path)
    write_prompt_suite_drift_csv(prompt_suite_drift, prompt_suite_drift_csv_path)
    write_junit(deltas, regressions, coverage_violations, prompt_suite_drift, junit_path)
    return json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Benchmark report JSON, optionally named as BUILD=path. Repeat for each build.",
    )
    parser.add_argument("--baseline", help="Build name to compare against. Defaults to first --input build.")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument(
        "--max-tok-regression-pct",
        type=float,
        default=5.0,
        help="Allowed median tok/s drop before a regression is reported",
    )
    parser.add_argument(
        "--max-memory-growth-pct",
        type=float,
        help="Allowed max memory growth before a regression is reported; omitted disables memory gating",
    )
    parser.add_argument(
        "--max-wall-tok-regression-pct",
        type=float,
        help="Allowed host wall-clock median tok/s drop before a regression is reported; omitted disables wall-time gating",
    )
    parser.add_argument(
        "--max-p05-tok-regression-pct",
        type=float,
        help="Allowed P05 tok/s drop before a low-tail throughput regression is reported; omitted disables P05 gating",
    )
    parser.add_argument(
        "--max-p05-wall-tok-regression-pct",
        type=float,
        help=(
            "Allowed host wall-clock P05 tok/s drop before a low-tail wall-time regression is reported; "
            "omitted disables P05 wall-time gating"
        ),
    )
    parser.add_argument(
        "--max-ttft-growth-pct",
        type=float,
        help="Allowed median first-token latency growth before a regression is reported; omitted disables TTFT gating",
    )
    parser.add_argument(
        "--max-us-per-token-growth-pct",
        type=float,
        help="Allowed guest median us/token growth before a regression is reported; omitted disables token-latency gating",
    )
    parser.add_argument(
        "--max-wall-us-per-token-growth-pct",
        type=float,
        help="Allowed host wall-clock median us/token growth before a regression is reported; omitted disables wall token-latency gating",
    )
    parser.add_argument(
        "--max-host-child-cpu-growth-pct",
        type=float,
        help="Allowed median direct-child CPU time growth before a regression is reported; omitted disables CPU-time gating",
    )
    parser.add_argument(
        "--max-host-child-cpu-pct-growth-pct",
        type=float,
        help="Allowed median direct-child CPU utilization growth before a regression is reported; omitted disables CPU-utilization gating",
    )
    parser.add_argument(
        "--max-host-child-tok-per-cpu-s-regression-pct",
        type=float,
        help="Allowed direct-child tok/CPU-second drop before a regression is reported; omitted disables CPU-efficiency gating",
    )
    parser.add_argument(
        "--max-host-child-rss-growth-pct",
        type=float,
        help="Allowed direct-child peak RSS growth before a regression is reported; omitted disables host RSS gating",
    )
    parser.add_argument(
        "--min-ok-runs-per-build",
        type=int,
        default=0,
        help="Minimum successful runs required for each comparable baseline/candidate build point; 0 disables",
    )
    parser.add_argument("--fail-on-regression", action="store_true", help="Return non-zero on benchmark regression")
    parser.add_argument("--fail-on-coverage", action="store_true", help="Return non-zero on OK-run coverage violations")
    parser.add_argument(
        "--fail-on-prompt-suite-drift",
        action="store_true",
        help="Return non-zero when comparable builds report different prompt-suite hashes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        first_build, _ = parse_input_spec(args.input[0])
        baseline = args.baseline or first_build
        metrics = load_build_metrics(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if baseline not in {metric.build for metric in metrics}:
        print(f"error: baseline build {baseline!r} was not found in inputs", file=sys.stderr)
        return 2

    deltas = compare_builds(metrics, baseline)
    output = write_report(
        metrics,
        deltas,
        baseline,
        args.output_dir,
        max_tok_regression_pct=args.max_tok_regression_pct,
        max_wall_tok_regression_pct=args.max_wall_tok_regression_pct,
        max_memory_growth_pct=args.max_memory_growth_pct,
        max_ttft_growth_pct=args.max_ttft_growth_pct,
        max_p05_tok_regression_pct=args.max_p05_tok_regression_pct,
        max_p05_wall_tok_regression_pct=args.max_p05_wall_tok_regression_pct,
        max_us_per_token_growth_pct=args.max_us_per_token_growth_pct,
        max_wall_us_per_token_growth_pct=args.max_wall_us_per_token_growth_pct,
        max_host_child_cpu_growth_pct=args.max_host_child_cpu_growth_pct,
        max_host_child_cpu_pct_growth_pct=args.max_host_child_cpu_pct_growth_pct,
        max_host_child_tok_per_cpu_s_regression_pct=args.max_host_child_tok_per_cpu_s_regression_pct,
        max_host_child_rss_growth_pct=args.max_host_child_rss_growth_pct,
        min_ok_runs_per_build=args.min_ok_runs_per_build,
    )
    regressions = find_regressions(
        deltas,
        args.max_tok_regression_pct,
        args.max_memory_growth_pct,
        max_wall_tok_regression_pct=args.max_wall_tok_regression_pct,
        max_ttft_growth_pct=args.max_ttft_growth_pct,
        max_p05_tok_regression_pct=args.max_p05_tok_regression_pct,
        max_p05_wall_tok_regression_pct=args.max_p05_wall_tok_regression_pct,
        max_us_per_token_growth_pct=args.max_us_per_token_growth_pct,
        max_wall_us_per_token_growth_pct=args.max_wall_us_per_token_growth_pct,
        max_host_child_cpu_growth_pct=args.max_host_child_cpu_growth_pct,
        max_host_child_cpu_pct_growth_pct=args.max_host_child_cpu_pct_growth_pct,
        max_host_child_tok_per_cpu_s_regression_pct=args.max_host_child_tok_per_cpu_s_regression_pct,
        max_host_child_rss_growth_pct=args.max_host_child_rss_growth_pct,
    )
    coverage_violations = find_coverage_violations(deltas, args.min_ok_runs_per_build)
    prompt_suite_drift = find_prompt_suite_drift(deltas)
    print(f"wrote_json={output}")
    print(f"compared_deltas={len(deltas)}")
    print(f"regressions={len(regressions)}")
    print(f"coverage_violations={len(coverage_violations)}")
    print(f"prompt_suite_drift={len(prompt_suite_drift)}")
    if args.fail_on_regression and regressions:
        return 1
    if args.fail_on_coverage and coverage_violations:
        return 1
    if args.fail_on_prompt_suite_drift and prompt_suite_drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
