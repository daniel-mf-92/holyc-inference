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
    runs: int
    ok_runs: int
    median_tokens: float | None
    median_elapsed_us: float | None
    median_tok_per_s: float | None
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
    baseline_tok_per_s: float | None
    candidate_tok_per_s: float | None
    tok_per_s_delta_pct: float | None
    baseline_elapsed_us: float | None
    candidate_elapsed_us: float | None
    elapsed_delta_pct: float | None
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
        memory_values = [
            value for row in key_rows if (value := parse_int(row.get("memory_bytes"))) is not None and value >= 0
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
                runs=len(key_rows),
                ok_runs=ok_runs,
                median_tokens=statistics.median(token_values) if token_values else None,
                median_elapsed_us=statistics.median(elapsed_values) if elapsed_values else None,
                median_tok_per_s=statistics.median(tok_values) if tok_values else None,
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
                    baseline_tok_per_s=baseline.median_tok_per_s,
                    candidate_tok_per_s=candidate.median_tok_per_s,
                    tok_per_s_delta_pct=pct_delta(candidate.median_tok_per_s, baseline.median_tok_per_s),
                    baseline_elapsed_us=baseline.median_elapsed_us,
                    candidate_elapsed_us=candidate.median_elapsed_us,
                    elapsed_delta_pct=pct_delta(candidate.median_elapsed_us, baseline.median_elapsed_us),
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
) -> list[BuildRegression]:
    threshold = -abs(max_tok_regression_pct)
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
    return regressions


def throughput_regressions(deltas: list[BuildDelta], max_tok_regression_pct: float) -> list[BuildRegression]:
    return [
        regression
        for regression in find_regressions(deltas, max_tok_regression_pct)
        if regression.metric == "tok_per_s"
    ]


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
        f"Memory regressions: {len([row for row in report['regressions'] if row['metric'] == 'memory_bytes'])}",
        "",
        "## Deltas",
        "",
    ]
    if report["deltas"]:
        lines.append(
            "| Candidate | Prompt key | Base tok/s | Candidate tok/s | Tok/s delta % | Base elapsed us | Candidate elapsed us | Elapsed delta % | Base memory bytes | Candidate memory bytes | Memory delta % |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for delta in report["deltas"]:
            lines.append(
                "| {candidate_build} | {key} | {baseline_tok_per_s} | {candidate_tok_per_s} | "
                "{tok_per_s_delta_pct} | {baseline_elapsed_us} | {candidate_elapsed_us} | {elapsed_delta_pct} | "
                "{baseline_memory_bytes} | {candidate_memory_bytes} | {memory_delta_pct} |".format(
                    **{key: format_value(value) for key, value in delta.items()}
                )
            )
    else:
        lines.append("No comparable prompt metrics found.")
    return "\n".join(lines) + "\n"


def write_csv(deltas: list[BuildDelta], path: Path) -> None:
    fields = [
        "key",
        "baseline_build",
        "candidate_build",
        "baseline_commit",
        "candidate_commit",
        "baseline_tok_per_s",
        "candidate_tok_per_s",
        "tok_per_s_delta_pct",
        "baseline_elapsed_us",
        "candidate_elapsed_us",
        "elapsed_delta_pct",
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


def write_junit(deltas: list[BuildDelta], regressions: list[BuildRegression], path: Path) -> None:
    regression_by_key: dict[tuple[str, str], list[BuildRegression]] = {}
    for regression in regressions:
        regression_by_key.setdefault((regression.candidate_build, regression.key), []).append(regression)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_build_compare",
            "tests": str(len(deltas)),
            "failures": str(len(regressions)),
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
                f"baseline_memory_bytes={format_value(delta.baseline_memory_bytes)}\n"
                f"candidate_memory_bytes={format_value(delta.candidate_memory_bytes)}\n"
                f"memory_delta_pct={format_value(delta.memory_delta_pct)}\n"
                f"baseline_commit={delta.baseline_commit}\n"
                f"candidate_commit={delta.candidate_commit}\n"
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
    max_memory_growth_pct: float | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    regressions = find_regressions(deltas, max_tok_regression_pct, max_memory_growth_pct)
    report = {
        "generated_at": iso_now(),
        "status": "fail" if regressions else "pass",
        "baseline_build": baseline_build,
        "builds": sorted({metric.build for metric in metrics}),
        "max_tok_regression_pct": abs(max_tok_regression_pct),
        "max_memory_growth_pct": abs(max_memory_growth_pct) if max_memory_growth_pct is not None else None,
        "metrics": [asdict(metric) for metric in metrics],
        "deltas": [asdict(delta) for delta in deltas],
        "regressions": [asdict(regression) for regression in regressions],
    }
    json_path = output_dir / "build_compare_latest.json"
    md_path = output_dir / "build_compare_latest.md"
    csv_path = output_dir / "build_compare_latest.csv"
    junit_path = output_dir / "build_compare_junit_latest.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(deltas, csv_path)
    write_junit(deltas, regressions, junit_path)
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
    parser.add_argument("--fail-on-regression", action="store_true", help="Return non-zero on benchmark regression")
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
        max_memory_growth_pct=args.max_memory_growth_pct,
    )
    regressions = find_regressions(deltas, args.max_tok_regression_pct, args.max_memory_growth_pct)
    print(f"wrote_json={output}")
    print(f"compared_deltas={len(deltas)}")
    print(f"regressions={len(regressions)}")
    if args.fail_on_regression and regressions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
