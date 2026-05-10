#!/usr/bin/env python3
"""Recommend QEMU benchmark timeout budgets from saved prompt artifacts.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
ROW_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class TimeoutSample:
    source: str
    benchmark: str
    profile: str
    model: str
    quantization: str
    prompt: str
    wall_elapsed_s: float
    timeout_seconds: float | None


@dataclass(frozen=True)
class TimeoutRecommendation:
    benchmark: str
    profile: str
    model: str
    quantization: str
    samples: int
    prompts: int
    timeout_samples: int
    min_wall_s: float
    median_wall_s: float
    p95_wall_s: float
    max_wall_s: float
    current_timeout_s: float | None
    current_timeout_headroom_pct: float | None
    recommended_timeout_s: int
    recommendation_delta_s: float | None
    sources: str


@dataclass(frozen=True)
class Finding:
    severity: str
    benchmark: str
    profile: str
    model: str
    quantization: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_input_files(paths: Iterable[Path], patterns: list[str]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            seen: set[Path] = set()
            for pattern in patterns:
                for child in sorted(path.rglob(pattern)):
                    if child.is_file() and child not in seen:
                        seen.add(child)
                        yield child
        elif path.is_file():
            yield path


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "artifact root must be a JSON object"
    return payload, ""


def finite_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def row_is_ok(row: dict[str, Any]) -> bool:
    exit_class = str(row.get("exit_class") or "").lower()
    if exit_class and exit_class != "ok":
        return False
    if row.get("timed_out") is True:
        return False
    returncode = finite_float(row.get("returncode"))
    return returncode in (None, 0.0)


def iter_artifact_rows(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ROW_KEYS:
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for row in value:
            if isinstance(row, dict):
                yield row


def read_samples(path: Path, args: argparse.Namespace) -> tuple[list[TimeoutSample], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return [], [Finding("error", "", "", "", "", "load_error", f"{path}: {error}")]

    samples: list[TimeoutSample] = []
    findings: list[Finding] = []
    for row in iter_artifact_rows(payload):
        phase = str(row.get("phase") or "")
        if args.phase and phase != args.phase:
            continue
        if args.require_success and not row_is_ok(row):
            continue
        wall_elapsed_us = finite_float(row.get("wall_elapsed_us"))
        if wall_elapsed_us is None:
            continue
        wall_elapsed_s = wall_elapsed_us / 1_000_000.0
        if wall_elapsed_s <= 0:
            continue
        timeout_seconds = finite_float(row.get("timeout_seconds"))
        samples.append(
            TimeoutSample(
                source=str(path),
                benchmark=str(row.get("benchmark") or payload.get("benchmark") or ""),
                profile=str(row.get("profile") or payload.get("profile") or ""),
                model=str(row.get("model") or payload.get("model") or ""),
                quantization=str(row.get("quantization") or payload.get("quantization") or ""),
                prompt=str(row.get("prompt") or ""),
                wall_elapsed_s=wall_elapsed_s,
                timeout_seconds=timeout_seconds if timeout_seconds is not None and timeout_seconds > 0 else None,
            )
        )

    if not samples and args.require_rows:
        findings.append(Finding("error", "", "", "", "", "missing_samples", f"{path}: no usable timing rows found"))
    return samples, findings


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def group_key(sample: TimeoutSample) -> tuple[str, str, str, str]:
    return (sample.benchmark, sample.profile, sample.model, sample.quantization)


def recommend_for_group(samples: list[TimeoutSample], args: argparse.Namespace) -> TimeoutRecommendation:
    walls = [sample.wall_elapsed_s for sample in samples]
    timeouts = [sample.timeout_seconds for sample in samples if sample.timeout_seconds is not None]
    p95 = percentile(walls, 95.0)
    recommended = max(args.min_timeout_seconds, math.ceil(p95 * args.p95_multiplier + args.additive_seconds))
    current = max(timeouts) if timeouts else None
    headroom = None if current is None else ((current - max(walls)) / current) * 100.0
    delta = None if current is None else recommended - current
    first = samples[0]
    return TimeoutRecommendation(
        benchmark=first.benchmark,
        profile=first.profile,
        model=first.model,
        quantization=first.quantization,
        samples=len(samples),
        prompts=len({sample.prompt for sample in samples}),
        timeout_samples=len(timeouts),
        min_wall_s=min(walls),
        median_wall_s=percentile(walls, 50.0),
        p95_wall_s=p95,
        max_wall_s=max(walls),
        current_timeout_s=current,
        current_timeout_headroom_pct=headroom,
        recommended_timeout_s=int(recommended),
        recommendation_delta_s=delta,
        sources=";".join(sorted({sample.source for sample in samples})),
    )


def build_report(samples: list[TimeoutSample], args: argparse.Namespace, findings: list[Finding]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[TimeoutSample]] = {}
    for sample in samples:
        groups.setdefault(group_key(sample), []).append(sample)

    recommendations = [recommend_for_group(group_samples, args) for _, group_samples in sorted(groups.items())]
    for recommendation in recommendations:
        if recommendation.samples < args.min_samples:
            findings.append(
                Finding(
                    "error",
                    recommendation.benchmark,
                    recommendation.profile,
                    recommendation.model,
                    recommendation.quantization,
                    "low_sample_count",
                    f"samples {recommendation.samples} below minimum {args.min_samples}",
                )
            )
        if args.require_timeout_telemetry and recommendation.timeout_samples < recommendation.samples:
            findings.append(
                Finding(
                    "error",
                    recommendation.benchmark,
                    recommendation.profile,
                    recommendation.model,
                    recommendation.quantization,
                    "missing_timeout_telemetry",
                    f"{recommendation.samples - recommendation.timeout_samples} row(s) lack timeout_seconds",
                )
            )
        if (
            args.fail_if_current_below_recommended
            and recommendation.current_timeout_s is not None
            and recommendation.current_timeout_s < recommendation.recommended_timeout_s
        ):
            findings.append(
                Finding(
                    "error",
                    recommendation.benchmark,
                    recommendation.profile,
                    recommendation.model,
                    recommendation.quantization,
                    "current_timeout_below_recommended",
                    f"current {recommendation.current_timeout_s:g}s below recommended {recommendation.recommended_timeout_s}s",
                )
            )
        if (
            args.min_current_timeout_headroom_pct is not None
            and recommendation.current_timeout_headroom_pct is not None
            and recommendation.current_timeout_headroom_pct < args.min_current_timeout_headroom_pct
        ):
            findings.append(
                Finding(
                    "error",
                    recommendation.benchmark,
                    recommendation.profile,
                    recommendation.model,
                    recommendation.quantization,
                    "current_timeout_headroom_low",
                    (
                        f"current timeout headroom {recommendation.current_timeout_headroom_pct:.3f}% "
                        f"below minimum {args.min_current_timeout_headroom_pct:.3f}%"
                    ),
                )
            )

    if args.require_rows and not recommendations:
        findings.append(Finding("error", "", "", "", "", "missing_groups", "no timeout recommendation groups found"))

    return {
        "generated_at": iso_now(),
        "status": "fail" if any(finding.severity == "error" for finding in findings) else "pass",
        "settings": {
            "phase": args.phase,
            "require_success": args.require_success,
            "min_samples": args.min_samples,
            "min_timeout_seconds": args.min_timeout_seconds,
            "p95_multiplier": args.p95_multiplier,
            "additive_seconds": args.additive_seconds,
            "require_timeout_telemetry": args.require_timeout_telemetry,
            "fail_if_current_below_recommended": args.fail_if_current_below_recommended,
            "min_current_timeout_headroom_pct": args.min_current_timeout_headroom_pct,
        },
        "summary": {
            "groups": len(recommendations),
            "samples": len(samples),
            "findings": len(findings),
            "max_recommended_timeout_s": max((item.recommended_timeout_s for item in recommendations), default=None),
        },
        "recommendations": [asdict(item) for item in recommendations],
        "findings": [asdict(item) for item in findings],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# QEMU Timeout Recommendations",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Groups | Samples | Findings | Max recommended timeout |",
        "| ---: | ---: | ---: | ---: |",
        "| {groups} | {samples} | {findings} | {max_timeout} |".format(
            groups=report["summary"]["groups"],
            samples=report["summary"]["samples"],
            findings=report["summary"]["findings"],
            max_timeout=report["summary"]["max_recommended_timeout_s"] or "",
        ),
        "",
        "| Benchmark | Profile | Model | Quantization | Samples | P95 wall s | Current timeout s | Current headroom % | Recommended timeout s |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["recommendations"]:
        lines.append(
            "| {benchmark} | {profile} | {model} | {quantization} | {samples} | {p95:.6g} | {current} | {headroom} | {recommended} |".format(
                benchmark=row["benchmark"],
                profile=row["profile"],
                model=row["model"],
                quantization=row["quantization"],
                samples=row["samples"],
                p95=row["p95_wall_s"],
                current="" if row["current_timeout_s"] is None else f"{row['current_timeout_s']:.6g}",
                headroom=(
                    ""
                    if row["current_timeout_headroom_pct"] is None
                    else f"{row['current_timeout_headroom_pct']:.6g}"
                ),
                recommended=row["recommended_timeout_s"],
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", "", "| Severity | Kind | Detail |", "| --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['severity']} | {finding['kind']} | {finding['detail']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    root = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_timeout_recommend",
            "tests": "1",
            "failures": str(sum(1 for finding in findings if finding["severity"] == "error")),
        },
    )
    case = ET.SubElement(root, "testcase", {"name": "timeout_recommendations"})
    if report["status"] == "fail":
        failure = ET.SubElement(case, "failure", {"message": "timeout recommendation findings"})
        failure.text = "\n".join(f"{item['kind']}: {item['detail']}" for item in findings if item["severity"] == "error")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_outputs(report: dict[str, Any], output_dir: Path, output_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{output_stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir / f"{output_stem}.md", report)
    write_csv(
        output_dir / f"{output_stem}.csv",
        report["recommendations"],
        [
            "benchmark",
            "profile",
            "model",
            "quantization",
            "samples",
            "prompts",
            "timeout_samples",
            "min_wall_s",
            "median_wall_s",
            "p95_wall_s",
            "max_wall_s",
            "current_timeout_s",
            "current_timeout_headroom_pct",
            "recommended_timeout_s",
            "recommendation_delta_s",
            "sources",
        ],
    )
    write_csv(
        output_dir / f"{output_stem}_findings.csv",
        report["findings"],
        ["severity", "benchmark", "profile", "model", "quantization", "kind", "detail"],
    )
    write_junit(output_dir / f"{output_stem}_junit.xml", report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", dest="patterns", help="Glob to use when scanning directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_timeout_recommend_latest")
    parser.add_argument("--phase", default="measured", help="Benchmark phase to include, or empty for all phases")
    parser.add_argument("--include-failures", action="store_false", dest="require_success", help="Include failed rows")
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--min-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--p95-multiplier", type=float, default=3.0)
    parser.add_argument("--additive-seconds", type=float, default=5.0)
    parser.add_argument("--require-timeout-telemetry", action="store_true")
    parser.add_argument("--fail-if-current-below-recommended", action="store_true")
    parser.add_argument(
        "--min-current-timeout-headroom-pct",
        type=float,
        help="Fail groups whose recorded timeout leaves less than this percent headroom above max wall time",
    )
    parser.add_argument("--no-require-rows", action="store_false", dest="require_rows")
    parser.set_defaults(require_success=True, require_rows=True)
    return parser


def audit(paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    patterns = args.patterns or list(DEFAULT_PATTERNS)
    samples: list[TimeoutSample] = []
    findings: list[Finding] = []
    for path in iter_input_files(paths, patterns):
        path_samples, path_findings = read_samples(path, args)
        samples.extend(path_samples)
        findings.extend(path_findings)
    return build_report(samples, args, findings)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_samples < 1:
        parser.error("--min-samples must be at least 1")
    if args.min_timeout_seconds <= 0:
        parser.error("--min-timeout-seconds must be positive")
    if args.p95_multiplier <= 0:
        parser.error("--p95-multiplier must be positive")
    if args.additive_seconds < 0:
        parser.error("--additive-seconds must be non-negative")
    if args.min_current_timeout_headroom_pct is not None and args.min_current_timeout_headroom_pct < 0:
        parser.error("--min-current-timeout-headroom-pct must be non-negative")
    report = audit(args.paths, args)
    write_outputs(report, args.output_dir, args.output_stem)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
