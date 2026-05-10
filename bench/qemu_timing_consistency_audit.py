#!/usr/bin/env python3
"""Audit QEMU benchmark artifacts for timing metric consistency.

This host-side tool reads saved benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class TimingRecord:
    source: str
    row_index: int
    benchmark: str
    profile: str
    quantization: str
    phase: str
    prompt: str
    iteration: int | None
    launch_index: int | None
    tokens: int | None
    elapsed_us: int | None
    wall_elapsed_us: int | None
    timeout_seconds: float | None
    tok_per_s: float | None
    wall_tok_per_s: float | None
    us_per_token: float | None
    wall_us_per_token: float | None
    host_overhead_us: int | None
    host_overhead_pct: float | None
    findings: int


@dataclass(frozen=True)
class Finding:
    source: str
    row_index: int
    severity: str
    kind: str
    field: str
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


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return None


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def close_enough(actual: float | None, expected: float | None, args: argparse.Namespace) -> bool:
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    return math.isclose(actual, expected, rel_tol=args.rel_tolerance, abs_tol=args.abs_tolerance)


def add_mismatch(
    findings: list[Finding],
    source: Path,
    row_index: int,
    field: str,
    actual: float | int | None,
    expected: float | int | None,
) -> None:
    findings.append(
        Finding(
            str(source),
            row_index,
            "error",
            "timing_metric_drift",
            field,
            f"stored={actual!r} expected={expected!r}",
        )
    )


def derived_rate(tokens: int | None, elapsed_us: int | None) -> float | None:
    if tokens is None or elapsed_us is None or elapsed_us <= 0:
        return None
    return tokens * 1_000_000.0 / elapsed_us


def derived_us_per_token(tokens: int | None, elapsed_us: int | None) -> float | None:
    if tokens is None or tokens <= 0 or elapsed_us is None or elapsed_us <= 0:
        return None
    return elapsed_us / tokens


def derived_timeout_pct(wall_elapsed_us: int | None, timeout_seconds: float | None) -> float | None:
    if wall_elapsed_us is None or timeout_seconds is None or timeout_seconds <= 0:
        return None
    return wall_elapsed_us * 100.0 / (timeout_seconds * 1_000_000.0)


def check_required_positive(
    findings: list[Finding],
    source: Path,
    row_index: int,
    field: str,
    value: int | float | None,
) -> None:
    if value is None:
        findings.append(Finding(str(source), row_index, "error", "missing_timing_field", field, "field is absent or not numeric"))
    elif value <= 0:
        findings.append(Finding(str(source), row_index, "error", "nonpositive_timing_field", field, f"value={value!r}"))


def audit_row(source: Path, row: dict[str, Any], row_index: int, args: argparse.Namespace) -> tuple[TimingRecord, list[Finding]]:
    findings: list[Finding] = []
    tokens = as_int(row.get("tokens"))
    elapsed_us = as_int(row.get("elapsed_us"))
    wall_elapsed_us = as_int(row.get("wall_elapsed_us"))
    timeout_seconds = as_float(row.get("timeout_seconds"))
    host_overhead_us = as_int(row.get("host_overhead_us"))

    check_required_positive(findings, source, row_index, "elapsed_us", elapsed_us)
    check_required_positive(findings, source, row_index, "wall_elapsed_us", wall_elapsed_us)
    check_required_positive(findings, source, row_index, "timeout_seconds", timeout_seconds)

    if tokens is not None and tokens < 0:
        findings.append(Finding(str(source), row_index, "error", "negative_tokens", "tokens", f"value={tokens!r}"))

    if elapsed_us is not None and wall_elapsed_us is not None:
        if wall_elapsed_us < elapsed_us:
            findings.append(
                Finding(
                    str(source),
                    row_index,
                    "error",
                    "wall_elapsed_before_guest_elapsed",
                    "wall_elapsed_us",
                    f"wall_elapsed_us={wall_elapsed_us} elapsed_us={elapsed_us}",
                )
            )
        expected_overhead = wall_elapsed_us - elapsed_us
        if host_overhead_us is None:
            findings.append(Finding(str(source), row_index, "error", "missing_timing_field", "host_overhead_us", "field is absent or not numeric"))
        elif host_overhead_us != expected_overhead:
            add_mismatch(findings, source, row_index, "host_overhead_us", host_overhead_us, expected_overhead)

    ttft_us = as_int(row.get("ttft_us"))
    if ttft_us is not None and elapsed_us is not None and ttft_us > elapsed_us:
        findings.append(Finding(str(source), row_index, "error", "ttft_exceeds_elapsed", "ttft_us", f"ttft_us={ttft_us} elapsed_us={elapsed_us}"))

    expected_tok_per_s = derived_rate(tokens, elapsed_us)
    expected_wall_tok_per_s = derived_rate(tokens, wall_elapsed_us)
    expected_us_per_token = derived_us_per_token(tokens, elapsed_us)
    expected_wall_us_per_token = derived_us_per_token(tokens, wall_elapsed_us)
    expected_host_overhead_pct = None
    if host_overhead_us is not None and elapsed_us is not None and elapsed_us > 0:
        expected_host_overhead_pct = host_overhead_us * 100.0 / elapsed_us
    expected_wall_timeout_pct = derived_timeout_pct(wall_elapsed_us, timeout_seconds)

    comparisons: tuple[tuple[str, float | None, float | None], ...] = (
        ("tok_per_s", as_float(row.get("tok_per_s")), expected_tok_per_s),
        ("wall_tok_per_s", as_float(row.get("wall_tok_per_s")), expected_wall_tok_per_s),
        ("us_per_token", as_float(row.get("us_per_token")), expected_us_per_token),
        ("wall_us_per_token", as_float(row.get("wall_us_per_token")), expected_wall_us_per_token),
        ("host_overhead_pct", as_float(row.get("host_overhead_pct")), expected_host_overhead_pct),
        ("wall_timeout_pct", as_float(row.get("wall_timeout_pct")), expected_wall_timeout_pct),
    )
    for field, actual, expected in comparisons:
        if expected is None:
            continue
        if actual is None:
            findings.append(Finding(str(source), row_index, "error", "missing_timing_field", field, "field is absent or not numeric"))
        elif not close_enough(actual, expected, args):
            add_mismatch(findings, source, row_index, field, actual, expected)

    host_child_user_cpu_us = as_int(row.get("host_child_user_cpu_us"))
    host_child_system_cpu_us = as_int(row.get("host_child_system_cpu_us"))
    host_child_cpu_us = as_int(row.get("host_child_cpu_us"))
    if host_child_user_cpu_us is not None and host_child_system_cpu_us is not None:
        expected_cpu_us = host_child_user_cpu_us + host_child_system_cpu_us
        if host_child_cpu_us is None:
            findings.append(Finding(str(source), row_index, "error", "missing_timing_field", "host_child_cpu_us", "field is absent or not numeric"))
        elif host_child_cpu_us != expected_cpu_us:
            add_mismatch(findings, source, row_index, "host_child_cpu_us", host_child_cpu_us, expected_cpu_us)

    record = TimingRecord(
        source=str(source),
        row_index=row_index,
        benchmark=as_text(row.get("benchmark")),
        profile=as_text(row.get("profile")),
        quantization=as_text(row.get("quantization")),
        phase=as_text(row.get("phase")),
        prompt=as_text(row.get("prompt")),
        iteration=as_int(row.get("iteration")),
        launch_index=as_int(row.get("launch_index")),
        tokens=tokens,
        elapsed_us=elapsed_us,
        wall_elapsed_us=wall_elapsed_us,
        timeout_seconds=timeout_seconds,
        tok_per_s=as_float(row.get("tok_per_s")),
        wall_tok_per_s=as_float(row.get("wall_tok_per_s")),
        us_per_token=as_float(row.get("us_per_token")),
        wall_us_per_token=as_float(row.get("wall_us_per_token")),
        host_overhead_us=host_overhead_us,
        host_overhead_pct=as_float(row.get("host_overhead_pct")),
        findings=len(findings),
    )
    return record, findings


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[list[TimingRecord], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return [], [Finding(str(path), -1, "error", "load_error", "", error)]

    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list):
        return [], [Finding(str(path), -1, "error", "missing_benchmarks", "benchmarks", "benchmarks must be a list")]

    records: list[TimingRecord] = []
    findings: list[Finding] = []
    for row_index, row in enumerate(benchmarks):
        if not isinstance(row, dict):
            findings.append(Finding(str(path), row_index, "error", "benchmark_row_type", "benchmarks", "benchmark row must be an object"))
            continue
        if args.measured_only and row.get("phase") != "measured":
            continue
        record, row_findings = audit_row(path, row, row_index, args)
        records.append(record)
        findings.extend(row_findings)

    if not records:
        findings.append(Finding(str(path), -1, "error", "no_benchmark_rows", "benchmarks", "no benchmark rows were audited"))
    return records, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[TimingRecord], list[Finding]]:
    records: list[TimingRecord] = []
    findings: list[Finding] = []
    seen = 0
    for path in iter_input_files(paths, args.pattern):
        seen += 1
        artifact_records, artifact_findings = audit_artifact(path, args)
        records.extend(artifact_records)
        findings.extend(artifact_findings)
    if seen == 0:
        findings.append(Finding("", -1, "error", "no_inputs", "inputs", "no benchmark artifacts matched"))
    return records, findings


def write_outputs(records: list[TimingRecord], findings: list[Finding], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "pass" if not any(finding.severity == "error" for finding in findings) else "fail"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "summary": {
            "rows": len(records),
            "findings": len(findings),
            "artifacts": len({record.source for record in records}),
        },
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (args.output_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(TimingRecord.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    with (args.output_dir / f"{stem}_findings.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(Finding.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))

    lines = [
        "# QEMU Timing Consistency Audit",
        "",
        f"Status: {status}",
        f"Rows: {len(records)}",
        f"Findings: {len(findings)}",
    ]
    if findings:
        lines.extend(["", "## Findings", ""])
        for finding in findings:
            lines.append(f"- {finding.severity}: row {finding.row_index} {finding.kind} {finding.field} - {finding.detail}")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = sum(1 for finding in findings if finding.severity == "error")
    suite = ET.Element("testsuite", name="qemu_timing_consistency_audit", tests=str(max(1, len(records))), failures=str(failures))
    if not records:
        case = ET.SubElement(suite, "testcase", name="inputs")
        failure = ET.SubElement(case, "failure", message="no benchmark rows audited")
        failure.text = "no benchmark rows audited"
    for record in records:
        name = f"{Path(record.source).name}:{record.launch_index}:{record.prompt}"
        case = ET.SubElement(suite, "testcase", name=name)
        record_findings = [
            finding
            for finding in findings
            if finding.source == record.source and finding.row_index == record.row_index
        ]
        if record.findings and record_findings:
            failure = ET.SubElement(case, "failure", message=f"{len(record_findings)} timing findings")
            failure.text = "\n".join(f"{finding.kind}: {finding.field}: {finding.detail}" for finding in record_findings)
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)
    with (args.output_dir / f"{stem}_junit.xml").open("ab") as handle:
        handle.write(b"\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob pattern used when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_timing_consistency_audit_latest")
    parser.add_argument("--measured-only", action="store_true", help="audit only rows with phase=measured")
    parser.add_argument("--rel-tolerance", type=float, default=1e-6, help="relative tolerance for derived float checks")
    parser.add_argument("--abs-tolerance", type=float, default=1e-6, help="absolute tolerance for derived float checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    records, findings = audit(args.inputs, args)
    write_outputs(records, findings, args)
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
