#!/usr/bin/env python3
"""Audit saved QEMU benchmark matrix plans for launch and work budgets.

This host-side tool reads JSON artifacts produced by qemu_benchmark_matrix.py.
It never launches QEMU and never touches the TempleOS guest.
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


@dataclass(frozen=True)
class BudgetRow:
    source: str
    build: str
    profile: str
    model: str
    quantization: str
    launches: int
    warmup_launches: int
    measured_launches: int
    prompt_bytes_total: int
    expected_tokens_total: int
    missing_expected_tokens: int
    command_airgap_ok: bool


@dataclass(frozen=True)
class BudgetFinding:
    source: str
    build: str
    severity: str
    kind: str
    metric: str
    value: int | float | str
    limit: int | float | str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_int(value: Any, default: int = 0) -> int:
    if value in (None, "") or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return int(number)


def text_value(value: Any, default: str = "-") -> str:
    if value in (None, ""):
        return default
    return str(value)


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: matrix artifact root must be a JSON object")
    return payload


def iter_matrix_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(child for child in path.rglob("qemu_benchmark_matrix*.json") if child.is_file())
        elif path.is_file():
            yield path


def build_rows(path: Path, payload: dict[str, Any]) -> list[BudgetRow]:
    builds = payload.get("builds")
    launches = payload.get("launches")
    if not isinstance(builds, list) or not isinstance(launches, list):
        raise ValueError(f"{path}: matrix artifact must include builds and launches arrays")

    launch_groups: dict[str, list[dict[str, Any]]] = {}
    for launch in launches:
        if isinstance(launch, dict):
            launch_groups.setdefault(text_value(launch.get("build")), []).append(launch)

    rows: list[BudgetRow] = []
    for build in builds:
        if not isinstance(build, dict):
            continue
        name = text_value(build.get("build"))
        build_launches = launch_groups.get(name, [])
        expected_tokens = [launch.get("expected_tokens") for launch in build_launches]
        rows.append(
            BudgetRow(
                source=str(path),
                build=name,
                profile=text_value(build.get("profile")),
                model=text_value(build.get("model")),
                quantization=text_value(build.get("quantization")),
                launches=len(build_launches),
                warmup_launches=sum(1 for launch in build_launches if text_value(launch.get("phase")) == "warmup"),
                measured_launches=sum(1 for launch in build_launches if text_value(launch.get("phase")) == "measured"),
                prompt_bytes_total=sum(finite_int(launch.get("prompt_bytes")) for launch in build_launches),
                expected_tokens_total=sum(finite_int(value) for value in expected_tokens),
                missing_expected_tokens=sum(1 for value in expected_tokens if value in (None, "")),
                command_airgap_ok=bool(build.get("command_airgap_ok")),
            )
        )
    return rows


def add_limit_finding(
    findings: list[BudgetFinding],
    row: BudgetRow,
    *,
    kind: str,
    metric: str,
    value: int,
    limit: int | None,
    detail: str,
) -> None:
    if limit is not None and value > limit:
        findings.append(BudgetFinding(row.source, row.build, "error", kind, metric, value, limit, detail))


def evaluate(rows: list[BudgetRow], args: argparse.Namespace) -> list[BudgetFinding]:
    findings: list[BudgetFinding] = []
    if len(rows) < args.min_builds:
        findings.append(
            BudgetFinding(
                "",
                "",
                "error",
                "min_builds",
                "builds",
                len(rows),
                args.min_builds,
                f"found {len(rows)} build budget row(s), below minimum {args.min_builds}",
            )
        )

    for row in rows:
        add_limit_finding(
            findings,
            row,
            kind="max_launches_per_build",
            metric="launches",
            value=row.launches,
            limit=args.max_launches_per_build,
            detail=f"{row.launches} planned launches exceed per-build limit {args.max_launches_per_build}",
        )
        add_limit_finding(
            findings,
            row,
            kind="max_prompt_bytes_per_build",
            metric="prompt_bytes_total",
            value=row.prompt_bytes_total,
            limit=args.max_prompt_bytes_per_build,
            detail=f"{row.prompt_bytes_total} prompt bytes exceed per-build limit {args.max_prompt_bytes_per_build}",
        )
        add_limit_finding(
            findings,
            row,
            kind="max_expected_tokens_per_build",
            metric="expected_tokens_total",
            value=row.expected_tokens_total,
            limit=args.max_expected_tokens_per_build,
            detail=f"{row.expected_tokens_total} expected tokens exceed per-build limit {args.max_expected_tokens_per_build}",
        )
        if args.require_expected_tokens and row.missing_expected_tokens:
            findings.append(
                BudgetFinding(
                    row.source,
                    row.build,
                    "error",
                    "missing_expected_tokens",
                    "missing_expected_tokens",
                    row.missing_expected_tokens,
                    0,
                    f"{row.missing_expected_tokens} planned launch(es) omit expected token counts",
                )
            )
        if args.require_airgap and not row.command_airgap_ok:
            findings.append(
                BudgetFinding(
                    row.source,
                    row.build,
                    "error",
                    "command_airgap",
                    "command_airgap_ok",
                    "false",
                    "true",
                    "matrix build command is not marked air-gap safe",
                )
            )

    total_launches = sum(row.launches for row in rows)
    if args.max_launches is not None and total_launches > args.max_launches:
        findings.append(
            BudgetFinding(
                "",
                "",
                "error",
                "max_launches",
                "launches",
                total_launches,
                args.max_launches,
                f"{total_launches} planned launches exceed global limit {args.max_launches}",
            )
        )
    total_prompt_bytes = sum(row.prompt_bytes_total for row in rows)
    if args.max_prompt_bytes is not None and total_prompt_bytes > args.max_prompt_bytes:
        findings.append(
            BudgetFinding(
                "",
                "",
                "error",
                "max_prompt_bytes",
                "prompt_bytes_total",
                total_prompt_bytes,
                args.max_prompt_bytes,
                f"{total_prompt_bytes} planned prompt bytes exceed global limit {args.max_prompt_bytes}",
            )
        )
    return findings


def build_report(paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    rows: list[BudgetRow] = []
    for path in iter_matrix_files(paths):
        rows.extend(build_rows(path, load_json_object(path)))
    findings = evaluate(rows, args)
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "sources": len({row.source for row in rows}),
            "builds": len(rows),
            "launches": sum(row.launches for row in rows),
            "warmup_launches": sum(row.warmup_launches for row in rows),
            "measured_launches": sum(row.measured_launches for row in rows),
            "prompt_bytes_total": sum(row.prompt_bytes_total for row in rows),
            "expected_tokens_total": sum(row.expected_tokens_total for row in rows),
            "missing_expected_tokens": sum(row.missing_expected_tokens for row in rows),
            "findings": len(findings),
        },
        "config": {
            "min_builds": args.min_builds,
            "max_launches": args.max_launches,
            "max_launches_per_build": args.max_launches_per_build,
            "max_prompt_bytes": args.max_prompt_bytes,
            "max_prompt_bytes_per_build": args.max_prompt_bytes_per_build,
            "max_expected_tokens_per_build": args.max_expected_tokens_per_build,
            "require_expected_tokens": args.require_expected_tokens,
            "require_airgap": args.require_airgap,
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(BudgetRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_findings_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    fieldnames = list(BudgetFinding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(finding)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Matrix Budget Audit",
        "",
        f"- Status: {report['status']}",
        f"- Sources: {summary['sources']}",
        f"- Builds: {summary['builds']}",
        f"- Launches: {summary['launches']}",
        f"- Prompt bytes: {summary['prompt_bytes_total']}",
        f"- Expected tokens: {summary['expected_tokens_total']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Build | Profile | Model | Quantization | Launches | Prompt bytes | Expected tokens | Missing expected tokens | Air-gap |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {build} | {profile} | {model} | {quantization} | {launches} | {prompt_bytes_total} | {expected_tokens_total} | {missing_expected_tokens} | {command_airgap_ok} |".format(
                **row
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append("- {kind}: {build} {metric}={value} limit={limit} ({detail})".format(**finding))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_matrix_budget_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding['kind']}:{finding['build'] or 'global'}"})
            failure = ET.SubElement(case, "failure", {"message": finding["detail"]})
            failure.text = finding["detail"]
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_matrix_budget"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def write_report(report: dict[str, Any], output_dir: Path, output_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / output_stem
    write_json(base.with_suffix(".json"), report)
    write_csv(base.with_suffix(".csv"), report["rows"])
    write_findings_csv(base.with_name(f"{base.name}_findings.csv"), report["findings"])
    write_markdown(base.with_suffix(".md"), report)
    write_junit(base.with_name(f"{base.name}_junit.xml"), report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Matrix JSON artifact(s) or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_matrix_budget_audit_latest")
    parser.add_argument("--min-builds", type=int, default=1)
    parser.add_argument("--max-launches", type=int)
    parser.add_argument("--max-launches-per-build", type=int)
    parser.add_argument("--max-prompt-bytes", type=int)
    parser.add_argument("--max-prompt-bytes-per-build", type=int)
    parser.add_argument("--max-expected-tokens-per-build", type=int)
    parser.add_argument("--require-expected-tokens", action="store_true")
    parser.add_argument("--require-airgap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(args.paths, args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), flush=True)
        return 2
    write_report(report, args.output_dir, args.output_stem)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
