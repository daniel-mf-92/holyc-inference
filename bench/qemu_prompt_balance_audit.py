#!/usr/bin/env python3
"""Audit QEMU prompt benchmark artifacts for balanced prompt sampling.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
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
from typing import Any, Iterable


@dataclass(frozen=True)
class PromptBalanceRow:
    source: str
    prompt: str
    measured_runs: int
    successful_runs: int
    failed_runs: int
    warmup_runs: int
    min_iteration: int | None
    max_iteration: int | None
    expected_successful_runs: int
    successful_run_delta: int


@dataclass(frozen=True)
class ArtifactBalance:
    source: str
    status: str
    profile: str
    model: str
    quantization: str
    prompts: int
    measured_runs: int
    successful_runs: int
    failed_runs: int
    min_successful_runs: int
    max_successful_runs: int
    successful_run_delta: int
    error: str = ""


@dataclass(frozen=True)
class BalanceFinding:
    source: str
    prompt: str
    severity: str
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


def int_or_none(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def benchmark_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    rows = payload.get("benchmarks")
    if not isinstance(rows, list):
        return [], "missing benchmarks list"
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return [], f"benchmark row {index} must be an object"
        selected.append(row)
    return selected, ""


def is_measured(row: dict[str, Any]) -> bool:
    return str(row.get("phase") or "measured") == "measured"


def is_success(row: dict[str, Any]) -> bool:
    return str(row.get("exit_class") or "") == "ok" and not bool(row.get("timed_out")) and row.get("failure_reason") in (None, "")


def prompt_name(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or "")


def build_prompt_rows(source: Path, rows: list[dict[str, Any]]) -> list[PromptBalanceRow]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        prompt = prompt_name(row)
        if prompt:
            by_prompt.setdefault(prompt, []).append(row)

    success_counts = [
        sum(1 for row in grouped if is_measured(row) and is_success(row))
        for grouped in by_prompt.values()
    ]
    expected = max(success_counts) if success_counts else 0

    prompt_rows: list[PromptBalanceRow] = []
    for prompt in sorted(by_prompt):
        grouped = by_prompt[prompt]
        measured = [row for row in grouped if is_measured(row)]
        successful = [row for row in measured if is_success(row)]
        iterations = [value for value in (int_or_none(row.get("iteration")) for row in measured) if value is not None]
        prompt_rows.append(
            PromptBalanceRow(
                source=str(source),
                prompt=prompt,
                measured_runs=len(measured),
                successful_runs=len(successful),
                failed_runs=len(measured) - len(successful),
                warmup_runs=len(grouped) - len(measured),
                min_iteration=min(iterations) if iterations else None,
                max_iteration=max(iterations) if iterations else None,
                expected_successful_runs=expected,
                successful_run_delta=expected - len(successful),
            )
        )
    return prompt_rows


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactBalance, list[PromptBalanceRow], list[BalanceFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return (
            ArtifactBalance(str(path), "fail", "", "", "", 0, 0, 0, 0, 0, 0, 0, error),
            [],
            [BalanceFinding(str(path), "", "error", "load_error", error)],
        )

    rows, row_error = benchmark_rows(payload)
    if row_error:
        return (
            ArtifactBalance(
                str(path),
                "fail",
                str(payload.get("profile") or ""),
                str(payload.get("model") or ""),
                str(payload.get("quantization") or ""),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                row_error,
            ),
            [],
            [BalanceFinding(str(path), "", "error", "row_error", row_error)],
        )

    prompt_rows = build_prompt_rows(path, rows)
    findings: list[BalanceFinding] = []
    measured_runs = sum(row.measured_runs for row in prompt_rows)
    successful_runs = sum(row.successful_runs for row in prompt_rows)
    failed_runs = sum(row.failed_runs for row in prompt_rows)
    success_counts = [row.successful_runs for row in prompt_rows]
    min_success = min(success_counts) if success_counts else 0
    max_success = max(success_counts) if success_counts else 0
    delta = max_success - min_success

    if len(prompt_rows) < args.min_prompts:
        findings.append(
            BalanceFinding(
                str(path),
                "",
                "error",
                "min_prompts",
                f"found {len(prompt_rows)} prompts, expected at least {args.min_prompts}",
            )
        )
    if measured_runs < args.min_measured_runs:
        findings.append(
            BalanceFinding(
                str(path),
                "",
                "error",
                "min_measured_runs",
                f"found {measured_runs} measured runs, expected at least {args.min_measured_runs}",
            )
        )
    if delta > args.max_successful_run_delta:
        findings.append(
            BalanceFinding(
                str(path),
                "",
                "error",
                "successful_run_delta",
                f"successful run delta {delta} exceeds limit {args.max_successful_run_delta}",
            )
        )
    for row in prompt_rows:
        if row.successful_runs < args.min_successful_runs_per_prompt:
            findings.append(
                BalanceFinding(
                    str(path),
                    row.prompt,
                    "error",
                    "min_successful_runs_per_prompt",
                    f"{row.prompt}: {row.successful_runs} successful runs, expected at least {args.min_successful_runs_per_prompt}",
                )
            )
        if row.failed_runs and args.fail_on_failed_runs:
            findings.append(
                BalanceFinding(str(path), row.prompt, "error", "failed_runs", f"{row.prompt}: {row.failed_runs} failed measured runs")
            )

    artifact = ArtifactBalance(
        source=str(path),
        status="fail" if findings else "pass",
        profile=str(payload.get("profile") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        prompts=len(prompt_rows),
        measured_runs=measured_runs,
        successful_runs=successful_runs,
        failed_runs=failed_runs,
        min_successful_runs=min_success,
        max_successful_runs=max_success,
        successful_run_delta=delta,
    )
    return artifact, prompt_rows, findings


def write_csv(path: Path, rows: list[Any]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[BalanceFinding]) -> None:
    fieldnames = list(asdict(findings[0]).keys()) if findings else ["source", "prompt", "severity", "kind", "detail"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# QEMU Prompt Balance Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Artifacts: {payload['summary']['artifacts']}",
        f"- Prompts: {payload['summary']['prompts']}",
        f"- Measured runs: {payload['summary']['measured_runs']}",
        f"- Successful runs: {payload['summary']['successful_runs']}",
        f"- Findings: {len(payload['findings'])}",
        "",
        "## Artifacts",
        "",
        "| Source | Status | Prompts | Measured | Successful | Failed | Success delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["artifacts"]:
        lines.append(
            "| {source} | {status} | {prompts} | {measured_runs} | {successful_runs} | {failed_runs} | {successful_run_delta} |".format(
                **item
            )
        )
    lines.extend(["", "## Findings", ""])
    if payload["findings"]:
        for finding in payload["findings"]:
            scope = f"{finding['source']}:{finding['prompt']}" if finding["prompt"] else finding["source"]
            lines.append(f"- {finding['severity']} {finding['kind']} {scope}: {finding['detail']}")
    else:
        lines.append("No prompt balance findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[BalanceFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {"name": "holyc_qemu_prompt_balance_audit", "tests": "1", "failures": "1" if findings else "0"},
    )
    case = ET.SubElement(suite, "testcase", {"name": "prompt_balance"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} prompt balance finding(s)"})
        failure.text = "\n".join(f"{finding.kind}: {finding.source}:{finding.prompt} {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact JSON files or directories")
    parser.add_argument("--pattern", action="append", default=["qemu_prompt_bench*.json"], help="Directory glob pattern")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_prompt_balance_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-prompts", type=int, default=1)
    parser.add_argument("--min-measured-runs", type=int, default=1)
    parser.add_argument("--min-successful-runs-per-prompt", type=int, default=1)
    parser.add_argument("--max-successful-run-delta", type=int, default=0)
    parser.add_argument("--fail-on-failed-runs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = list(iter_input_files(args.inputs, args.pattern))
    artifacts: list[ArtifactBalance] = []
    prompt_rows: list[PromptBalanceRow] = []
    findings: list[BalanceFinding] = []

    for path in files:
        artifact, rows, artifact_findings = audit_artifact(path, args)
        artifacts.append(artifact)
        prompt_rows.extend(rows)
        findings.extend(artifact_findings)

    if len(artifacts) < args.min_artifacts:
        findings.append(
            BalanceFinding("", "", "error", "min_artifacts", f"found {len(artifacts)} artifacts, expected at least {args.min_artifacts}")
        )

    status = "fail" if findings else "pass"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "summary": {
            "artifacts": len(artifacts),
            "prompts": sum(artifact.prompts for artifact in artifacts),
            "measured_runs": sum(artifact.measured_runs for artifact in artifacts),
            "successful_runs": sum(artifact.successful_runs for artifact in artifacts),
            "failed_runs": sum(artifact.failed_runs for artifact in artifacts),
            "findings": len(findings),
        },
        "gates": {
            "min_artifacts": args.min_artifacts,
            "min_prompts": args.min_prompts,
            "min_measured_runs": args.min_measured_runs,
            "min_successful_runs_per_prompt": args.min_successful_runs_per_prompt,
            "max_successful_run_delta": args.max_successful_run_delta,
            "fail_on_failed_runs": args.fail_on_failed_runs,
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "prompt_rows": [asdict(row) for row in prompt_rows],
        "findings": [asdict(finding) for finding in findings],
    }

    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", prompt_rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", payload)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
