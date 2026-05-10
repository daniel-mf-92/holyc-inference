#!/usr/bin/env python3
"""Audit QEMU prompt benchmark artifacts for prompt-suite coverage.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_prompt_bench


@dataclass(frozen=True)
class ExpectedPrompt:
    prompt: str
    prompt_sha256: str
    prompt_bytes: int
    expected_tokens: int | None


@dataclass(frozen=True)
class PromptCoverageRow:
    source: str
    prompt: str
    prompt_sha256: str
    expected: bool
    measured_runs: int
    successful_runs: int
    failed_runs: int
    min_tokens: int | None
    max_tokens: int | None
    expected_tokens: int | None
    expected_tokens_mismatches: int
    guest_sha_mismatches: int
    guest_bytes_mismatches: int


@dataclass(frozen=True)
class ArtifactCoverage:
    source: str
    status: str
    profile: str
    model: str
    quantization: str
    prompt_suite_source: str
    prompt_suite_sha256: str
    prompt_suite_file_exists: bool
    prompt_suite_file_sha256_matches: bool | None
    expected_prompts: int
    measured_prompts: int
    missing_prompts: int
    unexpected_prompts: int
    measured_runs: int
    successful_runs: int
    failed_runs: int
    min_successful_runs_per_expected_prompt: int
    error: str = ""


@dataclass(frozen=True)
class CoverageFinding:
    source: str
    prompt: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def resolve_suite_path(path: Path, payload: dict[str, Any]) -> Path | None:
    suite = payload.get("prompt_suite")
    if not isinstance(suite, dict):
        return None
    source = suite.get("source")
    if not isinstance(source, str) or not source:
        return None
    suite_path = Path(source)
    if suite_path.is_absolute():
        return suite_path
    root = Path(__file__).resolve().parents[1]
    root_candidate = root / suite_path
    if root_candidate.exists():
        return root_candidate
    return path.parent / suite_path


def load_expected_prompts(path: Path, payload: dict[str, Any]) -> tuple[list[ExpectedPrompt], bool, bool | None, str]:
    suite = payload.get("prompt_suite")
    suite_hash = str(suite.get("suite_sha256") or "") if isinstance(suite, dict) else ""
    suite_path = resolve_suite_path(path, payload)
    if suite_path is None:
        return [], False, None, "missing prompt_suite.source"
    if not suite_path.exists():
        return [], False, None, f"prompt suite file does not exist: {suite_path}"
    try:
        cases = qemu_prompt_bench.load_prompt_cases(suite_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [], True, None, f"cannot load prompt suite: {exc}"
    calculated_hash = qemu_prompt_bench.prompt_suite_hash(cases)
    expected = [
        ExpectedPrompt(
            prompt=case.prompt_id,
            prompt_sha256=qemu_prompt_bench.prompt_hash(case.prompt),
            prompt_bytes=qemu_prompt_bench.prompt_bytes(case.prompt),
            expected_tokens=case.expected_tokens,
        )
        for case in cases
    ]
    return expected, True, (calculated_hash == suite_hash if suite_hash else None), ""


def measured_benchmarks(payload: dict[str, Any], *, include_warmups: bool) -> list[dict[str, Any]]:
    rows = payload.get("benchmarks")
    if not isinstance(rows, list):
        return []
    selected = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not include_warmups and str(row.get("phase") or "measured") != "measured":
            continue
        selected.append(row)
    return selected


def row_prompt_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("prompt") or ""), str(row.get("prompt_sha256") or row.get("guest_prompt_sha256") or "")


def successful(row: dict[str, Any]) -> bool:
    return str(row.get("exit_class") or "") == "ok" and not bool(row.get("timed_out")) and row.get("failure_reason") in (None, "")


def build_prompt_rows(source: Path, expected: list[ExpectedPrompt], benchmark_rows: list[dict[str, Any]]) -> list[PromptCoverageRow]:
    expected_by_key = {(prompt.prompt, prompt.prompt_sha256): prompt for prompt in expected}
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in benchmark_rows:
        rows_by_key[row_prompt_key(row)].append(row)

    all_keys = sorted(set(expected_by_key) | set(rows_by_key))
    prompt_rows: list[PromptCoverageRow] = []
    for prompt_name, prompt_sha in all_keys:
        grouped = rows_by_key.get((prompt_name, prompt_sha), [])
        expected_prompt = expected_by_key.get((prompt_name, prompt_sha))
        tokens = [int_or_none(row.get("tokens")) for row in grouped]
        token_values = [token for token in tokens if token is not None]
        expected_tokens = expected_prompt.expected_tokens if expected_prompt else None
        prompt_rows.append(
            PromptCoverageRow(
                source=str(source),
                prompt=prompt_name,
                prompt_sha256=prompt_sha,
                expected=expected_prompt is not None,
                measured_runs=len(grouped),
                successful_runs=sum(1 for row in grouped if successful(row)),
                failed_runs=sum(1 for row in grouped if not successful(row)),
                min_tokens=min(token_values) if token_values else None,
                max_tokens=max(token_values) if token_values else None,
                expected_tokens=expected_tokens,
                expected_tokens_mismatches=sum(1 for row in grouped if row.get("expected_tokens_match") is False),
                guest_sha_mismatches=sum(1 for row in grouped if row.get("guest_prompt_sha256_match") is False),
                guest_bytes_mismatches=sum(1 for row in grouped if row.get("guest_prompt_bytes_match") is False),
            )
        )
    return prompt_rows


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactCoverage, list[PromptCoverageRow], list[CoverageFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        artifact = ArtifactCoverage(str(path), "fail", "", "", "", "", "", False, None, 0, 0, 0, 0, 0, 0, 0, 0, error)
        return artifact, [], [CoverageFinding(str(path), "", "invalid_artifact", error)]

    suite = payload.get("prompt_suite") if isinstance(payload.get("prompt_suite"), dict) else {}
    expected, suite_exists, suite_hash_matches, suite_error = load_expected_prompts(path, payload)
    rows = measured_benchmarks(payload, include_warmups=args.include_warmups)
    prompt_rows = build_prompt_rows(path, expected, rows)
    expected_count = len(expected)
    measured_expected = [row for row in prompt_rows if row.expected and row.measured_runs > 0]
    successful_expected = [row.successful_runs for row in prompt_rows if row.expected]
    min_success = min(successful_expected) if successful_expected else 0
    missing_count = sum(1 for row in prompt_rows if row.expected and row.measured_runs == 0)
    unexpected_count = sum(1 for row in prompt_rows if not row.expected)

    artifact = ArtifactCoverage(
        source=str(path),
        status="pass",
        profile=str(payload.get("profile") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        prompt_suite_source=str(suite.get("source") or ""),
        prompt_suite_sha256=str(suite.get("suite_sha256") or ""),
        prompt_suite_file_exists=suite_exists,
        prompt_suite_file_sha256_matches=suite_hash_matches,
        expected_prompts=expected_count,
        measured_prompts=len(measured_expected),
        missing_prompts=missing_count,
        unexpected_prompts=unexpected_count,
        measured_runs=len(rows),
        successful_runs=sum(1 for row in rows if successful(row)),
        failed_runs=sum(1 for row in rows if not successful(row)),
        min_successful_runs_per_expected_prompt=min_success,
    )
    findings = evaluate_artifact(artifact, prompt_rows, suite_error, args)
    if findings:
        artifact = ArtifactCoverage(**{**asdict(artifact), "status": "fail"})
    return artifact, prompt_rows, findings


def evaluate_artifact(
    artifact: ArtifactCoverage,
    prompt_rows: list[PromptCoverageRow],
    suite_error: str,
    args: argparse.Namespace,
) -> list[CoverageFinding]:
    findings: list[CoverageFinding] = []
    if suite_error and args.require_suite_file:
        findings.append(CoverageFinding(artifact.source, "", "prompt_suite_file", suite_error))
    if artifact.prompt_suite_file_sha256_matches is False:
        findings.append(
            CoverageFinding(artifact.source, "", "prompt_suite_hash", "recorded prompt_suite.suite_sha256 does not match suite file")
        )
    if artifact.expected_prompts < args.min_prompts:
        findings.append(
            CoverageFinding(
                artifact.source,
                "",
                "min_prompts",
                f"expected prompt count {artifact.expected_prompts} is below minimum {args.min_prompts}",
            )
        )
    for row in prompt_rows:
        if row.expected and row.measured_runs == 0:
            findings.append(CoverageFinding(row.source, row.prompt, "missing_prompt", "expected prompt has no measured runs"))
        if row.expected and row.successful_runs < args.min_runs_per_prompt:
            findings.append(
                CoverageFinding(
                    row.source,
                    row.prompt,
                    "min_runs_per_prompt",
                    f"successful runs {row.successful_runs} is below minimum {args.min_runs_per_prompt}",
                )
            )
        if args.fail_on_unexpected_prompts and not row.expected:
            findings.append(CoverageFinding(row.source, row.prompt, "unexpected_prompt", "measured prompt is not in the suite"))
        if args.require_success and row.failed_runs:
            findings.append(CoverageFinding(row.source, row.prompt, "failed_runs", f"prompt has {row.failed_runs} failed run(s)"))
        if row.expected_tokens_mismatches:
            findings.append(
                CoverageFinding(row.source, row.prompt, "expected_tokens_mismatch", f"{row.expected_tokens_mismatches} run(s) mismatched expected tokens")
            )
        if row.guest_sha_mismatches:
            findings.append(
                CoverageFinding(row.source, row.prompt, "guest_prompt_sha_mismatch", f"{row.guest_sha_mismatches} run(s) mismatched guest prompt SHA")
            )
        if row.guest_bytes_mismatches:
            findings.append(
                CoverageFinding(row.source, row.prompt, "guest_prompt_bytes_mismatch", f"{row.guest_bytes_mismatches} run(s) mismatched guest prompt bytes")
            )
    return findings


def build_report(artifacts: list[ArtifactCoverage], prompt_rows: list[PromptCoverageRow], findings: list[CoverageFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "expected_prompts": sum(artifact.expected_prompts for artifact in artifacts),
            "measured_prompts": sum(artifact.measured_prompts for artifact in artifacts),
            "measured_runs": sum(artifact.measured_runs for artifact in artifacts),
            "successful_runs": sum(artifact.successful_runs for artifact in artifacts),
            "failed_runs": sum(artifact.failed_runs for artifact in artifacts),
            "findings": len(findings),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "prompt_rows": [asdict(row) for row in prompt_rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Prompt Coverage Audit",
        "",
        f"- Status: {report['status']}",
        f"- Artifacts: {summary['artifacts']}",
        f"- Expected prompts: {summary['expected_prompts']}",
        f"- Measured prompts: {summary['measured_prompts']}",
        f"- Successful runs: {summary['successful_runs']}",
        f"- Failed runs: {summary['failed_runs']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Artifact | Status | Profile | Model | Quantization | Expected prompts | Measured prompts | Min successful runs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for artifact in report["artifacts"]:
        lines.append(
            "| {source} | {status} | {profile} | {model} | {quantization} | {expected_prompts} | {measured_prompts} | {min_successful_runs_per_expected_prompt} |".format(
                **artifact
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            prompt = f" {finding['prompt']}" if finding["prompt"] else ""
            lines.append(f"- {finding['kind']}: {finding['source']}{prompt} {finding['detail']}".strip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[CoverageFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_coverage_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            name = f"{finding.kind}:{finding.source}:{finding.prompt}".rstrip(":")
            case = ET.SubElement(suite, "testcase", {"name": name})
            failure = ET.SubElement(case, "failure", {"message": finding.detail})
            failure.text = finding.detail
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_prompt_coverage"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU prompt benchmark JSON files or directories")
    parser.add_argument("--pattern", action="append", default=["qemu_prompt_bench*_latest.json"], help="Directory glob for benchmark artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_prompt_coverage_audit_latest")
    parser.add_argument("--include-warmups", action="store_true", help="Include warmup rows in coverage checks")
    parser.add_argument("--require-suite-file", action="store_true", help="Fail when prompt_suite.source cannot be loaded")
    parser.add_argument("--require-success", action="store_true", help="Fail on any failed measured run")
    parser.add_argument("--fail-on-unexpected-prompts", action="store_true", help="Fail when artifact contains prompts outside the suite")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-prompts", type=int, default=1)
    parser.add_argument("--min-runs-per-prompt", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts: list[ArtifactCoverage] = []
    prompt_rows: list[PromptCoverageRow] = []
    findings: list[CoverageFinding] = []
    for path in sorted(iter_input_files(args.inputs, args.pattern)):
        artifact, artifact_rows, artifact_findings = audit_artifact(path, args)
        artifacts.append(artifact)
        prompt_rows.extend(artifact_rows)
        findings.extend(artifact_findings)
    if len(artifacts) < args.min_artifacts:
        findings.append(
            CoverageFinding(
                "",
                "",
                "min_artifacts",
                f"benchmark artifact count {len(artifacts)} is below minimum {args.min_artifacts}",
            )
        )
    report = build_report(artifacts, prompt_rows, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", artifacts, list(ArtifactCoverage.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_prompts.csv", prompt_rows, list(PromptCoverageRow.__dataclass_fields__))
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
