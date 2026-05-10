#!/usr/bin/env python3
"""Audit QEMU prompt benchmark JSON summaries against summary CSV sidecars.

This host-side tool reads saved benchmark artifacts only. It never launches
QEMU and never touches the TempleOS guest.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench_latest.json",)
SKIP_FIELDS = {"scope", "prompt"}


@dataclass(frozen=True)
class SummaryArtifact:
    source: str
    status: str
    csv_path: str
    suite_rows: int
    prompt_rows: int
    json_prompts: int
    compared_values: int
    findings: int
    error: str = ""


@dataclass(frozen=True)
class SummaryFinding:
    source: str
    csv_path: str
    row: str
    field: str
    kind: str
    json_value: str
    csv_value: str
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


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]] | None, list[str], str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader), list(reader.fieldnames or []), ""
    except OSError as exc:
        return None, [], str(exc)
    except csv.Error as exc:
        return None, [], f"invalid csv: {exc}"


def summary_csv_path(source: Path) -> Path:
    return source.with_name(source.stem.replace("_latest", "_summary_latest") + ".csv")


def normalize_json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def values_match(json_value: Any, csv_value: str, tolerance_pct: float) -> bool:
    json_text = normalize_json_value(json_value)
    if json_text == csv_value:
        return True
    if json_text == "" or csv_value == "":
        return False
    try:
        json_number = float(json_text)
        csv_number = float(csv_value)
    except ValueError:
        return False
    if not math.isfinite(json_number) or not math.isfinite(csv_number):
        return False
    tolerance = max(0.000001, abs(json_number) * tolerance_pct / 100.0)
    return abs(json_number - csv_number) <= tolerance


def csv_row_label(index: int, row: dict[str, str]) -> str:
    scope = row.get("scope") or "-"
    prompt = row.get("prompt") or ""
    return f"{index}:{scope}:{prompt}" if prompt else f"{index}:{scope}"


def compare_summary_row(
    path: Path,
    csv_path: Path,
    row_label: str,
    json_row: dict[str, Any],
    csv_row: dict[str, str],
    fields: list[str],
    args: argparse.Namespace,
) -> tuple[int, list[SummaryFinding]]:
    compared = 0
    findings: list[SummaryFinding] = []
    for field in fields:
        if field in SKIP_FIELDS:
            continue
        if field not in json_row:
            if args.require_all_json_fields and csv_row.get(field, "") != "":
                findings.append(
                    SummaryFinding(
                        str(path),
                        str(csv_path),
                        row_label,
                        field,
                        "missing_json_field",
                        "",
                        csv_row.get(field, ""),
                        "CSV summary column has no matching JSON summary field",
                    )
                )
            continue
        compared += 1
        json_value = json_row.get(field)
        csv_value = csv_row.get(field, "")
        if not values_match(json_value, csv_value, args.tolerance_pct):
            findings.append(
                SummaryFinding(
                    str(path),
                    str(csv_path),
                    row_label,
                    field,
                    "value_mismatch",
                    normalize_json_value(json_value),
                    csv_value,
                    "CSV summary value differs from JSON summary value",
                )
            )
    return compared, findings


def compare_artifact(path: Path, args: argparse.Namespace) -> tuple[SummaryArtifact, list[SummaryFinding]]:
    payload, error = load_json_object(path)
    csv_path = summary_csv_path(path)
    if payload is None:
        finding = SummaryFinding(str(path), str(csv_path), "artifact", "", "load_error", "", "", error)
        return SummaryArtifact(str(path), "fail", str(csv_path), 0, 0, 0, 0, 1, error), [finding]

    suite_summary = payload.get("suite_summary")
    summaries = payload.get("summaries")
    if not isinstance(suite_summary, dict):
        detail = "suite_summary must be an object"
        finding = SummaryFinding(str(path), str(csv_path), "suite_summary", "", "json_schema", "", "", detail)
        return SummaryArtifact(str(path), "fail", str(csv_path), 0, 0, 0, 0, 1, detail), [finding]
    if not isinstance(summaries, list) or not all(isinstance(row, dict) for row in summaries):
        detail = "summaries must be a list of objects"
        finding = SummaryFinding(str(path), str(csv_path), "summaries", "", "json_schema", "", "", detail)
        return SummaryArtifact(str(path), "fail", str(csv_path), 0, 0, 0, 0, 1, detail), [finding]
    json_by_prompt = {str(row.get("prompt") or ""): row for row in summaries}

    if not csv_path.exists():
        detail = "summary CSV sidecar is missing"
        finding = SummaryFinding(str(path), str(csv_path), "csv", "", "missing_csv", "", "", detail)
        return SummaryArtifact(str(path), "fail", str(csv_path), 0, 0, len(json_by_prompt), 0, 1, detail), [finding]

    csv_rows, csv_fields, csv_error = load_csv_rows(csv_path)
    if csv_rows is None:
        finding = SummaryFinding(str(path), str(csv_path), "csv", "", "load_error", "", "", csv_error)
        return SummaryArtifact(str(path), "fail", str(csv_path), 0, 0, len(json_by_prompt), 0, 1, csv_error), [finding]

    findings: list[SummaryFinding] = []
    compared_values = 0
    suite_rows = 0
    prompt_rows = 0
    seen_prompts: set[str] = set()
    for index, row in enumerate(csv_rows, 1):
        scope = row.get("scope", "")
        label = csv_row_label(index, row)
        if scope == "suite":
            suite_rows += 1
            compared, row_findings = compare_summary_row(path, csv_path, label, suite_summary, row, csv_fields, args)
            compared_values += compared
            findings.extend(row_findings)
        elif scope == "prompt":
            prompt = row.get("prompt", "")
            prompt_rows += 1
            if prompt in seen_prompts:
                findings.append(SummaryFinding(str(path), str(csv_path), label, "prompt", "duplicate_prompt_row", prompt, prompt, "prompt appears more than once in summary CSV"))
                continue
            seen_prompts.add(prompt)
            json_row = json_by_prompt.get(prompt)
            if json_row is None:
                findings.append(SummaryFinding(str(path), str(csv_path), label, "prompt", "unexpected_prompt_row", "", prompt, "prompt summary row is absent from JSON summaries"))
                continue
            compared, row_findings = compare_summary_row(path, csv_path, label, json_row, row, csv_fields, args)
            compared_values += compared
            findings.extend(row_findings)
        else:
            findings.append(SummaryFinding(str(path), str(csv_path), label, "scope", "invalid_scope", "", scope, "scope must be suite or prompt"))

    if suite_rows != 1:
        findings.append(SummaryFinding(str(path), str(csv_path), "csv", "scope", "suite_row_count", "1", str(suite_rows), "summary CSV must contain exactly one suite row"))
    missing_prompts = sorted(set(json_by_prompt) - seen_prompts)
    for prompt in missing_prompts:
        findings.append(SummaryFinding(str(path), str(csv_path), "csv", "prompt", "missing_prompt_row", prompt, "", "JSON prompt summary is absent from summary CSV"))
    if len(csv_rows) < args.min_rows:
        findings.append(SummaryFinding(str(path), str(csv_path), "csv", "rows", "min_rows", str(args.min_rows), str(len(csv_rows)), "summary CSV has fewer rows than required"))

    status = "fail" if findings else "pass"
    return (
        SummaryArtifact(
            str(path),
            status,
            str(csv_path),
            suite_rows,
            prompt_rows,
            len(json_by_prompt),
            compared_values,
            len(findings),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[SummaryArtifact], list[SummaryFinding]]:
    artifacts: list[SummaryArtifact] = []
    findings: list[SummaryFinding] = []
    for path in iter_input_files(paths, args.pattern):
        artifact, artifact_findings = compare_artifact(path, args)
        artifacts.append(artifact)
        findings.extend(artifact_findings)
    if len(artifacts) < args.min_artifacts:
        findings.append(
            SummaryFinding(
                "-",
                "-",
                "artifacts",
                "artifacts",
                "min_artifacts",
                str(args.min_artifacts),
                str(len(artifacts)),
                "fewer benchmark artifacts were found than required",
            )
        )
    return artifacts, findings


def write_json(path: Path, artifacts: list[SummaryArtifact], findings: list[SummaryFinding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": {
            "artifacts": len(artifacts),
            "passed_artifacts": sum(1 for artifact in artifacts if artifact.status == "pass"),
            "failed_artifacts": sum(1 for artifact in artifacts if artifact.status == "fail"),
            "suite_rows": sum(artifact.suite_rows for artifact in artifacts),
            "prompt_rows": sum(artifact.prompt_rows for artifact in artifacts),
            "json_prompts": sum(artifact.json_prompts for artifact in artifacts),
            "compared_values": sum(artifact.compared_values for artifact in artifacts),
            "findings": len(findings),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, artifacts: list[SummaryArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SummaryArtifact.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(asdict(artifact))


def write_findings_csv(path: Path, findings: list[SummaryFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SummaryFinding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, artifacts: list[SummaryArtifact], findings: list[SummaryFinding]) -> None:
    status = "pass" if not findings else "fail"
    lines = [
        "# QEMU Summary Parity Audit",
        "",
        f"- status: {status}",
        f"- artifacts: {len(artifacts)}",
        f"- findings: {len(findings)}",
        "",
        "| source | status | suite rows | prompt rows | compared values | findings |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for artifact in artifacts:
        lines.append(
            f"| {artifact.source} | {artifact.status} | {artifact.suite_rows} | {artifact.prompt_rows} | {artifact.compared_values} | {artifact.findings} |"
        )
    if findings:
        lines.extend(["", "## Findings", "", "| source | row | field | kind | detail |", "| --- | --- | --- | --- | --- |"])
        for finding in findings[:50]:
            lines.append(f"| {finding.source} | {finding.row} | {finding.field} | {finding.kind} | {finding.detail} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, artifacts: list[SummaryArtifact], findings: list[SummaryFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_summary_parity_audit",
            "tests": str(max(1, len(artifacts))),
            "failures": str(len(findings)),
            "errors": "0",
        },
    )
    if artifacts:
        for artifact in artifacts:
            case = ET.SubElement(suite, "testcase", {"name": artifact.source})
            artifact_findings = [finding for finding in findings if finding.source == artifact.source]
            if artifact_findings:
                failure = ET.SubElement(case, "failure", {"message": f"{len(artifact_findings)} summary parity finding(s)"})
                failure.text = "\n".join(f"{finding.row} {finding.field}: {finding.detail}" for finding in artifact_findings[:20])
    else:
        case = ET.SubElement(suite, "testcase", {"name": "artifact_discovery"})
        failure = ET.SubElement(case, "failure", {"message": "no artifacts"})
        failure.text = "\n".join(finding.detail for finding in findings)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Benchmark JSON artifact files or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Glob pattern used when a path is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_summary_parity_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--tolerance-pct", type=float, default=0.001)
    parser.add_argument("--require-all-json-fields", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.pattern:
        args.pattern = list(DEFAULT_PATTERNS)
    artifacts, findings = audit(args.paths, args)
    output_dir = args.output_dir
    stem = args.output_stem
    write_json(output_dir / f"{stem}.json", artifacts, findings)
    write_csv(output_dir / f"{stem}.csv", artifacts)
    write_findings_csv(output_dir / f"{stem}_findings.csv", findings)
    write_markdown(output_dir / f"{stem}.md", artifacts, findings)
    write_junit(output_dir / f"{stem}_junit.xml", artifacts, findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
