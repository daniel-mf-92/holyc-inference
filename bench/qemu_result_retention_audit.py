#!/usr/bin/env python3
"""Audit QEMU benchmark latest JSONs for timestamped retention artifacts.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TIMESTAMP_RE = re.compile(r"_(\d{8}T\d{6}Z)\.json$")


@dataclass(frozen=True)
class RetentionRecord:
    source: str
    status: str
    generated_at: str
    expected_history: str
    expected_history_present: bool
    latest_sha256: str
    history_sha256: str
    hashes_match: bool
    history_count: int
    newest_history: str
    error: str = ""


@dataclass(frozen=True)
class RetentionFinding:
    source: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generated_at_to_stamp(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def history_files_for_latest(path: Path) -> list[Path]:
    if not path.stem.endswith("_latest"):
        return []
    prefix = path.stem[: -len("_latest")]
    candidates = []
    for candidate in path.parent.glob(f"{prefix}_*.json"):
        if candidate.name == path.name:
            continue
        if TIMESTAMP_RE.search(candidate.name):
            candidates.append(candidate)
    return sorted(candidates)


def iter_latest_jsons(paths: Iterable[Path], patterns: list[str]) -> Iterable[Path]:
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


def audit_latest(path: Path) -> RetentionRecord:
    payload, error = load_json_object(path)
    if payload is None:
        return RetentionRecord(str(path), "fail", "", "", False, "", "", False, 0, "", error)
    if not path.stem.endswith("_latest"):
        return RetentionRecord(str(path), "fail", str(payload.get("generated_at") or ""), "", False, "", "", False, 0, "", "artifact name must end with _latest.json")

    generated_at = str(payload.get("generated_at") or "")
    histories = history_files_for_latest(path)
    newest_history = str(histories[-1]) if histories else ""
    try:
        stamp = generated_at_to_stamp(generated_at)
    except (TypeError, ValueError) as exc:
        return RetentionRecord(
            str(path),
            "fail",
            generated_at,
            "",
            False,
            file_sha256(path),
            "",
            False,
            len(histories),
            newest_history,
            f"invalid generated_at timestamp: {exc}",
        )

    prefix = path.stem[: -len("_latest")]
    expected_history = path.with_name(f"{prefix}_{stamp}.json")
    expected_history_present = expected_history.exists()
    latest_hash = file_sha256(path)
    history_hash = file_sha256(expected_history) if expected_history_present else ""
    hashes_match = bool(history_hash) and latest_hash == history_hash
    status = "pass" if expected_history_present and hashes_match else "fail"
    return RetentionRecord(
        source=str(path),
        status=status,
        generated_at=generated_at,
        expected_history=str(expected_history),
        expected_history_present=expected_history_present,
        latest_sha256=latest_hash,
        history_sha256=history_hash,
        hashes_match=hashes_match,
        history_count=len(histories),
        newest_history=newest_history,
    )


def evaluate(records: list[RetentionRecord], min_latest: int, min_history_per_latest: int) -> list[RetentionFinding]:
    findings: list[RetentionFinding] = []
    if len(records) < min_latest:
        findings.append(
            RetentionFinding(
                "",
                "min_latest",
                f"found {len(records)} latest QEMU benchmark artifact(s), below minimum {min_latest}",
            )
        )
    for record in records:
        if record.error:
            findings.append(RetentionFinding(record.source, "invalid_latest", record.error))
        if record.history_count < min_history_per_latest:
            findings.append(
                RetentionFinding(
                    record.source,
                    "history_count",
                    f"found {record.history_count} timestamped history artifact(s), below minimum {min_history_per_latest}",
                )
            )
        if not record.expected_history_present:
            findings.append(
                RetentionFinding(
                    record.source,
                    "missing_expected_history",
                    f"missing generated_at-matched history artifact: {record.expected_history}",
                )
            )
        elif not record.hashes_match:
            findings.append(
                RetentionFinding(
                    record.source,
                    "latest_history_mismatch",
                    f"latest hash {record.latest_sha256} does not match {record.expected_history}",
                )
            )
    return findings


def build_report(records: list[RetentionRecord], findings: list[RetentionFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "latest_artifacts": len(records),
            "findings": len(findings),
            "missing_expected_history": sum(1 for record in records if not record.expected_history_present),
            "hash_mismatches": sum(1 for record in records if record.expected_history_present and not record.hashes_match),
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[RetentionRecord]) -> None:
    fieldnames = list(RetentionRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[RetentionFinding]) -> None:
    fieldnames = list(RetentionFinding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Result Retention Audit",
        "",
        f"- Status: {report['status']}",
        f"- Latest artifacts: {summary['latest_artifacts']}",
        f"- Findings: {summary['findings']}",
        f"- Missing expected history: {summary['missing_expected_history']}",
        f"- Hash mismatches: {summary['hash_mismatches']}",
        "",
        "| Latest | Status | History count | Expected history present | Hashes match |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in report["artifacts"]:
        lines.append(
            "| {source} | {status} | {history_count} | {expected_history_present} | {hashes_match} |".format(
                **record
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['source']} {finding['detail']}".strip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[RetentionFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_result_retention_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding.kind}:{finding.source}"})
            failure = ET.SubElement(case, "failure", {"message": finding.detail})
            failure.text = finding.detail
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_result_retention"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Latest JSON files or directories to audit")
    parser.add_argument("--pattern", action="append", default=["qemu_prompt_bench*_latest.json"], help="Directory glob for latest artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_result_retention_audit_latest")
    parser.add_argument("--min-latest", type=int, default=1)
    parser.add_argument("--min-history-per-latest", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = [audit_latest(path) for path in sorted(iter_latest_jsons(args.inputs, args.pattern))]
    findings = evaluate(records, args.min_latest, args.min_history_per_latest)
    report = build_report(records, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
