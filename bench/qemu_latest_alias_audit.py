#!/usr/bin/env python3
"""Audit QEMU latest JSON aliases against stamped artifacts.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LATEST_PATTERNS = ("qemu_prompt_bench*_latest.json",)
STAMP_RE = re.compile(r"^(?P<prefix>.+)_(?P<stamp>\d{8}T\d{6}Z)\.json$")


@dataclass(frozen=True)
class AliasRecord:
    latest_source: str
    status: str
    artifact_group: str
    stamped_source: str
    latest_generated_at: str
    stamped_generated_at: str
    latest_sha256: str
    stamped_sha256: str
    stamped_candidates: int
    error: str = ""


@dataclass(frozen=True)
class AliasFinding:
    source: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_latest_files(paths: Iterable[Path], patterns: list[str]) -> Iterable[Path]:
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


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def latest_group(path: Path) -> str:
    suffix = "_latest.json"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def stamped_candidates_for(latest: Path) -> list[Path]:
    group = latest_group(latest)
    candidates: list[tuple[str, Path]] = []
    for child in latest.parent.glob(f"{group}_*.json"):
        if child.name == latest.name:
            continue
        match = STAMP_RE.match(child.name)
        if match and match.group("prefix") == group:
            candidates.append((match.group("stamp"), child))
    return [path for _, path in sorted(candidates)]


def newest_stamped_candidate(latest: Path) -> tuple[Path | None, int]:
    candidates = stamped_candidates_for(latest)
    if not candidates:
        return None, 0
    return candidates[-1], len(candidates)


def text_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def audit_latest(path: Path) -> tuple[AliasRecord, list[AliasFinding]]:
    latest_payload, latest_error = load_json_object(path)
    if latest_payload is None:
        record = AliasRecord(str(path), "fail", latest_group(path), "", "", "", "", "", 0, latest_error)
        return record, [AliasFinding(str(path), "invalid_latest_json", latest_error)]

    stamped_path, candidate_count = newest_stamped_candidate(path)
    latest_hash = canonical_sha256(latest_payload)
    latest_generated_at = text_value(latest_payload.get("generated_at"))
    if stamped_path is None:
        record = AliasRecord(
            str(path),
            "fail",
            latest_group(path),
            "",
            latest_generated_at,
            "",
            latest_hash,
            "",
            candidate_count,
            "no stamped sibling artifact found",
        )
        return record, [AliasFinding(str(path), "missing_stamped_sibling", "no stamped sibling artifact found")]

    stamped_payload, stamped_error = load_json_object(stamped_path)
    if stamped_payload is None:
        record = AliasRecord(
            str(path),
            "fail",
            latest_group(path),
            str(stamped_path),
            latest_generated_at,
            "",
            latest_hash,
            "",
            candidate_count,
            stamped_error,
        )
        return record, [AliasFinding(str(stamped_path), "invalid_stamped_json", stamped_error)]

    stamped_hash = canonical_sha256(stamped_payload)
    stamped_generated_at = text_value(stamped_payload.get("generated_at"))
    findings: list[AliasFinding] = []
    if latest_hash != stamped_hash:
        findings.append(
            AliasFinding(
                str(path),
                "latest_alias_payload_drift",
                f"latest payload differs from newest stamped sibling {stamped_path}",
            )
        )
    if latest_generated_at != stamped_generated_at:
        findings.append(
            AliasFinding(
                str(path),
                "latest_alias_generated_at_drift",
                f"latest generated_at {latest_generated_at!r} differs from stamped {stamped_generated_at!r}",
            )
        )

    return (
        AliasRecord(
            latest_source=str(path),
            status="fail" if findings else "pass",
            artifact_group=latest_group(path),
            stamped_source=str(stamped_path),
            latest_generated_at=latest_generated_at,
            stamped_generated_at=stamped_generated_at,
            latest_sha256=latest_hash,
            stamped_sha256=stamped_hash,
            stamped_candidates=candidate_count,
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[AliasRecord], list[AliasFinding]]:
    records: list[AliasRecord] = []
    findings: list[AliasFinding] = []
    for path in iter_latest_files(paths, args.pattern):
        record, artifact_findings = audit_latest(path)
        records.append(record)
        findings.extend(artifact_findings)
    if len(records) < args.min_latest:
        findings.append(
            AliasFinding("", "min_latest", f"found {len(records)} latest artifact(s), required {args.min_latest}")
        )
    return records, findings


def build_report(records: list[AliasRecord], findings: list[AliasFinding], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "latest_artifacts": len(records),
            "passing_aliases": sum(1 for record in records if record.status == "pass"),
            "failing_aliases": sum(1 for record in records if record.status == "fail"),
            "findings": len(findings),
        },
        "config": {
            "patterns": args.pattern,
            "min_latest": args.min_latest,
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[AliasRecord]) -> None:
    fieldnames = list(AliasRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[AliasFinding]) -> None:
    fieldnames = list(AliasFinding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Latest Alias Audit",
        "",
        f"- Status: {report['status']}",
        f"- Latest artifacts: {summary['latest_artifacts']}",
        f"- Passing aliases: {summary['passing_aliases']}",
        f"- Failing aliases: {summary['failing_aliases']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Latest | Status | Newest stamped sibling | Stamped candidates |",
        "| --- | --- | --- | ---: |",
    ]
    for record in report["artifacts"]:
        lines.append(
            "| {latest_source} | {status} | {stamped_source} | {stamped_candidates} |".format(**record)
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['detail']} ({finding['source']})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    failure_count = len(report["findings"])
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_latest_alias_audit",
            "tests": "1",
            "failures": str(1 if failure_count else 0),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "latest_alias"})
    if failure_count:
        failure = ET.SubElement(case, "failure", {"message": f"{failure_count} latest alias finding(s)"})
        failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in report["findings"])
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("bench/results")])
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_LATEST_PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_latest_alias_audit_latest")
    parser.add_argument("--min-latest", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    records, findings = audit(args.paths, args)
    report = build_report(records, findings, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_junit(args.output_dir / f"{stem}_junit.xml", report)
    print(f"status={report['status']}")
    print(f"latest_artifacts={report['summary']['latest_artifacts']}")
    print(f"findings={report['summary']['findings']}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
