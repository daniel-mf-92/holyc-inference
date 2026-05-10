#!/usr/bin/env python3
"""Audit eval_compare reports for apples-to-apples artifact drift.

This host-side tool consumes existing eval_compare JSON artifacts, checks that
gold and prediction hashes are present, and flags inconsistent gold hashes for
the same dataset/split. It never launches QEMU and never touches TempleOS.
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
from typing import Any


@dataclass(frozen=True)
class EvalArtifactRecord:
    source: str
    status: str
    dataset: str
    split: str
    model: str
    quantization: str
    record_count: int
    gold_sha256: str
    holyc_predictions_sha256: str
    llama_predictions_sha256: str
    error: str = ""


@dataclass(frozen=True)
class DriftFinding:
    gate: str
    scope: str
    value: str | int
    threshold: str | int
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def iter_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*.json") if child.is_file()))
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    return sorted(files)


def load_record(path: Path) -> EvalArtifactRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return EvalArtifactRecord(str(path), "missing", "", "", "", "", 0, "", "", "", str(exc))
    except json.JSONDecodeError as exc:
        return EvalArtifactRecord(str(path), "invalid", "", "", "", "", 0, "", "", "", f"invalid json: {exc}")
    if not isinstance(payload, dict):
        return EvalArtifactRecord(str(path), "invalid", "", "", "", "", 0, "", "", "", "root must be an object")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return EvalArtifactRecord(
        source=str(path),
        status=str(payload.get("status") or "pass").lower(),
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        record_count=as_int(summary.get("record_count")),
        gold_sha256=str(payload.get("gold_sha256") or ""),
        holyc_predictions_sha256=str(payload.get("holyc_predictions_sha256") or ""),
        llama_predictions_sha256=str(payload.get("llama_predictions_sha256") or ""),
    )


def group_key(record: EvalArtifactRecord, fields: tuple[str, ...]) -> str:
    return "/".join(str(getattr(record, field)) or "<missing>" for field in fields)


def distinct_values(records: list[EvalArtifactRecord], field: str) -> set[str]:
    return {str(getattr(record, field)) for record in records if str(getattr(record, field))}


def evaluate(records: list[EvalArtifactRecord], args: argparse.Namespace) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    if len(records) < args.min_reports:
        findings.append(
            DriftFinding(
                "min_reports",
                "suite",
                len(records),
                args.min_reports,
                f"found {len(records)} report(s), below minimum {args.min_reports}",
            )
        )
    for record in records:
        if record.status in {"missing", "invalid"}:
            findings.append(DriftFinding(record.status, record.source, record.status, "valid", record.error))
            continue
        if args.fail_on_failed_reports and record.status != "pass":
            findings.append(
                DriftFinding(
                    "report_status",
                    record.source,
                    record.status,
                    "pass",
                    f"{record.source} status is {record.status}",
                )
            )
        if args.require_hashes:
            for field in ("gold_sha256", "holyc_predictions_sha256", "llama_predictions_sha256"):
                if not getattr(record, field):
                    findings.append(
                        DriftFinding(
                            f"missing_{field}",
                            record.source,
                            "",
                            "non-empty",
                            f"{record.source} is missing {field}",
                        )
                    )

    by_dataset_split: dict[str, list[EvalArtifactRecord]] = {}
    by_report_key: dict[str, list[EvalArtifactRecord]] = {}
    for record in records:
        if record.status in {"missing", "invalid"}:
            continue
        by_dataset_split.setdefault(group_key(record, ("dataset", "split")), []).append(record)
        by_report_key.setdefault(group_key(record, ("dataset", "split", "model", "quantization")), []).append(record)

    for scope, grouped_records in sorted(by_dataset_split.items()):
        hashes = distinct_values(grouped_records, "gold_sha256")
        if len(hashes) > 1:
            findings.append(
                DriftFinding(
                    "gold_sha256_drift",
                    scope,
                    ",".join(sorted(hashes)),
                    "one gold_sha256",
                    f"{scope} has {len(hashes)} distinct gold dataset hashes",
                )
            )

    if args.fail_on_duplicate_key_drift:
        for scope, grouped_records in sorted(by_report_key.items()):
            signatures = {
                (
                    record.gold_sha256,
                    record.holyc_predictions_sha256,
                    record.llama_predictions_sha256,
                    record.record_count,
                )
                for record in grouped_records
            }
            if len(signatures) > 1:
                findings.append(
                    DriftFinding(
                        "duplicate_report_key_drift",
                        scope,
                        len(signatures),
                        1,
                        f"{scope} has {len(signatures)} distinct artifact signatures",
                    )
                )
    return findings


def build_report(records: list[EvalArtifactRecord], findings: list[DriftFinding]) -> dict[str, Any]:
    dataset_split_count = len({(record.dataset, record.split) for record in records if record.dataset or record.split})
    report_key_count = len(
        {
            (record.dataset, record.split, record.model, record.quantization)
            for record in records
            if record.dataset or record.split or record.model or record.quantization
        }
    )
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "reports": len(records),
            "records": sum(record.record_count for record in records),
            "dataset_splits": dataset_split_count,
            "report_keys": report_key_count,
            "findings": len(findings),
        },
        "reports": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[EvalArtifactRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvalArtifactRecord.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[DriftFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DriftFinding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Eval Artifact Drift Audit",
        "",
        f"- Status: {report['status']}",
        f"- Reports: {summary['reports']}",
        f"- Eval rows: {summary['records']}",
        f"- Dataset/splits: {summary['dataset_splits']}",
        f"- Report keys: {summary['report_keys']}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(f"- {finding['gate']}: {finding['message']}")
    else:
        lines.append("No eval artifact drift findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[DriftFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element(
        "testsuite",
        {"name": "holyc_eval_artifact_drift_audit", "tests": "1", "failures": "1" if findings else "0"},
    )
    testcase = ET.SubElement(root, "testcase", {"name": "eval_artifact_drift"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"type": "eval_artifact_drift_failure"})
        failure.text = "\n".join(finding.message for finding in findings)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="eval_compare JSON files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_artifact_drift_audit_latest")
    parser.add_argument("--min-reports", type=int, default=1)
    parser.add_argument("--require-hashes", action="store_true")
    parser.add_argument("--fail-on-failed-reports", action="store_true")
    parser.add_argument("--fail-on-duplicate-key-drift", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    records = [load_record(path) for path in iter_input_files(args.inputs)]
    findings = evaluate(records, args)
    report = build_report(records, findings)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
