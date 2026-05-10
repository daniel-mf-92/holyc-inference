#!/usr/bin/env python3
"""Audit eval prediction streams for apples-to-apples model identity evidence.

This host-side tool reads local HolyC and llama.cpp prediction JSONL artifacts
only. It verifies that paired prediction rows expose stable, matching model,
tokenizer, and quantization identity metadata before quality comparisons are
trusted. It never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ID_FIELDS = ("id", "record_id", "prompt_id", "sample_id")
IDENTITY_ALIASES = {
    "model": ("model", "model_id", "model_name"),
    "model_sha256": ("model_sha256", "weights_sha256", "model_file_sha256", "gguf_sha256"),
    "tokenizer_sha256": ("tokenizer_sha256", "vocab_sha256", "tokenizer_file_sha256"),
    "quantization": ("quantization", "quant", "quant_type"),
    "quantization_sha256": ("quantization_sha256", "quant_sha256", "weights_quant_sha256"),
    "prompt_template_sha256": ("prompt_template_sha256", "prompt_template_hash", "template_sha256"),
}
DEFAULT_REQUIRED_FIELDS = ("model_sha256", "tokenizer_sha256", "quantization")


@dataclass(frozen=True)
class IdentitySummary:
    source: str
    field: str
    rows: int
    present: int
    missing: int
    distinct_values: int
    values: str


@dataclass(frozen=True)
class PairIdentity:
    record_id: str
    holyc_row: int
    llama_row: int
    compared_fields: int
    mismatched_fields: int
    missing_required_fields: int


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    source: str
    record_id: str
    field: str
    holyc: str
    llama: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).strip()


def metadata_value(row: dict[str, Any], aliases: Iterable[str]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in aliases:
        value = row.get(key)
        if value in (None, ""):
            value = metadata.get(key)
        text = stable_text(value)
        if text:
            return text
    return ""


def identity_values(row: dict[str, Any]) -> dict[str, str]:
    return {field: metadata_value(row, aliases) for field, aliases in IDENTITY_ALIASES.items()}


def record_id(row: dict[str, Any]) -> str:
    for field in ID_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def load_jsonl(path: Path, source: str) -> tuple[list[dict[str, Any]], list[Finding]]:
    rows: list[dict[str, Any]] = []
    findings: list[Finding] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        return rows, [Finding("error", "unreadable_predictions", source, "", "", "", "", str(exc))]
    with handle:
        for row_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                findings.append(Finding("error", "invalid_jsonl", source, str(row_number), "", "", "", str(exc)))
                continue
            if not isinstance(payload, dict):
                findings.append(Finding("error", "non_object_row", source, str(row_number), "", "", "", "row is not a JSON object"))
                continue
            payload["_row_number"] = row_number
            rows.append(payload)
    return rows, findings


def index_rows(source: str, rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[Finding]]:
    indexed: dict[str, dict[str, Any]] = {}
    findings: list[Finding] = []
    for row in rows:
        row_number = str(row.get("_row_number", ""))
        rid = record_id(row)
        if not rid:
            findings.append(Finding("error", "missing_record_id", source, row_number, "", "", "", "row has no stable id field"))
            continue
        if rid in indexed:
            findings.append(Finding("error", "duplicate_record_id", source, rid, "", "", "", "duplicate prediction id"))
            continue
        indexed[rid] = row
    return indexed, findings


def summarize_source(
    source: str,
    indexed: dict[str, dict[str, Any]],
    required_fields: set[str],
    findings: list[Finding],
) -> list[IdentitySummary]:
    summaries: list[IdentitySummary] = []
    for field in IDENTITY_ALIASES:
        values = [identity_values(row)[field] for row in indexed.values()]
        present_values = [value for value in values if value]
        distinct = sorted(set(present_values))
        missing = len(values) - len(present_values)
        if field in required_fields and missing:
            findings.append(
                Finding(
                    "error",
                    "missing_required_identity",
                    source,
                    "",
                    field,
                    "",
                    "",
                    f"{missing} of {len(values)} row(s) lack required {field} metadata",
                )
            )
        if field in required_fields and len(distinct) > 1:
            findings.append(
                Finding(
                    "error",
                    "identity_drift",
                    source,
                    "",
                    field,
                    "",
                    "",
                    f"required {field} has {len(distinct)} distinct values within one prediction stream",
                )
            )
        summaries.append(
            IdentitySummary(
                source=source,
                field=field,
                rows=len(values),
                present=len(present_values),
                missing=missing,
                distinct_values=len(distinct),
                values=";".join(distinct[:8]),
            )
        )
    return summaries


def audit_identity(
    holyc_path: Path,
    llama_path: Path,
    *,
    min_records: int,
    required_fields: set[str],
) -> dict[str, Any]:
    findings: list[Finding] = []
    holyc_rows, load_findings = load_jsonl(holyc_path, "holyc")
    findings.extend(load_findings)
    llama_rows, load_findings = load_jsonl(llama_path, "llama.cpp")
    findings.extend(load_findings)

    holyc_index, index_findings = index_rows("holyc", holyc_rows)
    findings.extend(index_findings)
    llama_index, index_findings = index_rows("llama.cpp", llama_rows)
    findings.extend(index_findings)

    summaries = [
        *summarize_source("holyc", holyc_index, required_fields, findings),
        *summarize_source("llama.cpp", llama_index, required_fields, findings),
    ]

    holyc_ids = set(holyc_index)
    llama_ids = set(llama_index)
    for rid in sorted(holyc_ids - llama_ids):
        findings.append(Finding("error", "missing_llama_record", "paired", rid, "", "", "", "HolyC row has no llama.cpp counterpart"))
    for rid in sorted(llama_ids - holyc_ids):
        findings.append(Finding("error", "missing_holyc_record", "paired", rid, "", "", "", "llama.cpp row has no HolyC counterpart"))

    pair_rows: list[PairIdentity] = []
    for rid in sorted(holyc_ids & llama_ids):
        holyc_identity = identity_values(holyc_index[rid])
        llama_identity = identity_values(llama_index[rid])
        compared = 0
        mismatched = 0
        missing_required = 0
        for field in IDENTITY_ALIASES:
            holyc_value = holyc_identity[field]
            llama_value = llama_identity[field]
            if field in required_fields and (not holyc_value or not llama_value):
                missing_required += 1
            if holyc_value and llama_value:
                compared += 1
                if holyc_value != llama_value:
                    mismatched += 1
                    findings.append(
                        Finding(
                            "error",
                            "paired_identity_mismatch",
                            "paired",
                            rid,
                            field,
                            holyc_value,
                            llama_value,
                            "paired HolyC and llama.cpp rows use different identity metadata",
                        )
                    )
        pair_rows.append(
            PairIdentity(
                record_id=rid,
                holyc_row=int(holyc_index[rid].get("_row_number", 0)),
                llama_row=int(llama_index[rid].get("_row_number", 0)),
                compared_fields=compared,
                mismatched_fields=mismatched,
                missing_required_fields=missing_required,
            )
        )

    if len(pair_rows) < min_records:
        findings.append(
            Finding(
                "error",
                "insufficient_paired_records",
                "paired",
                "",
                "paired_records",
                str(len(pair_rows)),
                str(min_records),
                "not enough paired records for an identity audit",
            )
        )

    return {
        "generated_at": iso_now(),
        "tool": "eval_artifact_identity_audit",
        "status": "fail" if any(finding.severity == "error" for finding in findings) else "pass",
        "inputs": {"holyc": str(holyc_path), "llama": str(llama_path)},
        "required_fields": sorted(required_fields),
        "summary": {
            "holyc_records": len(holyc_index),
            "llama_records": len(llama_index),
            "paired_records": len(pair_rows),
            "identity_fields": len(IDENTITY_ALIASES),
            "findings": len(findings),
        },
        "identity": [asdict(row) for row in summaries],
        "pairs": [asdict(row) for row in pair_rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Eval Artifact Identity Audit",
        "",
        f"- status: {payload['status']}",
        f"- holyc_records: {summary['holyc_records']}",
        f"- llama_records: {summary['llama_records']}",
        f"- paired_records: {summary['paired_records']}",
        f"- required_fields: {', '.join(payload['required_fields'])}",
        f"- findings: {summary['findings']}",
        "",
        "## Identity Coverage",
        "",
        "| Source | Field | Present | Missing | Distinct |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["identity"]:
        lines.append(f"| {row['source']} | {row['field']} | {row['present']} | {row['missing']} | {row['distinct_values']} |")
    lines.append("")
    if payload["findings"]:
        lines.append("## Findings")
        lines.append("")
        for finding in payload["findings"][:20]:
            lines.append(
                f"- {finding['kind']} source={finding['source']} record={finding['record_id'] or '-'} "
                f"field={finding['field'] or '-'}: {finding['detail']}"
            )
    else:
        lines.append("No eval artifact identity findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    failures = 1 if payload["status"] == "fail" else 0
    testsuite = ET.Element("testsuite", name="holyc_eval_artifact_identity_audit", tests="1", failures=str(failures))
    testcase = ET.SubElement(testsuite, "testcase", name="eval_artifact_identity")
    if failures:
        failure = ET.SubElement(testcase, "failure", message=f"{len(payload['findings'])} identity finding(s)")
        failure.text = "\n".join(
            f"{finding['kind']} {finding['source']} {finding['record_id']} {finding['field']} {finding['detail']}"
            for finding in payload["findings"]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def output_paths(output_dir: Path, output_stem: str) -> dict[str, Path]:
    return {
        "json": output_dir / f"{output_stem}.json",
        "md": output_dir / f"{output_stem}.md",
        "identity_csv": output_dir / f"{output_stem}.csv",
        "pairs_csv": output_dir / f"{output_stem}_pairs.csv",
        "findings_csv": output_dir / f"{output_stem}_findings.csv",
        "junit": output_dir / f"{output_stem}_junit.xml",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holyc", required=True, type=Path, help="HolyC prediction JSONL")
    parser.add_argument("--llama", required=True, type=Path, help="llama.cpp prediction JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_artifact_identity_audit_latest")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument(
        "--required-field",
        action="append",
        choices=sorted(IDENTITY_ALIASES),
        help="Canonical identity field that must be present and stable; repeatable. Defaults to model_sha256, tokenizer_sha256, quantization.",
    )
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required_fields = set(args.required_field or DEFAULT_REQUIRED_FIELDS)
    payload = audit_identity(args.holyc, args.llama, min_records=args.min_records, required_fields=required_fields)
    paths = output_paths(args.output_dir, args.output_stem)
    write_json(paths["json"], payload)
    write_markdown(paths["md"], payload)
    write_csv(paths["identity_csv"], payload["identity"], list(IdentitySummary.__dataclass_fields__))
    write_csv(paths["pairs_csv"], payload["pairs"], list(PairIdentity.__dataclass_fields__))
    write_csv(paths["findings_csv"], payload["findings"], list(Finding.__dataclass_fields__))
    write_junit(paths["junit"], payload)
    if args.fail_on_findings and payload["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
