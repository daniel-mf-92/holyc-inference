#!/usr/bin/env python3
"""Audit HolyC-vs-llama prediction stream pairing before eval comparison.

This host-side tool reads local JSONL prediction artifacts only. It verifies
that both engines evaluated the same records, in the same order when requested,
with matching dataset/split/model/quantization and prompt/choice hashes when
those fields are present.
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


ID_FIELDS = ("id", "record_id", "prompt_id", "sample_id")
COMPARE_FIELDS = (
    "dataset",
    "split",
    "model",
    "model_sha256",
    "tokenizer_sha256",
    "quantization",
    "prompt_template_sha256",
    "prompt_sha256",
    "prompt_hash",
    "choices_sha256",
    "choice_hash",
    "input_sha256",
)


@dataclass(frozen=True)
class PairRow:
    record_id: str
    holyc_row: int
    llama_row: int
    order_match: bool
    dataset: str
    split: str
    model: str
    quantization: str
    holyc_prediction: str
    llama_prediction: str
    holyc_score_count: int
    llama_score_count: int


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
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
    return str(value)


def record_id(row: dict[str, Any]) -> str:
    for field in ID_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def prediction_text(row: dict[str, Any]) -> str:
    if "prediction" in row:
        return stable_text(row.get("prediction"))
    scores = row.get("scores")
    if isinstance(scores, list) and scores:
        try:
            best = max(range(len(scores)), key=lambda index: float(scores[index]))
        except (TypeError, ValueError):
            return ""
        return str(best)
    return ""


def score_count(row: dict[str, Any]) -> int:
    scores = row.get("scores")
    return len(scores) if isinstance(scores, list) else 0


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[Finding]]:
    rows: list[dict[str, Any]] = []
    findings: list[Finding] = []
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                findings.append(Finding("error", "invalid_jsonl", f"{row_number}", "", "", "", f"{path}:{row_number}: {exc}"))
                continue
            if not isinstance(payload, dict):
                findings.append(Finding("error", "non_object_row", f"{row_number}", "", "", "", f"{path}:{row_number}: row is not an object"))
                continue
            payload["_row_number"] = row_number
            rows.append(payload)
    return rows, findings


def index_rows(engine: str, rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str], list[Finding]]:
    indexed: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    findings: list[Finding] = []
    for row in rows:
        row_number = int(row.get("_row_number", 0))
        rid = record_id(row)
        if not rid:
            findings.append(Finding("error", "missing_record_id", f"{row_number}", "", engine if engine == "holyc" else "", engine if engine == "llama" else "", "row has no stable id field"))
            continue
        if rid in indexed:
            findings.append(Finding("error", "duplicate_record_id", rid, "", engine if engine == "holyc" else "", engine if engine == "llama" else "", f"duplicate {engine} record id"))
            continue
        indexed[rid] = row
        order.append(rid)
    return indexed, order, findings


def comparable_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value in (None, "") and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get(field)
    return stable_text(value)


def build_pair_row(record_id_value: str, holyc: dict[str, Any], llama: dict[str, Any], order_match: bool) -> PairRow:
    return PairRow(
        record_id=record_id_value,
        holyc_row=int(holyc.get("_row_number", 0)),
        llama_row=int(llama.get("_row_number", 0)),
        order_match=order_match,
        dataset=comparable_value(holyc, "dataset") or comparable_value(llama, "dataset"),
        split=comparable_value(holyc, "split") or comparable_value(llama, "split"),
        model=comparable_value(holyc, "model") or comparable_value(llama, "model"),
        quantization=comparable_value(holyc, "quantization") or comparable_value(llama, "quantization"),
        holyc_prediction=prediction_text(holyc),
        llama_prediction=prediction_text(llama),
        holyc_score_count=score_count(holyc),
        llama_score_count=score_count(llama),
    )


def audit_pairing(
    holyc_path: Path,
    llama_path: Path,
    *,
    min_records: int,
    require_same_order: bool,
    require_predictions: bool,
) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        holyc_rows, load_findings = load_jsonl(holyc_path)
        findings.extend(load_findings)
    except OSError as exc:
        return {"generated_at": iso_now(), "status": "fail", "summary": {"paired_records": 0}, "pairs": [], "findings": [asdict(Finding("error", "unreadable_holyc", "", "", "", "", str(exc)))]}
    try:
        llama_rows, load_findings = load_jsonl(llama_path)
        findings.extend(load_findings)
    except OSError as exc:
        return {"generated_at": iso_now(), "status": "fail", "summary": {"paired_records": 0}, "pairs": [], "findings": [asdict(Finding("error", "unreadable_llama", "", "", "", "", str(exc)))]}

    holyc_index, holyc_order, index_findings = index_rows("holyc", holyc_rows)
    findings.extend(index_findings)
    llama_index, llama_order, index_findings = index_rows("llama", llama_rows)
    findings.extend(index_findings)

    holyc_ids = set(holyc_index)
    llama_ids = set(llama_index)
    for rid in sorted(holyc_ids - llama_ids):
        findings.append(Finding("error", "missing_llama_record", rid, "", "", "", "HolyC record has no llama counterpart"))
    for rid in sorted(llama_ids - holyc_ids):
        findings.append(Finding("error", "missing_holyc_record", rid, "", "", "", "llama record has no HolyC counterpart"))

    pair_rows: list[PairRow] = []
    for rid in [record for record in holyc_order if record in llama_index]:
        holyc = holyc_index[rid]
        llama = llama_index[rid]
        order_match = holyc_order.index(rid) == llama_order.index(rid) if rid in llama_order else False
        pair_rows.append(build_pair_row(rid, holyc, llama, order_match))
        if require_same_order and not order_match:
            findings.append(Finding("error", "order_mismatch", rid, "", str(holyc.get("_row_number", "")), str(llama.get("_row_number", "")), "paired records appear at different ordinal positions"))
        for field in COMPARE_FIELDS:
            holyc_value = comparable_value(holyc, field)
            llama_value = comparable_value(llama, field)
            if holyc_value and llama_value and holyc_value != llama_value:
                findings.append(Finding("error", "metadata_mismatch", rid, field, holyc_value, llama_value, "paired metadata differs"))
        if score_count(holyc) and score_count(llama) and score_count(holyc) != score_count(llama):
            findings.append(Finding("error", "score_count_mismatch", rid, "scores", str(score_count(holyc)), str(score_count(llama)), "candidate score vector lengths differ"))
        if require_predictions and (not prediction_text(holyc) or not prediction_text(llama)):
            findings.append(Finding("error", "missing_prediction", rid, "prediction", prediction_text(holyc), prediction_text(llama), "paired row lacks a prediction or score-derived prediction"))

    if len(pair_rows) < min_records:
        findings.append(Finding("error", "insufficient_paired_records", "", "paired_records", str(len(pair_rows)), str(min_records), "not enough paired records for comparison"))

    status = "fail" if findings else "pass"
    return {
        "generated_at": iso_now(),
        "status": status,
        "inputs": {"holyc": str(holyc_path), "llama": str(llama_path)},
        "summary": {
            "holyc_records": len(holyc_index),
            "llama_records": len(llama_index),
            "paired_records": len(pair_rows),
            "findings": len(findings),
        },
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
        "# Eval Pairing Audit",
        "",
        f"- status: {payload['status']}",
        f"- holyc_records: {summary['holyc_records']}",
        f"- llama_records: {summary['llama_records']}",
        f"- paired_records: {summary['paired_records']}",
        f"- findings: {summary['findings']}",
        "",
    ]
    if payload["findings"]:
        lines.append("## Findings")
        lines.append("")
        for finding in payload["findings"][:20]:
            lines.append(f"- {finding['kind']} record={finding['record_id'] or '-'} field={finding['field'] or '-'}: {finding['detail']}")
    else:
        lines.append("No eval pairing findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    failures = 1 if payload["findings"] else 0
    testsuite = ET.Element("testsuite", name="holyc_eval_pairing_audit", tests="1", failures=str(failures))
    testcase = ET.SubElement(testsuite, "testcase", name="eval_pairing")
    if failures:
        failure = ET.SubElement(testcase, "failure", message=f"{len(payload['findings'])} eval pairing finding(s)")
        failure.text = "\n".join(f"{finding['kind']} {finding['record_id']} {finding['detail']}" for finding in payload["findings"])
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holyc", required=True, type=Path, help="HolyC prediction JSONL")
    parser.add_argument("--llama", required=True, type=Path, help="llama.cpp prediction JSONL")
    parser.add_argument("--output", type=Path, default=Path("bench/results/eval_pairing_audit_latest.json"))
    parser.add_argument("--markdown", type=Path, default=Path("bench/results/eval_pairing_audit_latest.md"))
    parser.add_argument("--csv", type=Path, default=Path("bench/results/eval_pairing_audit_latest.csv"))
    parser.add_argument("--findings-csv", type=Path, default=Path("bench/results/eval_pairing_audit_findings_latest.csv"))
    parser.add_argument("--junit", type=Path, default=Path("bench/results/eval_pairing_audit_junit_latest.xml"))
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--require-same-order", action="store_true")
    parser.add_argument("--require-predictions", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_pairing(
        args.holyc,
        args.llama,
        min_records=args.min_records,
        require_same_order=args.require_same_order,
        require_predictions=args.require_predictions,
    )
    write_json(args.output, payload)
    write_markdown(args.markdown, payload)
    write_csv(args.csv, payload["pairs"], list(PairRow.__dataclass_fields__))
    write_csv(args.findings_csv, payload["findings"], list(Finding.__dataclass_fields__))
    write_junit(args.junit, payload)
    if args.fail_on_findings and payload["findings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
