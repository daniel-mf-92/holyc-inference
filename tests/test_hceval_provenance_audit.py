#!/usr/bin/env python3
"""Coverage for host-side HCEval provenance auditing."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_pack
import hceval_provenance_audit


def record(record_id: str, provenance: str = "synthetic smoke provenance") -> dataset_pack.EvalRecord:
    return dataset_pack.EvalRecord(
        record_id=record_id,
        dataset="smoke",
        split="validation",
        prompt="Choose the grounded answer.",
        choices=["guess", "evidence"],
        answer_index=1,
        provenance=provenance,
    )


def write_pack(path: Path, records: list[dataset_pack.EvalRecord]) -> None:
    dataset_pack.write_outputs(records, path, path.with_suffix(path.suffix + ".manifest.json"), "smoke", "validation")


def args_for(path: Path) -> object:
    return hceval_provenance_audit.build_parser().parse_args(["--input", str(path), "--require-manifest"])


def test_provenance_audit_passes_complete_bundle(tmp_path: Path) -> None:
    binary = tmp_path / "good.hceval"
    write_pack(binary, [record("a"), record("b", "second synthetic source")])

    report = hceval_provenance_audit.build_report(args_for(binary))

    assert report["status"] == "pass"
    assert report["record_count"] == 2
    assert report["records_with_provenance"] == 2
    assert report["provenance_coverage_pct"] == 100.0
    assert report["rows"][0]["distinct_provenance_count"] == 2


def test_provenance_audit_flags_missing_provenance_and_manifest(tmp_path: Path) -> None:
    binary = tmp_path / "bad.hceval"
    write_pack(binary, [record("missing", "")])
    binary.with_suffix(binary.suffix + ".manifest.json").unlink()

    report = hceval_provenance_audit.build_report(args_for(binary))

    assert report["status"] == "fail"
    kinds = {finding["kind"] for finding in report["findings"]}
    assert {"missing_manifest", "missing_provenance", "provenance_coverage"} <= kinds
    assert report["rows"][0]["missing_provenance_count"] == 1


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    binary = tmp_path / "single.hceval"
    write_pack(binary, [record("single")])
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    csv_path = tmp_path / "audit.csv"
    findings_csv = tmp_path / "findings.csv"
    junit = tmp_path / "audit.xml"

    status = hceval_provenance_audit.main(
        [
            "--input",
            str(binary),
            "--require-manifest",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--findings-csv",
            str(findings_csv),
            "--junit",
            str(junit),
            "--fail-on-findings",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    findings = list(csv.DictReader(findings_csv.open(encoding="utf-8", newline="")))
    junit_root = ET.parse(junit).getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert "HCEval Provenance Audit" in markdown.read_text(encoding="utf-8")
    assert rows[0]["status"] == "pass"
    assert findings == []
    assert junit_root.attrib["name"] == "holyc_hceval_provenance_audit"
    assert junit_root.attrib["failures"] == "0"
