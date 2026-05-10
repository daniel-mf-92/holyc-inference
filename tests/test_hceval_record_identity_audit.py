from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_pack
import hceval_record_identity_audit


def record(record_id: str, prompt: str, choices: list[str], answer_index: int) -> dataset_pack.EvalRecord:
    return dataset_pack.EvalRecord(
        record_id=record_id,
        dataset="identity-unit",
        split="validation",
        prompt=prompt,
        choices=choices,
        answer_index=answer_index,
        provenance="synthetic unit",
    )


def write_fixture(path: Path, records: list[dataset_pack.EvalRecord]) -> None:
    dataset_pack.write_outputs(records, path, path.with_suffix(path.suffix + ".manifest.json"), "identity-unit", "validation")


def parse_args(*extra: str):
    return hceval_record_identity_audit.build_parser().parse_args([*extra])


def test_audit_passes_unique_packed_records(tmp_path: Path) -> None:
    hceval = tmp_path / "clean.hceval"
    write_fixture(
        hceval,
        [
            record("clean-1", "Pick the warm color.", ["red", "blue", "green"], 0),
            record("clean-2", "Pick the cold color.", ["orange", "blue", "yellow"], 1),
        ],
    )

    artifact, rows, findings = hceval_record_identity_audit.audit_artifact(hceval)

    assert findings == []
    assert artifact.status == "pass"
    assert artifact.record_count == 2
    assert artifact.unique_record_ids == 2
    assert rows[0].record_id == "clean-1"


def test_audit_flags_duplicate_ids_and_payloads(tmp_path: Path) -> None:
    hceval = tmp_path / "dirty.hceval"
    choices = ["evaporation", "freezing", "melting", "condensation"]
    write_fixture(
        hceval,
        [
            record("dup", "Water turning into vapor is called what?", choices, 0),
            record("dup", "Water turning into vapor is called what?", choices, 0),
        ],
    )

    artifact, _rows, findings = hceval_record_identity_audit.audit_artifact(hceval)
    kinds = {finding.kind for finding in findings}

    assert artifact.status == "fail"
    assert {"duplicate_record_id", "duplicate_input_payload", "duplicate_answer_payload"} <= kinds


def test_cli_writes_identity_reports(tmp_path: Path) -> None:
    hceval = tmp_path / "clean.hceval"
    out = tmp_path / "out"
    write_fixture(hceval, [record("clean-1", "Pick the warm color.", ["red", "blue", "green"], 0)])

    status = hceval_record_identity_audit.main(
        [
            "--input",
            str(hceval),
            "--output",
            str(out / "identity.json"),
            "--markdown",
            str(out / "identity.md"),
            "--csv",
            str(out / "identity.csv"),
            "--artifacts-csv",
            str(out / "identity_artifacts.csv"),
            "--findings-csv",
            str(out / "identity_findings.csv"),
            "--junit",
            str(out / "identity_junit.xml"),
            "--fail-on-findings",
        ]
    )

    report = json.loads((out / "identity.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((out / "identity.csv").open(encoding="utf-8")))
    artifacts = list(csv.DictReader((out / "identity_artifacts.csv").open(encoding="utf-8")))
    findings = list(csv.DictReader((out / "identity_findings.csv").open(encoding="utf-8")))
    junit = ET.parse(out / "identity_junit.xml").getroot()

    assert status == 0
    assert report["status"] == "pass"
    assert rows[0]["record_id"] == "clean-1"
    assert artifacts[0]["status"] == "pass"
    assert findings == []
    assert junit.attrib["failures"] == "0"
