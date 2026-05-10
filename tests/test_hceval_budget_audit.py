from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_pack
import hceval_budget_audit


def write_fixture(path: Path, manifest: Path) -> None:
    records = [
        dataset_pack.EvalRecord(
            record_id="smoke-1",
            dataset="smoke-eval",
            split="validation",
            prompt="Pick the warm color.",
            choices=["red", "blue", "green", "black"],
            answer_index=0,
            provenance="synthetic smoke",
        ),
        dataset_pack.EvalRecord(
            record_id="smoke-2",
            dataset="smoke-eval",
            split="validation",
            prompt="Pick the cold color.",
            choices=["orange", "blue", "yellow", "red"],
            answer_index=1,
            provenance="synthetic smoke",
        ),
    ]
    dataset_pack.write_outputs(records, path, manifest, "smoke-eval", "validation")


def parse_args(*extra: str):
    return hceval_budget_audit.build_parser().parse_args(["input.hceval", *extra])


def test_audit_passes_budgeted_hceval_with_manifest(tmp_path: Path) -> None:
    hceval = tmp_path / "input.hceval"
    manifest = tmp_path / "input.manifest.json"
    write_fixture(hceval, manifest)

    args = parse_args("--require-manifest", "--max-binary-bytes", "4096", "--max-record-payload-bytes", "512")
    record, findings = hceval_budget_audit.audit_artifact(hceval, args)

    assert findings == []
    assert record.status == "pass"
    assert record.record_count == 2
    assert record.manifest == str(manifest)
    assert record.max_record_payload_bytes > 0


def test_audit_flags_budget_and_missing_manifest(tmp_path: Path) -> None:
    hceval = tmp_path / "input.hceval"
    manifest = tmp_path / "other.manifest.json"
    write_fixture(hceval, manifest)

    args = parse_args("--require-manifest", "--max-binary-bytes", "32", "--max-record-payload-bytes", "1")
    record, findings = hceval_budget_audit.audit_artifact(hceval, args)
    kinds = {finding.kind for finding in findings}

    assert record.status == "fail"
    assert "missing_manifest" in kinds
    assert "max_binary_bytes" in kinds
    assert any(finding.kind == "inspection_finding" for finding in findings)


def test_cli_writes_reports_and_junit(tmp_path: Path) -> None:
    hceval = tmp_path / "input.hceval"
    manifest = tmp_path / "input.manifest.json"
    output_dir = tmp_path / "out"
    write_fixture(hceval, manifest)

    status = hceval_budget_audit.main(
        [
            str(hceval),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "hceval_budget",
            "--require-manifest",
            "--max-binary-bytes",
            "4096",
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "hceval_budget.json").read_text(encoding="utf-8"))
    junit = ET.parse(output_dir / "hceval_budget_junit.xml").getroot()
    csv_text = (output_dir / "hceval_budget.csv").read_text(encoding="utf-8")
    assert report["status"] == "pass"
    assert report["summary"]["artifacts"] == 1
    assert junit.attrib["name"] == "holyc_hceval_budget_audit"
    assert junit.attrib["failures"] == "0"
    assert "binary_bytes" in csv_text
