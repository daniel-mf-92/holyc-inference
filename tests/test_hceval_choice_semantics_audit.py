#!/usr/bin/env python3
"""Host-side checks for HCEval choice semantics audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_pack
import hceval_choice_semantics_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def pack_hceval(path: Path, rows: list[dict[str, object]]) -> None:
    source = path.with_suffix(".jsonl")
    manifest = Path(str(path) + ".manifest.json")
    write_jsonl(source, rows)
    records = dataset_pack.normalize_records(dataset_pack.read_jsonl(source), "unit", "validation")
    dataset_pack.write_outputs(records, path, manifest, "unit", "validation")


def parse_args(path: Path, extra: list[str] | None = None) -> object:
    return hceval_choice_semantics_audit.build_parser().parse_args(
        ["--input", str(path), *(extra or [])]
    )


def test_build_report_passes_clean_hceval_choices(tmp_path: Path) -> None:
    binary = tmp_path / "clean.hceval"
    pack_hceval(
        binary,
        [
            {
                "id": "clean-1",
                "prompt": "Which tool measures air temperature?",
                "choices": ["thermometer", "ruler", "scale", "compass"],
                "answer_index": 0,
                "provenance": "unit",
            }
        ],
    )

    report = hceval_choice_semantics_audit.build_report(parse_args(binary))

    assert report["status"] == "pass"
    assert report["record_count"] == 1
    assert report["findings"] == []
    assert report["rows"][0]["unique_normalized_choices"] == 4


def test_build_report_flags_duplicate_answer_alias_and_prompt_overlap(tmp_path: Path) -> None:
    binary = tmp_path / "bad.hceval"
    pack_hceval(
        binary,
        [
            {
                "id": "bad-1",
                "prompt": "The clue already says alpha centauri is the nearest listed star.",
                "choices": ["Alpha Centauri", "alpha    centauri", "Barnard's Star", "Sirius"],
                "answer_index": 0,
                "provenance": "unit",
            }
        ],
    )

    report = hceval_choice_semantics_audit.build_report(parse_args(binary, ["--min-overlap-chars", "5"]))
    kinds = {finding["kind"] for finding in report["findings"]}

    assert report["status"] == "fail"
    assert {"duplicate_normalized_choice", "answer_choice_alias", "choice_text_in_prompt"} <= kinds
    assert report["rows"][0]["status"] == "fail"
    assert report["rows"][0]["answer_alias_count"] == 2


def test_cli_writes_choice_semantics_artifacts(tmp_path: Path) -> None:
    binary = tmp_path / "clean.hceval"
    output_dir = tmp_path / "out"
    pack_hceval(
        binary,
        [
            {
                "id": "clean-1",
                "prompt": "Pick the only color word.",
                "choices": ["blue", "table", "chair", "window"],
                "answer_index": 0,
                "provenance": "unit",
            }
        ],
    )

    status = hceval_choice_semantics_audit.main(
        [
            "--input",
            str(binary),
            "--output",
            str(output_dir / "choice_semantics.json"),
            "--markdown",
            str(output_dir / "choice_semantics.md"),
            "--csv",
            str(output_dir / "choice_semantics.csv"),
            "--findings-csv",
            str(output_dir / "choice_semantics_findings.csv"),
            "--junit",
            str(output_dir / "choice_semantics_junit.xml"),
            "--fail-on-findings",
        ]
    )

    payload = json.loads((output_dir / "choice_semantics.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "choice_semantics.csv").open(encoding="utf-8")))
    findings = list(csv.DictReader((output_dir / "choice_semantics_findings.csv").open(encoding="utf-8")))
    junit_root = ET.parse(output_dir / "choice_semantics_junit.xml").getroot()

    assert status == 0
    assert payload["status"] == "pass"
    assert len(rows) == 1
    assert findings == []
    assert "HCEval Choice Semantics Audit" in (output_dir / "choice_semantics.md").read_text(encoding="utf-8")
    assert junit_root.attrib["name"] == "holyc_hceval_choice_semantics_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pass"
        path.mkdir()
        test_build_report_passes_clean_hceval_choices(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fail"
        path.mkdir()
        test_build_report_flags_duplicate_answer_alias_and_prompt_overlap(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cli"
        path.mkdir()
        test_cli_writes_choice_semantics_artifacts(path)
    print("hceval_choice_semantics_audit_tests=ok")
