#!/usr/bin/env python3
"""Tests for HCEval binary diff tooling."""

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
import hceval_diff
import hceval_inspect


def rows() -> list[dict[str, object]]:
    return [
        {
            "id": "row-1",
            "dataset": "smoke-eval",
            "split": "validation",
            "prompt": "Pick the color.",
            "choices": ["red", "blue", "green", "gray"],
            "answer_index": 1,
            "provenance": "synthetic smoke",
        },
        {
            "id": "row-2",
            "dataset": "smoke-eval",
            "split": "validation",
            "prompt": "Pick the shape.",
            "choices": ["circle", "seven", "yellow", "quiet"],
            "answer_index": 0,
            "provenance": "synthetic smoke",
        },
    ]


def pack_jsonl(tmp_path: Path, name: str, payload_rows: list[dict[str, object]]) -> Path:
    source = tmp_path / f"{name}.jsonl"
    output = tmp_path / f"{name}.hceval"
    source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in payload_rows), encoding="utf-8")
    records = dataset_pack.normalize_records(dataset_pack.read_jsonl(source), "smoke-eval", "validation")
    dataset_pack.write_outputs(records, output, output.with_suffix(".manifest.json"), "smoke-eval", "validation")
    return output


def test_diff_accepts_identical_binaries(tmp_path: Path) -> None:
    reference = pack_jsonl(tmp_path, "reference", rows())
    candidate = pack_jsonl(tmp_path, "candidate", rows())

    reference_dataset = hceval_inspect.parse_hceval(reference)
    candidate_dataset = hceval_inspect.parse_hceval(candidate)
    diff_records = hceval_diff.build_diff_records(reference_dataset, candidate_dataset)
    findings = hceval_diff.evaluate(
        diff_records,
        hceval_diff.metadata_findings(reference_dataset, candidate_dataset),
        allow_order_changes=False,
    )

    assert findings == []
    assert {record.status for record in diff_records} == {"unchanged"}


def test_diff_flags_changed_added_removed_and_reordered_records(tmp_path: Path) -> None:
    reference = pack_jsonl(tmp_path, "reference", rows())
    candidate_rows = [dict(rows()[1]), dict(rows()[0])]
    candidate_rows[1]["answer_index"] = 3
    candidate_rows.append(
        {
            "id": "row-3",
            "dataset": "smoke-eval",
            "split": "validation",
            "prompt": "Pick the animal.",
            "choices": ["cat", "stone", "paper", "glass"],
            "answer_index": 0,
            "provenance": "synthetic smoke",
        }
    )
    candidate = pack_jsonl(tmp_path, "candidate", candidate_rows)

    reference_dataset = hceval_inspect.parse_hceval(reference)
    candidate_dataset = hceval_inspect.parse_hceval(candidate)
    diff_records = hceval_diff.build_diff_records(reference_dataset, candidate_dataset)
    findings = hceval_diff.evaluate(
        diff_records,
        hceval_diff.metadata_findings(reference_dataset, candidate_dataset),
        allow_order_changes=False,
    )
    kinds = {finding.kind for finding in findings}
    statuses = {record.record_id: record.status for record in diff_records}

    assert statuses["row-1"] == "changed"
    assert statuses["row-2"] == "reordered"
    assert statuses["row-3"] == "added"
    assert {"metadata_changed", "record_changed", "record_reordered", "record_added"} <= kinds


def test_cli_writes_reports(tmp_path: Path) -> None:
    reference = pack_jsonl(tmp_path, "reference", rows())
    candidate = pack_jsonl(tmp_path, "candidate", rows())
    output = tmp_path / "diff.json"
    markdown = tmp_path / "diff.md"
    csv_path = tmp_path / "diff.csv"
    findings_csv = tmp_path / "diff_findings.csv"
    junit = tmp_path / "diff_junit.xml"

    status = hceval_diff.main(
        [
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
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
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert "HCEval Dataset Diff" in markdown.read_text(encoding="utf-8")
    assert list(csv.DictReader(csv_path.open(encoding="utf-8")))[0]["status"] == "unchanged"
    assert list(csv.DictReader(findings_csv.open(encoding="utf-8"))) == []
    root = ET.parse(junit).getroot()
    assert root.attrib["name"] == "holyc_hceval_diff"
    assert root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_diff_accepts_identical_binaries(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_diff_flags_changed_added_removed_and_reordered_records(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_reports(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
