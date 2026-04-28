#!/usr/bin/env python3
"""Host-side checks for offline eval dataset artifact indexing."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_index


def test_junit_report_marks_status_and_finding_failures() -> None:
    report = {
        "artifacts": [
            {
                "source": "smoke_eval.manifest.json",
                "status": "fail",
                "findings": ["binary_sha256 does not match output file"],
            },
            {
                "source": "smoke_eval.inspect.json",
                "status": "pass",
                "findings": [],
            },
        ],
    }

    root = ET.fromstring(dataset_index.junit_report(report))

    assert root.attrib["name"] == "holyc_dataset_index"
    assert root.attrib["tests"] == "2"
    assert root.attrib["failures"] == "2"
    failures = root.findall("./testcase/failure")
    assert {failure.attrib["type"] for failure in failures} == {
        "dataset_artifact_failure",
        "dataset_artifact_findings",
    }


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "smoke.inspect.json").write_text(
        json.dumps(
            {
                "dataset": "smoke-eval",
                "split": "validation",
                "record_count": 3,
                "payload_sha256": "",
                "source_sha256": "a" * 64,
                "status": "pass",
                "findings": [],
            }
        ),
        encoding="utf-8",
    )

    status = dataset_index.main(["--input", str(input_dir), "--output-dir", str(output_dir), "--fail-on-findings"])

    assert status == 0
    payload = json.loads((output_dir / "dataset_index_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "dataset_index_latest.md").read_text(encoding="utf-8")
    csv_text = (output_dir / "dataset_index_latest.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "dataset_index_junit_latest.xml").getroot()

    assert payload["status"] == "pass"
    assert payload["artifacts"][0]["artifact_type"] == "inspect_report"
    assert "Eval Dataset Artifact Index" in markdown
    assert "smoke-eval" in csv_text
    assert junit_root.attrib["name"] == "holyc_dataset_index"
    assert junit_root.attrib["failures"] == "0"
