#!/usr/bin/env python3
"""Host-side checks for eval dataset split-leak audits."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

AUDIT_PATH = BENCH_PATH / "dataset_leak_audit.py"
spec = importlib.util.spec_from_file_location("dataset_leak_audit", AUDIT_PATH)
dataset_leak_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_leak_audit"] = dataset_leak_audit
spec.loader.exec_module(dataset_leak_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_detects_prompt_and_payload_leak_across_splits() -> None:
    rows = [
        {
            "id": "train-1",
            "dataset": "synthetic",
            "split": "train",
            "prompt": "Which item is hot?",
            "choices": ["ice", "fire", "snow"],
            "answer_index": 1,
            "provenance": "synthetic leak audit test",
        },
        {
            "id": "validation-1",
            "dataset": "synthetic",
            "split": "validation",
            "prompt": "  which item is hot? ",
            "choices": ["ice", "fire", "snow"],
            "answer_index": 1,
            "provenance": "synthetic leak audit test",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.jsonl"
        output = Path(tmp) / "leak.json"
        markdown = Path(tmp) / "leak.md"
        csv_path = Path(tmp) / "leak.csv"
        junit = Path(tmp) / "leak.xml"
        write_jsonl(source, rows)

        status = dataset_leak_audit.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--markdown",
                str(markdown),
                "--csv",
                str(csv_path),
                "--junit",
                str(junit),
                "--fail-on-leaks",
            ]
        )

        assert status == 1
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "fail"
        assert {finding["kind"] for finding in report["findings"]} >= {
            "prompt_split_leak",
            "payload_split_leak",
        }
        assert "payload_split_leak" in markdown.read_text(encoding="utf-8")
        assert "prompt_split_leak" in csv_path.read_text(encoding="utf-8")
        junit_root = ET.parse(junit).getroot()
        assert junit_root.attrib["name"] == "holyc_dataset_leak_audit"
        assert junit_root.attrib["failures"] == "2"
        assert junit_root.find("./testcase/failure") is not None


def test_duplicate_id_within_split_is_warning_only() -> None:
    rows = [
        {
            "id": "duplicate",
            "dataset": "synthetic",
            "split": "validation",
            "prompt": "Question A",
            "choices": ["yes", "no"],
            "answer_index": 0,
            "provenance": "synthetic leak audit test",
        },
        {
            "id": "duplicate",
            "dataset": "synthetic",
            "split": "validation",
            "prompt": "Question B",
            "choices": ["yes", "no"],
            "answer_index": 1,
            "provenance": "synthetic leak audit test",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.jsonl"
        output = Path(tmp) / "audit.json"
        junit = Path(tmp) / "audit.xml"
        write_jsonl(source, rows)

        status = dataset_leak_audit.main(
            ["--input", str(source), "--output", str(output), "--junit", str(junit), "--fail-on-leaks"]
        )

        assert status == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "pass"
        assert report["warning_count"] == 1
        assert report["findings"][0]["kind"] == "duplicate_record_id"
        junit_root = ET.parse(junit).getroot()
        assert junit_root.attrib["failures"] == "0"
        system_out = junit_root.find("./testcase/system-out")
        assert system_out is not None
        assert "duplicate_record_id" in (system_out.text or "")


if __name__ == "__main__":
    test_detects_prompt_and_payload_leak_across_splits()
    test_duplicate_id_within_split_is_warning_only()
    print("dataset_leak_audit_tests=ok")
