#!/usr/bin/env python3
"""Host-side checks for JSONL to HCEval roundtrip audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_roundtrip_audit.py"
spec = importlib.util.spec_from_file_location("dataset_roundtrip_audit", AUDIT_PATH)
dataset_roundtrip_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_roundtrip_audit"] = dataset_roundtrip_audit
spec.loader.exec_module(dataset_roundtrip_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_roundtrip_audit_passes_and_writes_fingerprints() -> None:
    rows = [
        {
            "id": "rt-a",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Choose the warm color",
            "choices": ["red", "blue", "green", "black"],
            "answer_index": 0,
            "provenance": "synthetic roundtrip unit test",
        },
        {
            "id": "rt-b",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Choose the cold color",
            "choices": ["orange", "blue", "yellow", "red"],
            "answer_index": 1,
            "provenance": "synthetic roundtrip unit test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "roundtrip.jsonl"
        output = Path(tmp) / "roundtrip.json"
        fingerprints = Path(tmp) / "roundtrip_fingerprints.csv"
        write_jsonl(source, rows)

        status = dataset_roundtrip_audit.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--fingerprints-csv",
                str(fingerprints),
                "--dataset",
                "unit",
                "--split",
                "validation",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["record_count"] == 2
        assert report["expected_source_sha256"] == report["actual_source_sha256"]
        assert report["expected_binary_layout"] == report["actual_binary_layout"]
        assert report["expected_record_fingerprints"] == report["actual_record_fingerprints"]
        assert report["byte_stats"]["max_prompt_bytes"] == len("Choose the warm color".encode("utf-8"))
        assert "full_payload_sha256" in fingerprints.read_text(encoding="utf-8").splitlines()[0]


def test_roundtrip_audit_fails_on_inspector_size_gate() -> None:
    rows = [
        {
            "id": "rt-large",
            "dataset": "unit",
            "split": "validation",
            "prompt": "This prompt should exceed the tiny test limit",
            "choices": ["yes", "no"],
            "answer_index": 0,
            "provenance": "synthetic roundtrip unit test",
        }
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "large.jsonl"
        output = Path(tmp) / "large.json"
        write_jsonl(source, rows)

        status = dataset_roundtrip_audit.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--max-prompt-bytes",
                "8",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 1
        assert report["status"] == "fail"
        assert any(finding["kind"] == "inspector_finding" for finding in report["findings"])


if __name__ == "__main__":
    test_roundtrip_audit_passes_and_writes_fingerprints()
    test_roundtrip_audit_fails_on_inspector_size_gate()
    print("dataset_roundtrip_audit_tests=ok")
