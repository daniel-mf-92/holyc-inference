#!/usr/bin/env python3
"""Host-side checks for eval dataset fingerprint diffs."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_PATH = ROOT / "bench" / "dataset_fingerprint.py"
DIFF_PATH = ROOT / "bench" / "dataset_fingerprint_diff.py"

fingerprint_spec = importlib.util.spec_from_file_location("dataset_fingerprint", FINGERPRINT_PATH)
dataset_fingerprint = importlib.util.module_from_spec(fingerprint_spec)
assert fingerprint_spec and fingerprint_spec.loader
sys.modules["dataset_fingerprint"] = dataset_fingerprint
fingerprint_spec.loader.exec_module(dataset_fingerprint)

diff_spec = importlib.util.spec_from_file_location("dataset_fingerprint_diff", DIFF_PATH)
dataset_fingerprint_diff = importlib.util.module_from_spec(diff_spec)
assert diff_spec and diff_spec.loader
sys.modules["dataset_fingerprint_diff"] = dataset_fingerprint_diff
diff_spec.loader.exec_module(dataset_fingerprint_diff)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def fingerprint(source: Path, output: Path) -> int:
    return dataset_fingerprint.main(["--input", str(source), "--output", str(output), "--fail-on-findings"])


def test_fingerprint_diff_gates_answer_content_and_membership_drift() -> None:
    baseline_rows = [
        {
            "id": "row-a",
            "dataset": "arc",
            "split": "validation",
            "prompt": "Choose A.",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "provenance": "synthetic fingerprint diff unit test",
        },
        {
            "id": "row-b",
            "dataset": "arc",
            "split": "validation",
            "prompt": "Choose B.",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 1,
            "provenance": "synthetic fingerprint diff unit test",
        },
    ]
    candidate_rows = [
        {**baseline_rows[0], "answer_index": 1},
        {**baseline_rows[1], "prompt": "Choose B after prompt edit."},
        {
            "id": "row-c",
            "dataset": "arc",
            "split": "validation",
            "prompt": "Choose C.",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 2,
            "provenance": "synthetic fingerprint diff unit test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        baseline_source = tmp_path / "baseline.jsonl"
        candidate_source = tmp_path / "candidate.jsonl"
        baseline_report = tmp_path / "baseline.json"
        candidate_report = tmp_path / "candidate.json"
        output = tmp_path / "diff.json"
        findings_csv = tmp_path / "diff_findings.csv"
        write_jsonl(baseline_source, baseline_rows)
        write_jsonl(candidate_source, candidate_rows)
        assert fingerprint(baseline_source, baseline_report) == 0
        assert fingerprint(candidate_source, candidate_report) == 0

        status = dataset_fingerprint_diff.main(
            [
                "--baseline",
                str(baseline_report),
                "--candidate",
                str(candidate_report),
                "--output",
                str(output),
                "--findings-csv",
                str(findings_csv),
                "--fail-on-added",
                "--fail-on-removed",
                "--fail-on-content-changes",
                "--fail-on-answer-changes",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert report["change_counts"]["added"] == 1
        assert report["change_counts"].get("removed", 0) == 0
        assert report["change_counts"]["answer_changed"] == 1
        assert report["change_counts"]["content_changed"] >= 1
        assert {"added", "answer_changed", "content_changed"}.issubset(kinds)
        csv_text = findings_csv.read_text(encoding="utf-8")
        assert "severity,kind,key,detail" in csv_text
        assert "answer_changed" in csv_text


if __name__ == "__main__":
    test_fingerprint_diff_gates_answer_content_and_membership_drift()
    print("dataset_fingerprint_diff_tests=ok")
