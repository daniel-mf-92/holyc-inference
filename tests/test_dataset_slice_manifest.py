#!/usr/bin/env python3
"""Host-side checks for eval dataset slice manifests."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "bench" / "dataset_slice_manifest.py"
spec = importlib.util.spec_from_file_location("dataset_slice_manifest", MANIFEST_PATH)
dataset_slice_manifest = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_slice_manifest"] = dataset_slice_manifest
spec.loader.exec_module(dataset_slice_manifest)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_slice_manifest_reports_dataset_split_hashes() -> None:
    rows = [
        {
            "id": "arc-a",
            "dataset": "arc",
            "split": "validation",
            "prompt": "ARC question A",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "provenance": "synthetic slice manifest test",
        },
        {
            "id": "arc-b",
            "dataset": "arc",
            "split": "validation",
            "prompt": "ARC question B",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 1,
            "provenance": "synthetic slice manifest test",
        },
        {
            "id": "truthfulqa-a",
            "dataset": "truthfulqa",
            "split": "validation",
            "prompt": "TruthfulQA question A",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "provenance": "synthetic slice manifest test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "dataset.jsonl"
        output = Path(tmp) / "manifest.json"
        write_jsonl(source, rows)

        status = dataset_slice_manifest.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--require-slice",
                "arc:validation",
                "--require-slice",
                "truthfulqa:validation",
                "--min-total-slices",
                "2",
                "--min-records-per-slice",
                "1",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["record_count"] == 3
        assert report["slice_count"] == 2
        assert report["slices"][0]["dataset"] == "arc"
        assert report["slices"][0]["record_count"] == 2
        assert report["slices"][0]["answer_histogram"] == {"0": 1, "1": 1}
        assert len(report["slices"][0]["slice_sha256"]) == 64
        assert len(report["records"]) == 3


def test_slice_manifest_gates_missing_and_small_slices() -> None:
    rows = [
        {
            "id": "arc-a",
            "dataset": "arc",
            "split": "validation",
            "prompt": "ARC question A",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "provenance": "synthetic slice manifest test",
        }
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "dataset.jsonl"
        output = Path(tmp) / "manifest.json"
        write_jsonl(source, rows)

        status = dataset_slice_manifest.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--require-slice",
                "truthfulqa:validation",
                "--min-total-slices",
                "2",
                "--min-records-per-slice",
                "2",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert kinds == {"missing_required_slice", "slice_too_small", "too_few_slices"}


if __name__ == "__main__":
    test_slice_manifest_reports_dataset_split_hashes()
    test_slice_manifest_gates_missing_and_small_slices()
    print("dataset_slice_manifest_tests=ok")
