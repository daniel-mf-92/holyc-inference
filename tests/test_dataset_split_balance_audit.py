#!/usr/bin/env python3
"""Tests for host-side dataset split-balance audit gates."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_split_balance_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def row(record_id: str, split: str, prompt: str = "Which tool measures temperature?") -> dict[str, object]:
    return {
        "id": record_id,
        "dataset": "unit",
        "split": split,
        "prompt": prompt,
        "choices": ["thermometer", "ruler", "scale", "compass"],
        "answer_index": 0,
    }


def test_split_balance_passes_balanced_required_splits(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    write_jsonl(
        path,
        [
            row("train-1", "train"),
            row("train-2", "train", "Which tool measures distance?"),
            row("validation-1", "validation", "Which object points north?"),
            row("validation-2", "validation", "Which object weighs flour?"),
        ],
    )
    out = tmp_path / "out"

    status = dataset_split_balance_audit.main(
        [
            "--input",
            str(path),
            "--output-dir",
            str(out),
            "--output-stem",
            "split_balance",
            "--require-split",
            "train",
            "--require-dataset-split",
            "unit:validation",
            "--min-splits-per-dataset",
            "2",
            "--max-largest-split-pct",
            "50",
        ]
    )

    report = json.loads((out / "split_balance.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["summary"]["records"] == 4
    assert report["datasets"][0]["largest_split_pct"] == 50.0
    assert report["findings"] == []
    assert "pct_of_dataset" in (out / "split_balance.csv").read_text(encoding="utf-8")
    assert "largest_split_pct" in (out / "split_balance_datasets.csv").read_text(encoding="utf-8")
    assert ET.parse(out / "split_balance_junit.xml").getroot().attrib["failures"] == "0"


def test_split_balance_flags_missing_and_dominant_splits(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    write_jsonl(path, [row("train-1", "train"), row("train-2", "train")])
    out = tmp_path / "out"

    status = dataset_split_balance_audit.main(
        [
            "--input",
            str(path),
            "--output-dir",
            str(out),
            "--output-stem",
            "split_balance",
            "--require-split",
            "validation",
            "--require-dataset-split",
            "unit:test",
            "--min-splits-per-dataset",
            "2",
            "--max-largest-split-pct",
            "60",
        ]
    )

    report = json.loads((out / "split_balance.json").read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in report["findings"]}
    assert status == 1
    assert report["status"] == "fail"
    assert {
        "required_split_missing",
        "required_dataset_split_missing",
        "min_splits_per_dataset",
        "largest_split_pct",
    } <= kinds
    assert ET.parse(out / "split_balance_junit.xml").getroot().attrib["failures"] == "1"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-split-balance-test-") as tmp:
        test_split_balance_passes_balanced_required_splits(Path(tmp) / "pass")
        test_split_balance_flags_missing_and_dominant_splits(Path(tmp) / "fail")
    print("test_dataset_split_balance_audit=ok")
