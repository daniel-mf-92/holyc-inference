#!/usr/bin/env python3
"""Tests for deterministic eval dataset selection."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_select


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def row(record_id: str, answer_index: int, dataset: str = "arc-smoke") -> dict[str, object]:
    return {
        "id": record_id,
        "dataset": dataset,
        "split": "validation",
        "prompt": f"Question {record_id}?",
        "choices": ["A", "B", "C", "D"],
        "answer_index": answer_index,
        "provenance": "synthetic dataset_select test",
    }


def test_select_balances_answer_buckets_per_slice(tmp_path: Path) -> None:
    source = tmp_path / "eval.jsonl"
    selected = tmp_path / "selected.jsonl"
    manifest = tmp_path / "manifest.json"
    selected_csv = tmp_path / "selected.csv"
    write_jsonl(
        source,
        [
            row("a0", 0),
            row("a1", 0),
            row("a2", 0),
            row("b0", 1),
            row("b1", 1),
            row("b2", 1),
        ],
    )

    status = dataset_select.main(
        [
            "--input",
            str(source),
            "--output",
            str(selected),
            "--manifest",
            str(manifest),
            "--csv",
            str(selected_csv),
            "--max-records-per-slice",
            "4",
            "--balance-answer",
            "--fail-on-findings",
        ]
    )

    report = json.loads(manifest.read_text(encoding="utf-8"))
    selected_rows = [json.loads(line) for line in selected.read_text(encoding="utf-8").splitlines()]
    selected_csv_rows = list(csv.DictReader(selected_csv.open(encoding="utf-8", newline="")))
    assert status == 0
    assert report["status"] == "pass"
    assert report["selected_count"] == 4
    assert report["slices"][0]["answer_histogram"] == {"0": 2, "1": 2}
    assert len(selected_rows) == 4
    assert all("source" not in item for item in selected_rows)
    assert len(selected_csv_rows) == 4
    assert {row["record_id"] for row in selected_csv_rows} == {row["record_id"] for row in selected_rows}
    assert all(len(row["payload_sha256"]) == 64 and len(row["rank_sha256"]) == 64 for row in selected_csv_rows)


def test_select_keeps_slices_separate_and_caps_total(tmp_path: Path) -> None:
    source = tmp_path / "eval.jsonl"
    selected = tmp_path / "selected.jsonl"
    manifest = tmp_path / "manifest.json"
    write_jsonl(source, [row("arc-0", 0, "arc"), row("arc-1", 1, "arc"), row("truth-0", 0, "truthfulqa")])

    status = dataset_select.main(
        [
            "--input",
            str(source),
            "--output",
            str(selected),
            "--manifest",
            str(manifest),
            "--max-records-total",
            "2",
            "--fail-on-findings",
        ]
    )

    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert status == 0
    assert report["selected_count"] == 2
    assert len(report["selected_sha256"]) == 64


def test_select_flags_duplicate_record_ids(tmp_path: Path) -> None:
    source = tmp_path / "eval.jsonl"
    selected = tmp_path / "selected.jsonl"
    manifest = tmp_path / "manifest.json"
    write_jsonl(source, [row("dup", 0), row("dup", 1)])

    status = dataset_select.main(
        [
            "--input",
            str(source),
            "--output",
            str(selected),
            "--manifest",
            str(manifest),
            "--fail-on-findings",
        ]
    )

    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert status == 1
    assert report["status"] == "fail"
    assert {finding["kind"] for finding in report["findings"]} == {"duplicate_record_id"}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_select_balances_answer_buckets_per_slice(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_select_keeps_slices_separate_and_caps_total(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_select_flags_duplicate_record_ids(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
