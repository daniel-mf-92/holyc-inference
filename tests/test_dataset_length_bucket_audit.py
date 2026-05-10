#!/usr/bin/env python3
"""Tests for eval dataset length-bucket coverage auditing."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_length_bucket_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def row(record_id: str, prompt: str, choices: list[str], answer_index: int = 0) -> dict[str, object]:
    return {
        "id": record_id,
        "dataset": "bucket-smoke",
        "split": "validation",
        "prompt": prompt,
        "choices": choices,
        "answer_index": answer_index,
        "provenance": "synthetic dataset length bucket test",
    }


def sample_rows() -> list[dict[str, object]]:
    return [
        row("short", "tiny?", ["a", "b", "c", "d"], 0),
        row("medium", "question " * 8, ["answer one", "answer two", "answer three", "answer four"], 1),
        row("long", "context " * 40, ["alpha", "beta", "gamma", "delta"], 2),
    ]


def test_bucket_for_size_uses_inclusive_edges() -> None:
    edges = dataset_length_bucket_audit.parse_bucket_edges("8,16")

    assert dataset_length_bucket_audit.bucket_for_size(8, edges) == ("0-8", 0, 8)
    assert dataset_length_bucket_audit.bucket_for_size(9, edges) == ("9-16", 9, 16)
    assert dataset_length_bucket_audit.bucket_for_size(17, edges) == ("17+", 17, -1)


def test_records_are_bucketed_by_prompt_plus_choice_bytes(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, sample_rows()[:2])
    loaded, _, findings = dataset_length_bucket_audit.load_records([path], "eval", "validation")
    buckets = [dataset_length_bucket_audit.record_bucket(record, [16, 128]) for record in loaded]
    summaries = dataset_length_bucket_audit.bucket_summaries(buckets)

    assert findings == []
    assert [row.bucket for row in buckets] == ["0-16", "17-128"]
    assert summaries[0].answer_index_histogram == "0:1"
    assert summaries[1].choice_count_histogram == "4:1"


def test_gate_findings_cover_bucket_skew_and_missing_buckets(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, sample_rows()[:1])
    output_dir = tmp_path / "out"

    status = dataset_length_bucket_audit.main(
        [
            "--input",
            str(path),
            "--bucket-edges",
            "16,128",
            "--min-records",
            "2",
            "--min-covered-buckets",
            "2",
            "--max-largest-bucket-pct",
            "90",
            "--require-bucket",
            "17-128",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "length",
        ]
    )

    report = json.loads((output_dir / "length.json").read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in report["findings"]}
    assert status == 1
    assert {"min_records", "min_covered_buckets", "largest_bucket_pct", "required_bucket_missing"} <= kinds
    assert ET.parse(output_dir / "length_junit.xml").getroot().attrib["failures"] == "1"


def test_cli_writes_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(path, sample_rows())

    status = dataset_length_bucket_audit.main(
        [
            "--input",
            str(path),
            "--bucket-edges",
            "16,128",
            "--min-covered-buckets",
            "3",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "length",
        ]
    )

    report = json.loads((output_dir / "length.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["summary"]["records"] == 3
    assert report["summary"]["buckets"] == 3
    assert "Dataset Length Bucket Audit" in (output_dir / "length.md").read_text(encoding="utf-8")
    assert "choice_count_histogram" in (output_dir / "length.csv").read_text(encoding="utf-8")
    assert "total_bytes" in (output_dir / "length_records.csv").read_text(encoding="utf-8")
    assert "kind" in (output_dir / "length_findings.csv").read_text(encoding="utf-8")
    assert ET.parse(output_dir / "length_junit.xml").getroot().attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-length-bucket-test-") as tmp:
        tmp_path = Path(tmp)
        test_bucket_for_size_uses_inclusive_edges()
        test_records_are_bucketed_by_prompt_plus_choice_bytes(tmp_path / "bucket")
        test_gate_findings_cover_bucket_skew_and_missing_buckets(tmp_path / "gates")
        test_cli_writes_sidecars(tmp_path / "cli")
    print("test_dataset_length_bucket_audit=ok")
