#!/usr/bin/env python3
"""Host-side checks for eval dataset mix audit gates."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_mix_audit
import dataset_pack


def make_loaded(record_id: str, dataset: str, split: str) -> dataset_mix_audit.LoadedRecord:
    return dataset_mix_audit.LoadedRecord(
        source="sample.jsonl",
        row_number=int(record_id.rsplit("-", 1)[1]) + 1,
        record=dataset_pack.EvalRecord(
            record_id=record_id,
            dataset=dataset,
            split=split,
            prompt=f"Prompt {record_id}",
            choices=["alpha", "bravo", "charlie", "delta"],
            answer_index=0,
            provenance="synthetic",
        ),
    )


def test_distribution_rows_cover_dataset_split_and_pair_scopes() -> None:
    records = [
        make_loaded("row-0", "arc", "validation"),
        make_loaded("row-1", "arc", "validation"),
        make_loaded("row-2", "truthfulqa", "test"),
    ]

    rows = dataset_mix_audit.distribution_rows(records)

    assert rows == [
        {"scope": "dataset", "dataset": "arc", "split": "", "records": 2, "pct_of_total": 66.66666666666666},
        {"scope": "dataset", "dataset": "truthfulqa", "split": "", "records": 1, "pct_of_total": 33.33333333333333},
        {"scope": "split", "dataset": "", "split": "test", "records": 1, "pct_of_total": 33.33333333333333},
        {
            "scope": "split",
            "dataset": "",
            "split": "validation",
            "records": 2,
            "pct_of_total": 66.66666666666666,
        },
        {
            "scope": "dataset_split",
            "dataset": "arc",
            "split": "validation",
            "records": 2,
            "pct_of_total": 66.66666666666666,
        },
        {
            "scope": "dataset_split",
            "dataset": "truthfulqa",
            "split": "test",
            "records": 1,
            "pct_of_total": 33.33333333333333,
        },
    ]


def test_mix_gates_report_missing_and_dominant_buckets(tmp_path: Path) -> None:
    source = tmp_path / "mix.jsonl"
    output = tmp_path / "mix.json"
    source.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in [
                {
                    "id": "arc-a",
                    "dataset": "arc",
                    "split": "validation",
                    "prompt": "A",
                    "choices": ["a", "b", "c", "d"],
                    "answer_index": 0,
                    "provenance": "synthetic",
                },
                {
                    "id": "arc-b",
                    "dataset": "arc",
                    "split": "validation",
                    "prompt": "B",
                    "choices": ["a", "b", "c", "d"],
                    "answer_index": 1,
                    "provenance": "synthetic",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status = dataset_mix_audit.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--min-datasets",
            "2",
            "--min-records-per-dataset",
            "3",
            "--require-dataset",
            "truthfulqa",
            "--require-dataset-split",
            "truthfulqa:test",
            "--max-dataset-pct",
            "75",
            "--max-dataset-split-pct",
            "75",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in report["findings"]}
    assert status == 1
    assert report["status"] == "fail"
    assert report["dataset_count"] == 1
    assert report["dataset_counts"] == {"arc": 2}
    assert {
        "min_datasets",
        "missing_dataset",
        "missing_dataset_split",
        "min_dataset_records",
        "max_dataset_pct",
        "max_dataset_split_pct",
    }.issubset(kinds)


def test_write_record_csv_emits_payload_hashes(tmp_path: Path) -> None:
    records = [make_loaded("row-0", "arc", "validation"), make_loaded("row-1", "truthfulqa", "test")]
    output = tmp_path / "records.csv"

    dataset_mix_audit.write_record_csv(output, dataset_mix_audit.record_rows(records))

    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert rows[0]["record_id"] == "row-0"
    assert rows[0]["dataset_split"] == "arc:validation"
    assert rows[0]["choice_count"] == "4"
    assert len(rows[0]["normalized_payload_sha256"]) == 64
    assert rows[1]["dataset_split"] == "truthfulqa:test"


if __name__ == "__main__":
    test_distribution_rows_cover_dataset_split_and_pair_scopes()
    with tempfile.TemporaryDirectory(prefix="dataset-mix-audit-test-") as tmp:
        test_mix_gates_report_missing_and_dominant_buckets(Path(tmp))
        test_write_record_csv_emits_payload_hashes(Path(tmp))
    print("dataset_mix_audit_tests=ok")
