#!/usr/bin/env python3
"""Host-side checks for eval dataset provenance balance audit gates."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_pack
import dataset_provenance_balance_audit


def make_loaded(
    record_id: str,
    dataset: str,
    split: str,
    provenance: str,
) -> dataset_provenance_balance_audit.LoadedRecord:
    return dataset_provenance_balance_audit.LoadedRecord(
        source="sample.jsonl",
        row_number=int(record_id.rsplit("-", 1)[1]) + 1,
        record=dataset_pack.EvalRecord(
            record_id=record_id,
            dataset=dataset,
            split=split,
            prompt=f"Prompt {record_id}",
            choices=["alpha", "bravo", "charlie", "delta"],
            answer_index=0,
            provenance=provenance,
        ),
    )


def test_distribution_rows_cover_overall_and_dataset_split_provenance() -> None:
    records = [
        make_loaded("row-0", "arc", "validation", "source-a"),
        make_loaded("row-1", "arc", "validation", "source-a"),
        make_loaded("row-2", "truthfulqa", "test", "source-b"),
    ]

    rows = dataset_provenance_balance_audit.distribution_rows(records)

    assert rows == [
        {
            "scope": "provenance",
            "dataset": "",
            "split": "",
            "provenance": "source-a",
            "records": 2,
            "pct_of_scope": 66.66666666666666,
        },
        {
            "scope": "provenance",
            "dataset": "",
            "split": "",
            "provenance": "source-b",
            "records": 1,
            "pct_of_scope": 33.33333333333333,
        },
        {
            "scope": "dataset_split_provenance",
            "dataset": "arc",
            "split": "validation",
            "provenance": "source-a",
            "records": 2,
            "pct_of_scope": 100.0,
        },
        {
            "scope": "dataset_split_provenance",
            "dataset": "truthfulqa",
            "split": "test",
            "provenance": "source-b",
            "records": 1,
            "pct_of_scope": 100.0,
        },
    ]


def test_provenance_balance_gates_report_missing_and_dominant_sources(tmp_path: Path) -> None:
    source = tmp_path / "provenance.jsonl"
    output = tmp_path / "provenance.json"
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
                    "provenance": "source-a",
                },
                {
                    "id": "arc-b",
                    "dataset": "arc",
                    "split": "validation",
                    "prompt": "B",
                    "choices": ["a", "b", "c", "d"],
                    "answer_index": 1,
                    "provenance": "",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status = dataset_provenance_balance_audit.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--require-provenance",
            "--require-provenance-source",
            "source-b",
            "--min-provenance-sources",
            "3",
            "--min-records-per-provenance",
            "2",
            "--max-provenance-pct",
            "49",
            "--max-dataset-split-provenance-pct",
            "49",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in report["findings"]}
    assert status == 1
    assert report["status"] == "fail"
    assert report["provenance_counts"] == {
        "arc": 1,
        "source-a": 1,
    }
    assert {
        "missing_provenance_source",
        "min_provenance_sources",
        "min_records_per_provenance",
        "max_provenance_pct",
        "max_dataset_split_provenance_pct",
    }.issubset(kinds)


def test_write_record_csv_emits_payload_hashes(tmp_path: Path) -> None:
    records = [
        make_loaded("row-0", "arc", "validation", "source-a"),
        make_loaded("row-1", "truthfulqa", "test", ""),
    ]
    output = tmp_path / "records.csv"

    dataset_provenance_balance_audit.write_record_csv(
        output,
        dataset_provenance_balance_audit.record_rows(records),
    )

    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert rows[0]["record_id"] == "row-0"
    assert rows[0]["dataset_split"] == "arc:validation"
    assert rows[0]["provenance_bucket"] == "source-a"
    assert len(rows[0]["normalized_payload_sha256"]) == 64
    assert rows[1]["provenance_bucket"] == "(missing)"


if __name__ == "__main__":
    test_distribution_rows_cover_overall_and_dataset_split_provenance()
    with tempfile.TemporaryDirectory(prefix="dataset-provenance-balance-audit-test-") as tmp:
        test_provenance_balance_gates_report_missing_and_dominant_sources(Path(tmp))
        test_write_record_csv_emits_payload_hashes(Path(tmp))
    print("dataset_provenance_balance_audit_tests=ok")
