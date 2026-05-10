#!/usr/bin/env python3
"""Tests for host-side dataset answer-order telemetry."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_order_audit
import dataset_pack


def make_loaded(record_id: str, answer_index: int) -> dataset_order_audit.LoadedRecord:
    return dataset_order_audit.LoadedRecord(
        source="sample.jsonl",
        row_number=int(record_id.rsplit("-", 1)[1]) + 1,
        record=dataset_pack.EvalRecord(
            record_id=record_id,
            dataset="order-smoke",
            split="validation",
            prompt=f"Prompt {record_id}",
            choices=["alpha", "bravo", "charlie", "delta"],
            answer_index=answer_index,
            provenance="synthetic",
        ),
    )


def test_record_order_rows_include_neighbors_and_run_membership() -> None:
    records = [
        make_loaded("row-0", 0),
        make_loaded("row-1", 0),
        make_loaded("row-2", 2),
    ]

    rows = dataset_order_audit.record_order_rows(records)

    assert [row["answer_index"] for row in rows] == [0, 0, 2]
    assert rows[0]["previous_answer_index"] is None
    assert rows[0]["next_answer_index"] == 0
    assert rows[0]["changed_from_previous"] is False
    assert rows[1]["changes_to_next"] is True
    assert rows[2]["previous_answer_index"] == 0
    assert rows[0]["run_index"] == 0
    assert rows[0]["run_length"] == 2
    assert rows[1]["run_position"] == 1
    assert rows[2]["run_index"] == 1
    assert rows[2]["is_trailing_run"] is True


def test_write_record_csv_emits_stable_header_and_rows(tmp_path: Path) -> None:
    records = [make_loaded("row-0", 1), make_loaded("row-1", 3)]
    output = tmp_path / "records.csv"

    dataset_order_audit.write_record_csv(records, output)

    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert rows[0]["sequence_index"] == "0"
    assert rows[0]["source"] == "sample.jsonl:1"
    assert rows[0]["previous_answer_index"] == ""
    assert rows[0]["next_answer_index"] == "3"
    assert rows[0]["changes_to_next"] == "True"
    assert rows[1]["changed_from_previous"] == "True"
    assert rows[1]["run_position"] == "0"
    assert rows[1]["is_trailing_run"] == "True"
