from pathlib import Path

import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import dataset_id_audit


def test_id_audit_flags_duplicate_implicit_and_pattern_mismatch(tmp_path: Path) -> None:
    rows = [
        {
            "id": "dup",
            "dataset": "unit",
            "split": "validation",
            "prompt": "First?",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 0,
            "provenance": "unit",
        },
        {
            "id": "dup",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Second?",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 1,
            "provenance": "unit",
        },
        {
            "dataset": "unit",
            "split": "validation",
            "prompt": "Implicit?",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 2,
            "provenance": "unit",
        },
        {
            "id": "Bad ID",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Pattern?",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 3,
            "provenance": "unit",
        },
    ]
    source = tmp_path / "ids.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = dataset_id_audit.parse_args(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.json"),
            "--require-explicit-id",
            "--id-pattern",
            r"[a-z0-9-]+",
            "--fail-duplicate-record-ids",
            "--fail-duplicate-dataset-split-record-ids",
        ]
    )

    report = dataset_id_audit.build_report(args)

    assert report["status"] == "fail"
    assert report["record_count"] == 4
    assert report["explicit_id_count"] == 3
    assert report["duplicate_record_id_count"] == 1
    assert report["duplicate_dataset_split_record_id_count"] == 1
    assert report["id_class_histogram"]["pattern_mismatch"] == 1
    kinds = {finding["kind"] for finding in report["findings"]}
    assert {
        "duplicate_record_id",
        "duplicate_dataset_split_record_id",
        "implicit_record_id",
        "record_id_pattern_mismatch",
    } <= kinds


def test_id_audit_allows_unique_explicit_ids(tmp_path: Path) -> None:
    rows = [
        {
            "id": "row-1",
            "dataset": "unit",
            "split": "validation",
            "prompt": "First?",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 0,
            "provenance": "unit",
        },
        {
            "question_id": "row-2",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Second?",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 1,
            "provenance": "unit",
        },
    ]
    source = tmp_path / "ids.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = dataset_id_audit.parse_args(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.json"),
            "--require-explicit-id",
            "--max-record-id-bytes",
            "16",
            "--id-pattern",
            r"[a-z0-9-]+",
            "--fail-duplicate-record-ids",
            "--fail-duplicate-dataset-split-record-ids",
        ]
    )

    report = dataset_id_audit.build_report(args)

    assert report["status"] == "pass"
    assert report["record_count"] == 2
    assert report["explicit_id_count"] == 2
    assert report["findings"] == []
