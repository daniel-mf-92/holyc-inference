#!/usr/bin/env python3
"""Host-side checks for deterministic eval dataset curation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

CURATE_PATH = BENCH_PATH / "dataset_curate.py"
spec = importlib.util.spec_from_file_location("dataset_curate", CURATE_PATH)
dataset_curate = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_curate"] = dataset_curate
spec.loader.exec_module(dataset_curate)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_balanced_answer_index_sampling_limits_label_skew() -> None:
    rows = [
        {
            "id": f"answer-{answer_index}-{offset}",
            "prompt": f"Question {answer_index}-{offset}",
            "choices": ["A", "B", "C", "D"],
            "answer_index": answer_index,
            "provenance": "synthetic balanced curation test",
        }
        for answer_index in range(4)
        for offset in range(3)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.jsonl"
        output = Path(tmp) / "curated.jsonl"
        manifest = Path(tmp) / "curated.manifest.json"
        write_jsonl(source, rows)

        status = dataset_curate.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--source-name",
                "synthetic-balanced",
                "--source-license",
                "synthetic",
                "--max-records",
                "6",
                "--balance-answer-index",
            ]
        )

        assert status == 0
        manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
        assert manifest_json["filters"]["balance_answer_index"] is True
        assert manifest_json["record_count"] == 6
        assert max(manifest_json["answer_histogram"].values()) - min(manifest_json["answer_histogram"].values()) <= 1


def test_duplicate_ids_fail_after_filtering() -> None:
    rows = [
        {
            "id": "duplicate",
            "dataset": "kept",
            "prompt": "Question A",
            "choices": ["yes", "no"],
            "answer_index": 0,
            "provenance": "synthetic duplicate curation test",
        },
        {
            "id": "duplicate",
            "dataset": "kept",
            "prompt": "Question B",
            "choices": ["yes", "no"],
            "answer_index": 1,
            "provenance": "synthetic duplicate curation test",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.jsonl"
        output = Path(tmp) / "curated.jsonl"
        manifest = Path(tmp) / "curated.manifest.json"
        write_jsonl(source, rows)

        status = dataset_curate.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--source-name",
                "synthetic-duplicates",
            ]
        )

        assert status == 2
        assert not output.exists()


def test_per_dataset_and_split_caps_are_deterministic() -> None:
    rows = [
        {
            "id": f"{dataset}-{split}-{index}",
            "dataset": dataset,
            "split": split,
            "prompt": f"Question {dataset} {split} {index}",
            "choices": ["A", "B"],
            "answer_index": index % 2,
            "provenance": "synthetic grouped curation test",
        }
        for dataset in ("arc", "hellaswag")
        for split in ("train", "validation")
        for index in range(4)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.jsonl"
        output = Path(tmp) / "curated.jsonl"
        manifest = Path(tmp) / "curated.manifest.json"
        write_jsonl(source, rows)

        status = dataset_curate.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--source-name",
                "synthetic-grouped",
                "--max-records-per-dataset",
                "3",
                "--max-records-per-split",
                "2",
            ]
        )

        assert status == 0
        manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
        curated_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        assert manifest_json["filters"]["max_records_per_dataset"] == 3
        assert manifest_json["filters"]["max_records_per_split"] == 2
        assert manifest_json["total_after_filters"] == 16
        assert manifest_json["total_after_group_caps"] == manifest_json["record_count"]
        assert manifest_json["record_count"] <= 4
        assert all(count <= 3 for count in manifest_json["dataset_counts"].values())
        assert all(count <= 2 for count in manifest_json["split_counts"].values())
        assert curated_rows == sorted(curated_rows, key=lambda row: (row["dataset"], row["split"], row["record_id"]))


def test_per_dataset_split_cap_limits_each_pair() -> None:
    rows = [
        {
            "id": f"{dataset}-{split}-{index}",
            "dataset": dataset,
            "split": split,
            "prompt": f"Question {dataset} {split} {index}",
            "choices": ["A", "B"],
            "answer_index": index % 2,
            "provenance": "synthetic dataset split cap curation test",
        }
        for dataset in ("arc", "hellaswag")
        for split in ("train", "validation")
        for index in range(5)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.jsonl"
        output = Path(tmp) / "curated.jsonl"
        manifest = Path(tmp) / "curated.manifest.json"
        write_jsonl(source, rows)

        status = dataset_curate.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--source-name",
                "synthetic-dataset-split",
                "--max-records-per-dataset-split",
                "2",
            ]
        )

        assert status == 0
        manifest_json = json.loads(manifest.read_text(encoding="utf-8"))

        assert manifest_json["filters"]["max_records_per_dataset_split"] == 2
        assert manifest_json["total_after_filters"] == 20
        assert manifest_json["total_after_group_caps"] == 8
        assert manifest_json["dataset_split_counts"] == {
            "arc": {"train": 2, "validation": 2},
            "hellaswag": {"train": 2, "validation": 2},
        }


if __name__ == "__main__":
    test_balanced_answer_index_sampling_limits_label_skew()
    test_duplicate_ids_fail_after_filtering()
    test_per_dataset_and_split_caps_are_deterministic()
    test_per_dataset_split_cap_limits_each_pair()
    print("eval_dataset_curate_tests=ok")
