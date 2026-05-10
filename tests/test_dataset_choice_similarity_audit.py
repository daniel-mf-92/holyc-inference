#!/usr/bin/env python3
"""Host-side checks for eval dataset choice similarity audit gates."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_choice_similarity_audit
import dataset_pack


def make_loaded(record_id: str, choices: list[str]) -> dataset_choice_similarity_audit.LoadedRecord:
    return dataset_choice_similarity_audit.LoadedRecord(
        source="sample.jsonl",
        row_number=1,
        record=dataset_pack.EvalRecord(
            record_id=record_id,
            dataset="arc",
            split="validation",
            prompt=f"Prompt {record_id}",
            choices=choices,
            answer_index=0,
            provenance="synthetic",
        ),
    )


def test_choice_normalization_collapses_case_punctuation_and_spacing() -> None:
    assert dataset_choice_similarity_audit.normalize_choice("  The QUICK, brown fox! ") == "the quick brown fox"


def test_record_telemetry_reports_duplicate_normalized_choices() -> None:
    loaded = make_loaded("row-0", ["Thermometer", "thermometer!", "ruler", "compass"])

    row = dataset_choice_similarity_audit.record_telemetry(loaded)

    assert row["unique_normalized_choices"] == 3
    assert row["duplicate_choice_pairs"] == 1
    assert row["pair_telemetry"][0]["duplicate_normalized"] is True
    assert row["pair_telemetry"][0]["similarity"] == 1.0


def test_similarity_gates_flag_duplicate_and_near_duplicate_rows(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "choices.jsonl"
    output = tmp_path / "choices.json"
    source.write_text(
        json.dumps(
            {
                "id": "near-dupe",
                "dataset": "arc",
                "split": "validation",
                "question": "Which is used to measure temperature?",
                "choices": [
                    {"label": "A", "text": "thermometer"},
                    {"label": "B", "text": "thermometers"},
                    {"label": "C", "text": "ruler"},
                    {"label": "D", "text": "scale"},
                ],
                "answerKey": "A",
                "provenance": "synthetic",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    status = dataset_choice_similarity_audit.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--min-unique-choices",
            "4",
            "--max-pair-similarity",
            "0.95",
            "--fail-duplicate-normalized",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in report["findings"]}
    assert status == 1
    assert report["status"] == "fail"
    assert "choice_similarity_exceeded" in kinds
    assert report["records"][0]["unique_normalized_choices"] == 4


def test_cli_writes_all_outputs_for_passing_rows(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "choices.jsonl"
    output = tmp_path / "choice_similarity.json"
    source.write_text(
        json.dumps(
            {
                "id": "clean",
                "dataset": "truthfulqa",
                "split": "validation",
                "question": "What should you do with an important surprising claim?",
                "mc1_targets": {
                    "choices": [
                        "Check reliable sources first.",
                        "Assume it is true immediately.",
                        "Ignore all available evidence.",
                        "Repeat the most dramatic version.",
                    ],
                    "labels": [1, 0, 0, 0],
                },
                "provenance": "synthetic",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    status = dataset_choice_similarity_audit.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--markdown",
            str(tmp_path / "choice_similarity.md"),
            "--csv",
            str(tmp_path / "choice_similarity.csv"),
            "--pair-csv",
            str(tmp_path / "choice_similarity_pairs.csv"),
            "--findings-csv",
            str(tmp_path / "choice_similarity_findings.csv"),
            "--junit",
            str(tmp_path / "choice_similarity_junit.xml"),
            "--min-unique-choices",
            "4",
            "--max-pair-similarity",
            "0.95",
            "--fail-duplicate-normalized",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["records"][0]["duplicate_choice_pairs"] == 0
    assert "No dataset choice similarity findings." in (tmp_path / "choice_similarity.md").read_text(encoding="utf-8")
    assert "normalized_choice_sha256" in (tmp_path / "choice_similarity.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "left_index" in (tmp_path / "choice_similarity_pairs.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "kind" in (tmp_path / "choice_similarity_findings.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "dataset_choice_similarity_audit" in (tmp_path / "choice_similarity_junit.xml").read_text(encoding="utf-8")


if __name__ == "__main__":
    test_choice_normalization_collapses_case_punctuation_and_spacing()
    test_record_telemetry_reports_duplicate_normalized_choices()
    with tempfile.TemporaryDirectory(prefix="dataset-choice-similarity-audit-test-") as tmp:
        test_similarity_gates_flag_duplicate_and_near_duplicate_rows(Path(tmp) / "fail")
        test_cli_writes_all_outputs_for_passing_rows(Path(tmp) / "pass")
    print("dataset_choice_similarity_audit_tests=ok")
