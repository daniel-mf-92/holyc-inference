#!/usr/bin/env python3
"""Host-side checks for offline eval dataset provenance audits."""

from __future__ import annotations

import csv
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
curate_spec = importlib.util.spec_from_file_location("dataset_curate", CURATE_PATH)
dataset_curate = importlib.util.module_from_spec(curate_spec)
assert curate_spec and curate_spec.loader
sys.modules["dataset_curate"] = dataset_curate
curate_spec.loader.exec_module(dataset_curate)

AUDIT_PATH = BENCH_PATH / "dataset_provenance_audit.py"
audit_spec = importlib.util.spec_from_file_location("dataset_provenance_audit", AUDIT_PATH)
dataset_provenance_audit = importlib.util.module_from_spec(audit_spec)
assert audit_spec and audit_spec.loader
sys.modules["dataset_provenance_audit"] = dataset_provenance_audit
audit_spec.loader.exec_module(dataset_provenance_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def skewed_dataset_rows() -> list[dict]:
    return [
        {
            "id": f"arc-{index}",
            "dataset": "arc",
            "split": "validation",
            "prompt": f"ARC question {index}",
            "choices": ["A", "B"],
            "answer_index": 0,
            "provenance": "synthetic provenance audit test",
        }
        for index in range(2)
    ] + [
        {
            "id": f"truthfulqa-{index}",
            "dataset": "truthfulqa",
            "split": "validation",
            "prompt": f"TruthfulQA question {index}",
            "choices": ["A", "B"],
            "answer_index": 1,
            "provenance": "synthetic provenance audit test",
        }
        for index in range(2)
    ]


def curate_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "curated.jsonl"
    manifest = tmp_path / "curated.manifest.json"
    write_jsonl(source, skewed_dataset_rows())
    status = dataset_curate.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--source-name",
            "synthetic-provenance-audit",
            "--source-version",
            "synthetic-v1",
            "--source-license",
            "synthetic",
        ]
    )
    assert status == 0
    return manifest


def test_provenance_audit_reports_dataset_answer_histograms() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = curate_fixture(tmp_path)
        output_dir = tmp_path / "audit"

        status = dataset_provenance_audit.main(
            ["--input", str(manifest), "--output-dir", str(output_dir), "--fail-on-findings"]
        )

        assert status == 0
        report = json.loads((output_dir / "dataset_provenance_audit_latest.json").read_text(encoding="utf-8"))
        artifact = report["artifacts"][0]
        csv_rows = list(
            csv.DictReader((output_dir / "dataset_provenance_audit_latest.csv").open(newline="", encoding="utf-8"))
        )

        assert artifact["answer_histogram"] == {"0": 2, "1": 2}
        assert artifact["dataset_answer_histograms"] == {"arc": {"0": 2}, "truthfulqa": {"1": 2}}
        assert artifact["dataset_majority_answers"]["arc"]["pct"] == 100.0
        assert json.loads(csv_rows[0]["dataset_answer_histograms"]) == {"arc": {"0": 2}, "truthfulqa": {"1": 2}}


def test_per_dataset_majority_answer_gate_fails_skewed_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = curate_fixture(tmp_path)
        output_dir = tmp_path / "audit"

        status = dataset_provenance_audit.main(
            [
                "--input",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--max-majority-answer-pct",
                "75",
                "--max-dataset-majority-answer-pct",
                "75",
                "--fail-on-findings",
            ]
        )

        report = json.loads((output_dir / "dataset_provenance_audit_latest.json").read_text(encoding="utf-8"))
        findings = [finding["kind"] for finding in report["artifacts"][0]["findings"]]
        assert status == 1
        assert "majority_answer_skew" not in findings
        assert findings.count("dataset_majority_answer_skew") == 2


if __name__ == "__main__":
    test_provenance_audit_reports_dataset_answer_histograms()
    test_per_dataset_majority_answer_gate_fails_skewed_dataset()
    print("dataset_provenance_audit_tests=ok")
