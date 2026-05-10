#!/usr/bin/env python3
"""Tests for eval score sparsity audits."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_score_sparsity_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_audit_accepts_dense_finite_scores(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(
        holyc,
        [
            {"id": "a", "scores": [4.0, 1.0, 0.5, 0.25]},
            {"id": "b", "scores": [0.5, 2.0, -0.25, -0.5]},
        ],
    )
    write_jsonl(
        llama,
        [
            {"id": "a", "scores": [3.9, 1.1, 0.4, 0.2]},
            {"id": "b", "scores": [0.4, 2.1, -0.2, -0.4]},
        ],
    )

    records, summaries, findings = eval_score_sparsity_audit.audit(
        holyc,
        llama,
        zero_epsilon=0.0,
        min_records=2,
        min_nonzero_scores_per_record=4,
        min_unique_scores_per_record=4,
        max_zero_score_pct=0.0,
    )

    assert findings == []
    assert len(records) == 4
    assert {summary.engine: summary.scored_record_count for summary in summaries} == {"holyc": 2, "llama": 2}
    assert all(summary.zero_score_pct == 0.0 for summary in summaries)


def test_audit_flags_sparse_duplicate_and_nonfinite_scores(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(
        holyc,
        [
            {"id": "dup", "scores": [1.0, 0.0, 0.0, 0.0]},
            {"id": "dup", "scores": [0.0, 0.0, 0.0, 0.0]},
        ],
    )
    write_jsonl(
        llama,
        [
            {"id": "bad", "scores": [1.0, float("nan"), 0.5, 0.25]},
            {"id": "ok", "scores": [1.0, 2.0, 3.0, 4.0]},
        ],
    )

    records, summaries, findings = eval_score_sparsity_audit.audit(
        holyc,
        llama,
        zero_epsilon=0.0,
        min_records=2,
        min_nonzero_scores_per_record=2,
        min_unique_scores_per_record=2,
        max_zero_score_pct=50.0,
    )
    metrics = {finding.metric for finding in findings}

    assert len(records) == 4
    assert {summary.engine: summary.invalid_record_count for summary in summaries} == {"holyc": 0, "llama": 1}
    assert {
        "duplicate_id",
        "finite",
        "zero_score_pct",
        "nonzero_score_count",
        "unique_score_count",
        "scored_record_count",
    } <= metrics


def test_cli_writes_json_markdown_csv_records_findings_and_junit(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, [{"id": "a", "scores": [2.0, 1.0, 0.5, 0.25]}])
    write_jsonl(llama, [{"id": "a", "scores": [2.1, 1.1, 0.4, 0.2]}])
    output_dir = tmp_path / "out"

    status = eval_score_sparsity_audit.main(
        [
            "--holyc",
            str(holyc),
            "--llama",
            str(llama),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "score_sparsity",
            "--min-records",
            "1",
            "--min-nonzero-scores-per-record",
            "4",
            "--min-unique-scores-per-record",
            "4",
            "--max-zero-score-pct",
            "0",
            "--fail-on-findings",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "score_sparsity.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["records"] == 2
    assert "Eval Score Sparsity Audit" in (output_dir / "score_sparsity.md").read_text(encoding="utf-8")
    assert "zero_score_pct" in (output_dir / "score_sparsity.csv").read_text(encoding="utf-8")
    assert "record_id" in (output_dir / "score_sparsity_records.csv").read_text(encoding="utf-8")
    assert "metric" in (output_dir / "score_sparsity_findings.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "score_sparsity_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_eval_score_sparsity_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-eval-score-sparsity-test-") as tmp:
        test_audit_accepts_dense_finite_scores(Path(tmp) / "pass")
        test_audit_flags_sparse_duplicate_and_nonfinite_scores(Path(tmp) / "fail")
        test_cli_writes_json_markdown_csv_records_findings_and_junit(Path(tmp) / "cli")
    print("test_eval_score_sparsity_audit=ok")
