#!/usr/bin/env python3
"""Host-side tests for eval artifact drift auditing."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_artifact_drift_audit


def write_report(
    path: Path,
    *,
    dataset: str = "smoke-eval",
    split: str = "validation",
    model: str = "synthetic-smoke",
    quantization: str = "Q4_0",
    status: str = "pass",
    gold_sha256: str = "a" * 64,
    holyc_sha256: str = "b" * 64,
    llama_sha256: str = "c" * 64,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "dataset": dataset,
                "split": split,
                "model": model,
                "quantization": quantization,
                "gold_sha256": gold_sha256,
                "holyc_predictions_sha256": holyc_sha256,
                "llama_predictions_sha256": llama_sha256,
                "summary": {"record_count": 4},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_gold_hash_drift_is_flagged(tmp_path: Path) -> None:
    q4 = tmp_path / "q4.json"
    q8 = tmp_path / "q8.json"
    write_report(q4, quantization="Q4_0", gold_sha256="a" * 64)
    write_report(q8, quantization="Q8_0", gold_sha256="d" * 64)

    records = [eval_artifact_drift_audit.load_record(q4), eval_artifact_drift_audit.load_record(q8)]
    args = eval_artifact_drift_audit.parse_args([str(q4), str(q8), "--require-hashes"])
    findings = eval_artifact_drift_audit.evaluate(records, args)

    assert "gold_sha256_drift" in {finding.gate for finding in findings}


def test_duplicate_report_key_signature_drift_is_flagged(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    write_report(left, holyc_sha256="b" * 64)
    write_report(right, holyc_sha256="e" * 64)

    records = [eval_artifact_drift_audit.load_record(left), eval_artifact_drift_audit.load_record(right)]
    args = eval_artifact_drift_audit.parse_args([str(left), str(right), "--fail-on-duplicate-key-drift"])
    findings = eval_artifact_drift_audit.evaluate(records, args)

    assert "duplicate_report_key_drift" in {finding.gate for finding in findings}


def test_cli_writes_pass_artifacts(tmp_path: Path) -> None:
    q4 = tmp_path / "q4.json"
    q8 = tmp_path / "q8.json"
    output_dir = tmp_path / "out"
    write_report(q4, quantization="Q4_0")
    write_report(q8, quantization="Q8_0")

    status = eval_artifact_drift_audit.main(
        [
            str(q4),
            str(q8),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "drift",
            "--min-reports",
            "2",
            "--require-hashes",
            "--fail-on-duplicate-key-drift",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "drift.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["reports"] == 2
    assert payload["summary"]["dataset_splits"] == 1
    assert list(csv.DictReader((output_dir / "drift_findings.csv").open(encoding="utf-8"))) == []
    assert "No eval artifact drift findings" in (output_dir / "drift.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "drift_junit.xml").getroot()
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_gold_hash_drift_is_flagged(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_duplicate_report_key_signature_drift_is_flagged(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_pass_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
