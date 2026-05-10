#!/usr/bin/env python3
"""Tests for host-side eval quantization delta audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_quant_delta_audit


def write_eval(
    path: Path,
    *,
    quantization: str,
    status: str = "pass",
    records: int = 4,
    holyc_accuracy: float = 1.0,
    llama_accuracy: float = 1.0,
    agreement: float = 1.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": quantization,
                "regressions": [] if status == "pass" else [{"metric": "accuracy"}],
                "summary": {
                    "record_count": records,
                    "holyc_accuracy": holyc_accuracy,
                    "llama_accuracy": llama_accuracy,
                    "agreement": agreement,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_comparisons_joins_candidate_and_reference(tmp_path: Path) -> None:
    q4 = tmp_path / "q4.json"
    q8 = tmp_path / "q8.json"
    write_eval(q4, quantization="Q4_0", holyc_accuracy=0.95, agreement=0.97)
    write_eval(q8, quantization="Q8_0", holyc_accuracy=1.0, agreement=1.0)

    reports = [eval_quant_delta_audit.load_eval_report(q4), eval_quant_delta_audit.load_eval_report(q8)]
    comparisons, findings = eval_quant_delta_audit.build_comparisons(reports, [("Q4_0", "Q8_0")])

    assert findings == []
    assert len(comparisons) == 1
    assert comparisons[0].holyc_accuracy_drop == 0.050000000000000044
    assert comparisons[0].agreement_delta == 0.030000000000000027


def test_evaluate_flags_missing_pair_and_quality_drop(tmp_path: Path) -> None:
    q4 = tmp_path / "q4.json"
    q8 = tmp_path / "q8.json"
    write_eval(q4, quantization="Q4_0", status="fail", holyc_accuracy=0.75, llama_accuracy=0.8, agreement=0.7)
    write_eval(q8, quantization="Q8_0", holyc_accuracy=1.0, llama_accuracy=1.0, agreement=1.0)
    reports = [eval_quant_delta_audit.load_eval_report(q4), eval_quant_delta_audit.load_eval_report(q8)]
    comparisons, pair_findings = eval_quant_delta_audit.build_comparisons(reports, [("Q4_0", "Q8_0")])
    args = eval_quant_delta_audit.parse_args(
        [
            str(tmp_path),
            "--fail-on-failed-eval",
            "--fail-on-regressions",
            "--min-comparisons",
            "1",
            "--max-holyc-accuracy-drop",
            "0.05",
            "--max-llama-accuracy-drop",
            "0.05",
            "--max-agreement-delta",
            "0.05",
        ]
    )

    findings = eval_quant_delta_audit.evaluate(comparisons, reports, pair_findings, args)

    assert {
        "candidate_status",
        "candidate_regressions",
        "max_holyc_accuracy_drop",
        "max_llama_accuracy_drop",
        "max_agreement_delta",
    }.issubset({finding.gate for finding in findings})


def test_cli_writes_quant_delta_artifacts(tmp_path: Path) -> None:
    write_eval(tmp_path / "q4.json", quantization="Q4_0", holyc_accuracy=0.98, agreement=0.99)
    write_eval(tmp_path / "q8.json", quantization="Q8_0", holyc_accuracy=1.0, agreement=1.0)
    output_dir = tmp_path / "out"

    status = eval_quant_delta_audit.main(
        [
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "quant_delta",
            "--fail-on-failed-eval",
            "--fail-on-regressions",
            "--min-records",
            "4",
            "--max-holyc-accuracy-drop",
            "0.05",
            "--max-llama-accuracy-drop",
            "0.01",
            "--max-agreement-delta",
            "0.05",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "quant_delta.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["comparisons"] == 1
    rows = list(csv.DictReader((output_dir / "quant_delta.csv").open(encoding="utf-8")))
    assert rows[0]["candidate_quantization"] == "Q4_0"
    markdown = (output_dir / "quant_delta.md").read_text(encoding="utf-8")
    assert "Eval Quant Delta Audit" in markdown
    junit_root = ET.parse(output_dir / "quant_delta_junit.xml").getroot()
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_build_comparisons_joins_candidate_and_reference(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_evaluate_flags_missing_pair_and_quality_drop(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_quant_delta_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
