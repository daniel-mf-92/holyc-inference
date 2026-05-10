#!/usr/bin/env python3
"""Host-side checks for eval calibration audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
AUDIT_PATH = BENCH_PATH / "eval_calibration_audit.py"
spec = importlib.util.spec_from_file_location("eval_calibration_audit", AUDIT_PATH)
eval_calibration_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_calibration_audit"] = eval_calibration_audit
spec.loader.exec_module(eval_calibration_audit)


def write_report(path: Path, *, holyc_ece: float, llama_ece: float, holyc_brier: float = 0.02) -> None:
    def calibration(ece: float, brier: float, scored: int = 4) -> dict[str, object]:
        return {
            "accuracy_when_scored": 0.75,
            "brier_score": brier,
            "calibration_bins": [],
            "ece": ece,
            "mean_confidence": 0.80,
            "score_coverage": scored / 4,
            "scored_count": scored,
            "total_count": 4,
        }

    path.write_text(
        json.dumps(
            {
                "dataset": "unit-eval",
                "split": "validation",
                "model": "synthetic",
                "quantization": "Q4_0",
                "summary": {
                    "holyc_calibration": calibration(holyc_ece, holyc_brier),
                    "llama_calibration": calibration(llama_ece, 0.01),
                },
            }
        ),
        encoding="utf-8",
    )


def test_calibration_audit_passes_clean_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = tmp_path / "eval_compare.json"
        write_report(report, holyc_ece=0.02, llama_ece=0.03)

        assert eval_calibration_audit.main(
            [
                str(report),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "audit",
                "--min-score-coverage",
                "1.0",
                "--max-ece",
                "0.05",
                "--fail-on-findings",
            ]
        ) == 0

        payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        assert payload["status"] == "pass"
        assert payload["report_count"] == 1
        assert payload["engine_summary_count"] == 2
        assert payload["engine_summaries"][0]["engine"] == "holyc"
        assert payload["findings"] == []
        assert "No calibration gate findings" in (tmp_path / "audit.md").read_text(encoding="utf-8")
        junit = ET.parse(tmp_path / "audit_junit.xml").getroot()
        assert junit.attrib["name"] == "holyc_eval_calibration_audit"
        assert junit.attrib["failures"] == "0"


def test_calibration_audit_fails_thresholds_and_deltas() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = tmp_path / "eval_compare.json"
        write_report(report, holyc_ece=0.20, llama_ece=0.01, holyc_brier=0.25)

        assert eval_calibration_audit.main(
            [
                str(report),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "audit",
                "--max-ece",
                "0.05",
                "--max-brier-score",
                "0.10",
                "--max-holyc-ece-delta",
                "0.05",
                "--max-holyc-brier-delta",
                "0.05",
                "--fail-on-findings",
            ]
        ) == 2

        payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in payload["findings"]}
        assert payload["status"] == "fail"
        assert "ece" in metrics
        assert "brier_score" in metrics
        assert "ece_delta_holyc_minus_llama" in metrics
        assert "brier_delta_holyc_minus_llama" in metrics
        assert "HolyC ECE delta" in (tmp_path / "audit_junit.xml").read_text(encoding="utf-8")


def test_invalid_gate_returns_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert eval_calibration_audit.main(
            [
                str(BENCH_PATH / "results" / "eval_compare_smoke_latest.json"),
                "--output-dir",
                tmp,
                "--min-score-coverage",
                "-1",
            ]
        ) == 2
