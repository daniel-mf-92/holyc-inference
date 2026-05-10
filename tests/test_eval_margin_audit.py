#!/usr/bin/env python3
"""Host-side checks for eval margin audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
AUDIT_PATH = BENCH_PATH / "eval_margin_audit.py"
spec = importlib.util.spec_from_file_location("eval_margin_audit", AUDIT_PATH)
eval_margin_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_margin_audit"] = eval_margin_audit
spec.loader.exec_module(eval_margin_audit)


def margin(mean: float, p10: float, low_rate: float = 0.0, scored: int = 4) -> dict[str, object]:
    return {
        "low_margin_count": int(scored * low_rate),
        "low_margin_rate": low_rate,
        "low_margin_threshold": 0.1,
        "mean_correct_margin": mean,
        "mean_margin": mean,
        "mean_wrong_margin": 0.0,
        "median_margin": mean,
        "min_margin": min(mean, p10),
        "p10_margin": p10,
        "score_coverage": scored / 4,
        "scored_count": scored,
        "total_count": 4,
    }


def write_report(path: Path, *, holyc_mean: float, llama_mean: float, holyc_p10: float = 0.4) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "unit-eval",
                "split": "validation",
                "model": "synthetic",
                "quantization": "Q4_0",
                "summary": {
                    "holyc_margin_metrics": margin(holyc_mean, holyc_p10),
                    "llama_margin_metrics": margin(llama_mean, 0.5),
                    "dataset_breakdown": [
                        {
                            "dataset": "arc-unit",
                            "split": "validation",
                            "holyc_margin_metrics": margin(holyc_mean, holyc_p10),
                            "llama_margin_metrics": margin(llama_mean, 0.5),
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_margin_audit_passes_clean_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = tmp_path / "eval_compare.json"
        write_report(report, holyc_mean=0.7, llama_mean=0.72)

        assert eval_margin_audit.main(
            [
                str(report),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "audit",
                "--min-score-coverage",
                "1.0",
                "--min-mean-margin",
                "0.5",
                "--max-holyc-mean-margin-loss",
                "0.05",
                "--fail-on-findings",
            ]
        ) == 0

        payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        assert payload["status"] == "pass"
        assert payload["report_count"] == 1
        assert payload["margin_summary_count"] == 4
        assert payload["margin_summaries"][0]["engine"] == "holyc"
        assert payload["findings"] == []
        assert "No margin gate findings" in (tmp_path / "audit.md").read_text(encoding="utf-8")
        assert (tmp_path / "audit_summaries.csv").read_text(encoding="utf-8").startswith("source,")
        junit = ET.parse(tmp_path / "audit_junit.xml").getroot()
        assert junit.attrib["name"] == "holyc_eval_margin_audit"
        assert junit.attrib["failures"] == "0"


def test_margin_audit_fails_thresholds_and_deltas() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = tmp_path / "eval_compare.json"
        write_report(report, holyc_mean=0.15, llama_mean=0.75, holyc_p10=0.02)

        assert eval_margin_audit.main(
            [
                str(report),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "audit",
                "--min-mean-margin",
                "0.5",
                "--min-p10-margin",
                "0.1",
                "--max-holyc-mean-margin-loss",
                "0.2",
                "--include-dataset-breakdown",
                "--fail-on-findings",
            ]
        ) == 2

        payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in payload["findings"]}
        assert payload["status"] == "fail"
        assert "mean_margin" in metrics
        assert "p10_margin" in metrics
        assert "mean_margin_loss_vs_llama" in metrics
        assert "HolyC mean margin loss" in (tmp_path / "audit_junit.xml").read_text(encoding="utf-8")


def test_invalid_gate_returns_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert eval_margin_audit.main(
            [
                str(BENCH_PATH / "results" / "eval_compare_smoke_latest.json"),
                "--output-dir",
                tmp,
                "--min-score-coverage",
                "-1",
            ]
        ) == 2
