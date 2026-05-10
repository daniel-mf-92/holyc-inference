#!/usr/bin/env python3
"""Host-side checks for eval rank audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
AUDIT_PATH = BENCH_PATH / "eval_rank_audit.py"
spec = importlib.util.spec_from_file_location("eval_rank_audit", AUDIT_PATH)
eval_rank_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_rank_audit"] = eval_rank_audit
spec.loader.exec_module(eval_rank_audit)


def rank(top1: float, mrr: float, scored: int = 4) -> dict[str, object]:
    return {
        "mean_gold_rank": 1.0 / mrr if mrr else 99.0,
        "mean_reciprocal_rank": mrr,
        "score_coverage": scored / 4,
        "scored_count": scored,
        "top_1_accuracy": top1,
        "top_2_accuracy": max(top1, 0.75),
        "top_3_accuracy": max(top1, 1.0),
        "total_count": 4,
    }


def write_report(path: Path, *, holyc_top1: float, llama_top1: float, holyc_mrr: float = 0.9) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "unit-eval",
                "split": "validation",
                "model": "synthetic",
                "quantization": "Q4_0",
                "summary": {
                    "holyc_rank_metrics": rank(holyc_top1, holyc_mrr),
                    "llama_rank_metrics": rank(llama_top1, 0.92),
                    "dataset_breakdown": [
                        {
                            "dataset": "arc-unit",
                            "split": "validation",
                            "holyc_rank_metrics": rank(holyc_top1, holyc_mrr),
                            "llama_rank_metrics": rank(llama_top1, 0.92),
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_rank_audit_passes_clean_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = tmp_path / "eval_compare.json"
        write_report(report, holyc_top1=0.9, llama_top1=0.92, holyc_mrr=0.91)

        assert eval_rank_audit.main(
            [
                str(report),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "audit",
                "--min-score-coverage",
                "1.0",
                "--min-top-1-accuracy",
                "0.85",
                "--min-mean-reciprocal-rank",
                "0.9",
                "--max-holyc-top-1-loss",
                "0.05",
                "--fail-on-findings",
            ]
        ) == 0

        payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        assert payload["status"] == "pass"
        assert payload["rank_summary_count"] == 4
        assert payload["rank_summaries"][0]["engine"] == "holyc"
        assert payload["findings"] == []
        assert "No rank gate findings" in (tmp_path / "audit.md").read_text(encoding="utf-8")
        assert (tmp_path / "audit_summaries.csv").read_text(encoding="utf-8").startswith("source,")
        junit = ET.parse(tmp_path / "audit_junit.xml").getroot()
        assert junit.attrib["name"] == "holyc_eval_rank_audit"
        assert junit.attrib["failures"] == "0"


def test_rank_audit_fails_thresholds_and_deltas() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = tmp_path / "eval_compare.json"
        write_report(report, holyc_top1=0.4, llama_top1=0.9, holyc_mrr=0.5)

        assert eval_rank_audit.main(
            [
                str(report),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "audit",
                "--min-top-1-accuracy",
                "0.8",
                "--min-mean-reciprocal-rank",
                "0.8",
                "--max-holyc-top-1-loss",
                "0.2",
                "--max-holyc-mrr-loss",
                "0.2",
                "--include-dataset-breakdown",
                "--fail-on-findings",
            ]
        ) == 2

        payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in payload["findings"]}
        assert payload["status"] == "fail"
        assert "top_1_accuracy" in metrics
        assert "mean_reciprocal_rank" in metrics
        assert "top_1_accuracy_loss_vs_llama" in metrics
        assert "mean_reciprocal_rank_loss_vs_llama" in metrics
        assert "HolyC MRR loss" in (tmp_path / "audit_junit.xml").read_text(encoding="utf-8")


def test_invalid_gate_returns_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert eval_rank_audit.main(
            [
                str(BENCH_PATH / "results" / "eval_compare_smoke_latest.json"),
                "--output-dir",
                tmp,
                "--min-score-coverage",
                "-1",
            ]
        ) == 2
