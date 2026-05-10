#!/usr/bin/env python3
"""Tests for host-side eval efficiency frontier reports."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_efficiency_frontier


def write_scorecard(path: Path, *, include_missing: bool = False) -> None:
    rows = [
        {
            "status": "pass",
            "model": "synthetic-smoke",
            "quantization": "Q4_0",
            "dataset": "smoke-eval",
            "split": "validation",
            "records": 8,
            "holyc_accuracy": 0.92,
            "median_wall_tok_per_s": 180.0,
            "max_memory_bytes": 64000000,
        },
        {
            "status": "pass",
            "model": "synthetic-smoke",
            "quantization": "Q8_0",
            "dataset": "smoke-eval",
            "split": "validation",
            "records": 8,
            "holyc_accuracy": 0.98,
            "median_wall_tok_per_s": 130.0,
            "max_memory_bytes": 72000000,
        },
        {
            "status": "pass",
            "model": "synthetic-smoke",
            "quantization": "Q2_K",
            "dataset": "smoke-eval",
            "split": "validation",
            "records": 8,
            "holyc_accuracy": 0.90,
            "median_wall_tok_per_s": 120.0,
            "max_memory_bytes": 60000000,
        },
    ]
    if include_missing:
        rows.append(
            {
                "status": "pass",
                "model": "synthetic-smoke",
                "quantization": "BROKEN",
                "dataset": "smoke-eval",
                "split": "validation",
                "records": 8,
                "holyc_accuracy": 0.5,
            }
        )
    path.write_text(json.dumps({"status": "pass", "scorecard": rows}, sort_keys=True) + "\n", encoding="utf-8")


def test_frontier_marks_non_dominated_rows(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    write_scorecard(scorecard)

    rows = eval_efficiency_frontier.mark_frontier(
        eval_efficiency_frontier.load_rows([scorecard], "holyc_accuracy", "median_wall_tok_per_s")
    )

    frontier = {row.quantization for row in rows if row.frontier}
    dominated = {row.quantization: row.dominated_by for row in rows if row.dominated_by}
    assert frontier == {"Q4_0", "Q8_0"}
    assert dominated["Q2_K"].startswith("synthetic-smoke:Q4_0:")


def test_memory_aware_frontier_preserves_lower_memory_tradeoff(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    write_scorecard(scorecard)

    rows = eval_efficiency_frontier.mark_frontier(
        eval_efficiency_frontier.load_rows([scorecard], "holyc_accuracy", "median_wall_tok_per_s"),
        memory_aware=True,
    )

    frontier = {row.quantization for row in rows if row.frontier}
    assert frontier == {"Q4_0", "Q8_0", "Q2_K"}


def test_evaluate_flags_missing_metrics_and_required_quantization(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    write_scorecard(scorecard, include_missing=True)
    rows = eval_efficiency_frontier.mark_frontier(
        eval_efficiency_frontier.load_rows([scorecard], "holyc_accuracy", "median_wall_tok_per_s")
    )
    args = eval_efficiency_frontier.parse_args(
        [
            str(scorecard),
            "--fail-on-missing-metrics",
            "--require-frontier-quantization",
            "Q5_0",
        ]
    )

    findings = eval_efficiency_frontier.evaluate(rows, args)

    assert {"missing_metric", "require_frontier_quantization"}.issubset({finding.gate for finding in findings})


def test_cli_writes_frontier_artifacts(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    output_dir = tmp_path / "out"
    write_scorecard(scorecard)

    status = eval_efficiency_frontier.main(
        [
            str(scorecard),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "frontier",
            "--min-rows",
            "3",
            "--min-frontier-rows",
            "2",
            "--require-frontier-quantization",
            "Q4_0",
            "--require-frontier-quantization",
            "Q8_0",
            "--memory-aware",
            "--fail-on-failed-scorecard",
            "--fail-on-missing-metrics",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "frontier.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["frontier_rows"] == 3
    assert payload["summary"]["memory_aware"] is True
    rows = list(csv.DictReader((output_dir / "frontier.csv").open(encoding="utf-8")))
    assert {row["quantization"] for row in rows if row["frontier"] == "True"} == {"Q4_0", "Q8_0", "Q2_K"}
    markdown = (output_dir / "frontier.md").read_text(encoding="utf-8")
    assert "Eval Efficiency Frontier" in markdown
    junit_root = ET.parse(output_dir / "frontier_junit.xml").getroot()
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_frontier_marks_non_dominated_rows(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_memory_aware_frontier_preserves_lower_memory_tradeoff(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_evaluate_flags_missing_metrics_and_required_quantization(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_frontier_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
