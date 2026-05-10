#!/usr/bin/env python3
"""CI smoke gate for eval efficiency frontier reports."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_efficiency_frontier


def write_scorecard(path: Path) -> None:
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
        {
            "status": "pass",
            "model": "synthetic-smoke",
            "quantization": "Q3_0",
            "dataset": "smoke-eval",
            "split": "validation",
            "records": 8,
            "holyc_accuracy": 0.91,
            "median_wall_tok_per_s": 150.0,
            "max_memory_bytes": 68000000,
        },
    ]
    path.write_text(json.dumps({"status": "pass", "scorecard": rows}, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        scorecard = tmp_path / "scorecard.json"
        write_scorecard(scorecard)
        return eval_efficiency_frontier.main(
            [
                str(scorecard),
                "--output-dir",
                "bench/results",
                "--output-stem",
                "eval_efficiency_frontier_smoke_latest",
                "--min-rows",
                "4",
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


if __name__ == "__main__":
    raise SystemExit(main())
