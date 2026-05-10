#!/usr/bin/env python3
"""Smoke gate for eval_score_delta_audit.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_score_delta_audit


def main() -> int:
    return eval_score_delta_audit.main(
        [
            "--gold",
            str(ROOT / "bench/datasets/samples/smoke_eval.jsonl"),
            "--holyc",
            str(ROOT / "bench/eval/samples/holyc_smoke_scored_predictions.jsonl"),
            "--llama",
            str(ROOT / "bench/eval/samples/llama_smoke_scored_predictions.jsonl"),
            "--dataset",
            "smoke",
            "--split",
            "validation",
            "--min-pair-coverage-pct",
            "100",
            "--min-top-index-match-pct",
            "100",
            "--max-abs-delta",
            "0.25",
            "--max-mean-abs-delta",
            "0.125",
            "--max-top-score-abs-delta",
            "0.25",
            "--output-dir",
            str(ROOT / "bench/results"),
            "--output-stem",
            "eval_score_delta_audit_smoke_latest",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
