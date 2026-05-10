#!/usr/bin/env python3
"""CI smoke gate for eval prompt-length bucket reporting."""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_length_bucket_report


def main() -> int:
    return eval_length_bucket_report.main(
        [
            "--gold",
            "bench/datasets/samples/smoke_eval.jsonl",
            "--holyc",
            "bench/eval/samples/holyc_smoke_scored_predictions.jsonl",
            "--llama",
            "bench/eval/samples/llama_smoke_scored_predictions.jsonl",
            "--dataset",
            "smoke-eval",
            "--split",
            "validation",
            "--bucket-edges",
            "128,256,512",
            "--min-records-per-bucket",
            "1",
            "--min-holyc-accuracy",
            "0.95",
            "--max-holyc-accuracy-loss",
            "0.05",
            "--output-dir",
            "bench/results",
            "--output-stem",
            "eval_length_bucket_report_smoke_latest",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
