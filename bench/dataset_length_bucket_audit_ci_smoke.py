#!/usr/bin/env python3
"""CI smoke runner for dataset_length_bucket_audit.py."""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_length_bucket_audit


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    return dataset_length_bucket_audit.main(
        [
            "--input",
            str(root / "bench/datasets/samples/smoke_eval.jsonl"),
            "--bucket-edges",
            "96,160,256",
            "--min-records",
            "3",
            "--min-covered-buckets",
            "2",
            "--output-dir",
            str(root / "bench/results/datasets"),
            "--output-stem",
            "dataset_length_bucket_audit_smoke_latest",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
