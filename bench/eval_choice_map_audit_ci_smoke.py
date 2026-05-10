#!/usr/bin/env python3
"""CI smoke for eval_choice_map_audit.py."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_choice_map_audit


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        status = eval_choice_map_audit.main(
            [
                "--gold",
                str(root / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"),
                "--holyc",
                str(root / "bench" / "eval" / "samples" / "holyc_smoke_predictions.jsonl"),
                "--llama",
                str(root / "bench" / "eval" / "samples" / "llama_smoke_predictions.jsonl"),
                "--dataset",
                "smoke-eval",
                "--split",
                "validation",
                "--min-valid-pct",
                "100",
                "--output-dir",
                tmp,
                "--output-stem",
                "eval_choice_map_audit_smoke",
            ]
        )
        return status


if __name__ == "__main__":
    raise SystemExit(main())
