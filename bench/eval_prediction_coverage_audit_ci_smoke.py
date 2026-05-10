#!/usr/bin/env python3
"""Smoke gate for eval_prediction_coverage_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_prediction_coverage_audit


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def passing_rows() -> list[dict[str, object]]:
    return [
        {"id": "smoke-hellaswag-1", "prediction": 0},
        {"id": "smoke-arc-1", "prediction": 0},
        {"id": "smoke-truthfulqa-1", "prediction": 0},
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-prediction-coverage-") as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(holyc, passing_rows())
        write_jsonl(llama, passing_rows())

        output_dir = ROOT / "bench" / "results"
        status = eval_prediction_coverage_audit.main(
            [
                "--gold",
                str(GOLD),
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--dataset",
                "smoke-eval",
                "--split",
                "validation",
                "--min-gold-records",
                "3",
                "--min-coverage-pct",
                "100",
                "--min-slice-coverage-pct",
                "100",
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_prediction_coverage_audit_smoke_latest",
                "--fail-on-findings",
            ]
        )
        require(status == 0, "passing coverage smoke should pass")
        payload = json.loads((output_dir / "eval_prediction_coverage_audit_smoke_latest.json").read_text(encoding="utf-8"))
        require(payload["status"] == "pass", "passing coverage payload should pass")
        require(payload["summary"]["paired_coverage_pct"] == 100.0, "paired coverage should be 100%")

        write_jsonl(llama, passing_rows()[:2])
        failing = eval_prediction_coverage_audit.main(
            [
                "--gold",
                str(GOLD),
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--dataset",
                "smoke-eval",
                "--split",
                "validation",
                "--min-coverage-pct",
                "100",
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "failing",
                "--fail-on-findings",
            ]
        )
        require(failing == 1, "missing llama row should fail")
    print("eval_prediction_coverage_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
