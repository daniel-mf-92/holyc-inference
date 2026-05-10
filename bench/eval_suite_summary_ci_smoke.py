#!/usr/bin/env python3
"""Smoke gate for eval_suite_summary.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_suite_summary


def write_report(path: Path, *, status: str = "pass", accuracy: float = 1.0, agreement: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "gold_sha256": "a" * 64,
                "holyc_predictions_sha256": "b" * 64,
                "llama_predictions_sha256": "c" * 64,
                "regressions": [] if status == "pass" else [{"metric": "accuracy"}],
                "summary": {
                    "record_count": 3,
                    "holyc_accuracy": accuracy,
                    "llama_accuracy": 1.0,
                    "accuracy_delta_holyc_minus_llama": accuracy - 1.0,
                    "agreement": agreement,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        passing = root / "eval_compare_pass.json"
        failing = root / "eval_compare_fail.json"
        output_dir = root / "out"
        write_report(passing)
        write_report(failing, status="fail", accuracy=0.5, agreement=0.5)

        ok = eval_suite_summary.main(
            [
                str(passing),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_suite_summary_smoke",
                "--min-reports",
                "1",
                "--min-records",
                "3",
                "--min-holyc-accuracy",
                "0.95",
                "--min-agreement",
                "0.95",
                "--require-dataset",
                "smoke-eval",
                "--require-split",
                "validation",
                "--require-model",
                "synthetic-smoke",
                "--require-quantization",
                "Q4_0",
                "--fail-on-failed-reports",
                "--fail-on-regressions",
            ]
        )
        if ok != 0:
            raise AssertionError("passing eval suite summary should pass")
        payload = json.loads((output_dir / "eval_suite_summary_smoke.json").read_text(encoding="utf-8"))
        if payload["status"] != "pass" or payload["summary"]["records"] != 3:
            raise AssertionError("passing summary payload is incorrect")

        bad = eval_suite_summary.main(
            [
                str(passing),
                str(failing),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_suite_summary_smoke_fail",
                "--min-holyc-accuracy",
                "0.95",
                "--min-agreement",
                "0.95",
                "--require-dataset",
                "arc",
                "--require-quantization",
                "Q8_0",
                "--fail-on-failed-reports",
                "--fail-on-regressions",
            ]
        )
        if bad == 0:
            raise AssertionError("failing eval suite summary should fail")
        failing_payload = json.loads((output_dir / "eval_suite_summary_smoke_fail.json").read_text(encoding="utf-8"))
        gates = {finding["gate"] for finding in failing_payload["findings"]}
        expected = {
            "required_dataset",
            "required_quantization",
            "report_status",
            "regressions",
            "min_holyc_accuracy",
            "min_agreement",
        }
        if not expected.issubset(gates):
            raise AssertionError(f"missing expected gates: {expected - gates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
