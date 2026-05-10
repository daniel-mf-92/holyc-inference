#!/usr/bin/env python3
"""Smoke gate for eval_slice_coverage_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_slice_coverage_audit


def write_report(path: Path, *, status: str = "pass", accuracy: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "dataset": "smoke-suite",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "regressions": [] if status == "pass" else [{"metric": "accuracy"}],
                "summary": {
                    "record_count": 3,
                    "holyc_accuracy": accuracy,
                    "llama_accuracy": 1.0,
                    "agreement": accuracy,
                    "dataset_breakdown": [
                        {
                            "dataset": "arc-smoke",
                            "split": "validation",
                            "record_count": 2,
                            "holyc_accuracy": accuracy,
                            "llama_accuracy": 1.0,
                            "agreement": accuracy,
                        },
                        {
                            "dataset": "truthfulqa-smoke",
                            "split": "validation",
                            "record_count": 1,
                            "holyc_accuracy": 1.0,
                            "llama_accuracy": 1.0,
                            "agreement": 1.0,
                        },
                    ],
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
        write_report(failing, status="fail", accuracy=0.25)

        ok = eval_slice_coverage_audit.main(
            [
                str(passing),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_slice_coverage_audit_smoke",
                "--require-slice",
                "arc-smoke:validation",
                "--require-slice",
                "truthfulqa-smoke:validation",
                "--min-slices",
                "2",
                "--min-records-per-slice",
                "1",
                "--min-slice-holyc-accuracy",
                "0.95",
                "--min-slice-agreement",
                "0.95",
                "--fail-on-failed-reports",
                "--fail-on-regressions",
            ]
        )
        if ok != 0:
            raise AssertionError("passing slice coverage audit should pass")
        payload = json.loads((output_dir / "eval_slice_coverage_audit_smoke.json").read_text(encoding="utf-8"))
        if payload["status"] != "pass" or payload["summary"]["slices"] != 2:
            raise AssertionError("passing slice coverage payload is incorrect")

        bad = eval_slice_coverage_audit.main(
            [
                str(failing),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_slice_coverage_audit_smoke_fail",
                "--require-slice",
                "hellaswag-smoke:validation",
                "--min-records-per-slice",
                "2",
                "--min-slice-holyc-accuracy",
                "0.95",
                "--min-slice-agreement",
                "0.95",
                "--fail-on-failed-reports",
                "--fail-on-regressions",
            ]
        )
        if bad == 0:
            raise AssertionError("failing slice coverage audit should fail")
        failing_payload = json.loads(
            (output_dir / "eval_slice_coverage_audit_smoke_fail.json").read_text(encoding="utf-8")
        )
        gates = {finding["gate"] for finding in failing_payload["findings"]}
        expected = {
            "required_slice",
            "min_records_per_slice",
            "min_slice_holyc_accuracy",
            "min_slice_agreement",
            "report_status",
            "regressions",
        }
        if not expected.issubset(gates):
            raise AssertionError(f"missing expected gates: {expected - gates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
