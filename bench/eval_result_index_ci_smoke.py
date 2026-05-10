#!/usr/bin/env python3
"""Smoke gate for eval_result_index.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_result_index


def write_eval_compare(path: Path, *, status: str = "pass", records: int = 4, accuracy: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:00:00Z",
                "status": status,
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q8_0",
                "gold_sha256": "a" * 64,
                "holyc_predictions_sha256": "b" * 64,
                "llama_predictions_sha256": "c" * 64,
                "regressions": [] if status == "pass" else [{"metric": "accuracy"}],
                "summary": {
                    "record_count": records,
                    "holyc_accuracy": accuracy,
                    "llama_accuracy": 1.0,
                    "accuracy_delta_holyc_minus_llama": accuracy - 1.0,
                    "agreement": accuracy,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_suite_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:01:00Z",
                "status": "pass",
                "summary": {
                    "reports": 1,
                    "records": 4,
                    "findings": 0,
                    "regressions": 0,
                    "weighted_holyc_accuracy": 1.0,
                    "weighted_agreement": 1.0,
                },
                "reports": [],
                "findings": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> int:
    if condition:
        return 0
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output_dir = root / "out"
        write_eval_compare(root / "eval_compare.json", records=4, accuracy=1.0)
        write_suite_summary(root / "eval_suite_summary.json")

        ok = eval_result_index.main(
            [
                str(root),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_result_index_smoke",
                "--min-artifacts",
                "2",
                "--min-records",
                "8",
                "--require-dataset",
                "smoke-eval",
                "--require-quantization",
                "Q8_0",
                "--min-holyc-accuracy",
                "0.99",
                "--min-agreement",
                "0.99",
                "--fail-on-failed",
                "--fail-on-regressions",
            ]
        )
        if rc := require(ok == 0, "passing eval result index failed"):
            return rc
        payload = json.loads((output_dir / "eval_result_index_smoke.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "passing payload status was not pass"):
            return rc
        if rc := require(payload["summary"]["artifacts"] == 2, "artifact count was not indexed"):
            return rc
        if rc := require(payload["summary"]["records"] == 8, "record count was not indexed"):
            return rc

        write_eval_compare(root / "bad_eval_compare.json", status="fail", records=1, accuracy=0.25)
        bad = eval_result_index.main(
            [
                str(root / "bad_eval_compare.json"),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_result_index_smoke_fail",
                "--min-artifacts",
                "2",
                "--min-records",
                "4",
                "--require-dataset",
                "arc",
                "--require-quantization",
                "Q4_0",
                "--min-holyc-accuracy",
                "0.9",
                "--min-agreement",
                "0.9",
                "--fail-on-failed",
                "--fail-on-regressions",
            ]
        )
        if rc := require(bad == 1, "failing eval result index did not fail"):
            return rc
        failing_payload = json.loads((output_dir / "eval_result_index_smoke_fail.json").read_text(encoding="utf-8"))
        gates = {finding["gate"] for finding in failing_payload["findings"]}
        expected = {
            "min_artifacts",
            "min_records",
            "required_dataset",
            "required_quantization",
            "status",
            "regressions",
            "min_holyc_accuracy",
            "min_agreement",
        }
        if rc := require(expected <= gates, f"missing expected failure gates: {expected - gates}"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
