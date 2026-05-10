#!/usr/bin/env python3
"""CI smoke gate for eval artifact drift auditing."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_artifact_drift_audit


def write_report(path: Path, *, quantization: str = "Q4_0", gold_sha256: str = "a" * 64) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": quantization,
                "gold_sha256": gold_sha256,
                "holyc_predictions_sha256": "b" * 64,
                "llama_predictions_sha256": "c" * 64,
                "summary": {
                    "record_count": 3,
                    "holyc_accuracy": 1.0,
                    "llama_accuracy": 1.0,
                    "agreement": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        q4 = tmp_path / "eval_q4.json"
        q8 = tmp_path / "eval_q8.json"
        drift = tmp_path / "eval_q8_drift.json"
        write_report(q4, quantization="Q4_0")
        write_report(q8, quantization="Q8_0")
        write_report(drift, quantization="Q8_0", gold_sha256="d" * 64)
        fail_status = eval_artifact_drift_audit.main(
            [
                str(q4),
                str(drift),
                "--output-dir",
                str(tmp_path / "fail"),
                "--require-hashes",
                "--fail-on-duplicate-key-drift",
            ]
        )
        if fail_status == 0:
            return 1
        return eval_artifact_drift_audit.main(
            [
                str(q4),
                str(q8),
                "--output-dir",
                "bench/results",
                "--output-stem",
                "eval_artifact_drift_audit_smoke_latest",
                "--min-reports",
                "2",
                "--require-hashes",
                "--fail-on-failed-reports",
                "--fail-on-duplicate-key-drift",
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())
