#!/usr/bin/env python3
"""Smoke test for qemu_matrix_result_coverage_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_matrix_result_coverage_audit


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        matrix = root / "matrix.json"
        result = root / "qemu_prompt_bench_latest.json"
        launch = {
            "build": "candidate",
            "command_sha256": "abc123",
            "profile": "ci",
            "model": "synthetic",
            "quantization": "Q4_0",
            "phase": "measured",
            "prompt_id": "tiny",
            "prompt_sha256": "f" * 64,
            "iteration": 1,
        }
        matrix.write_text(
            json.dumps(
                {
                    "builds": [{"build": "candidate", "command_sha256": "abc123"}],
                    "launches": [launch],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        result.write_text(json.dumps({"benchmarks": [dict(launch, prompt="tiny")]}), encoding="utf-8")

        status = qemu_matrix_result_coverage_audit.main(
            [
                str(matrix),
                str(root),
                "--output-dir",
                "bench/results",
                "--output-stem",
                "qemu_matrix_result_coverage_audit_smoke_latest",
            ]
        )
        if status != 0:
            return status
        payload = json.loads(Path("bench/results/qemu_matrix_result_coverage_audit_smoke_latest.json").read_text(encoding="utf-8"))
        return 0 if payload["summary"]["covered_launch_keys"] == 1 and payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
