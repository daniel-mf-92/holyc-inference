#!/usr/bin/env python3
"""Smoke gate for qemu_timing_consistency_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qemu_timing_consistency_audit


def benchmark_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "benchmark": "synthetic-smoke",
        "profile": "ci-airgap-smoke",
        "model": "toy",
        "quantization": "Q4_0",
        "phase": "measured",
        "launch_index": 0,
        "prompt": "Count to four.",
        "iteration": 1,
        "tokens": 4,
        "elapsed_us": 200_000,
        "wall_elapsed_us": 250_000,
        "timeout_seconds": 5.0,
        "tok_per_s": 20.0,
        "wall_tok_per_s": 16.0,
        "us_per_token": 50_000.0,
        "wall_us_per_token": 62_500.0,
        "host_overhead_us": 50_000,
        "host_overhead_pct": 25.0,
        "wall_timeout_pct": 5.0,
        "host_child_user_cpu_us": 120_000,
        "host_child_system_cpu_us": 30_000,
        "host_child_cpu_us": 150_000,
        "ttft_us": 100_000,
    }
    row.update(overrides)
    return row


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        artifact = tmp / "qemu_prompt_bench_latest.json"
        output_dir = tmp / "out"
        artifact.write_text(json.dumps({"status": "pass", "benchmarks": [benchmark_row()]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        status = qemu_timing_consistency_audit.main(
            [
                str(artifact),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_timing_consistency_audit_smoke",
            ]
        )
        if status != 0:
            return status
        payload = json.loads((output_dir / "qemu_timing_consistency_audit_smoke.json").read_text(encoding="utf-8"))
        if payload["status"] != "pass" or payload["summary"]["rows"] != 1:
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "status": "fail",
                    "benchmarks": [
                        benchmark_row(
                            tok_per_s=99.0,
                            wall_elapsed_us=150_000,
                            wall_tok_per_s=26.666666666666668,
                            wall_us_per_token=37_500.0,
                            host_overhead_us=-50_000,
                            host_overhead_pct=-25.0,
                            wall_timeout_pct=3.0,
                            ttft_us=300_000,
                        )
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        status = qemu_timing_consistency_audit.main(
            [
                str(artifact),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_timing_consistency_audit_smoke_fail",
            ]
        )
        if status == 0:
            return 1
        fail_payload = json.loads((output_dir / "qemu_timing_consistency_audit_smoke_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_payload["findings"]}
        return 0 if {"timing_metric_drift", "ttft_exceeds_elapsed", "wall_elapsed_before_guest_elapsed"} <= kinds else 1


if __name__ == "__main__":
    raise SystemExit(main())
