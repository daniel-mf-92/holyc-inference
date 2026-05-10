#!/usr/bin/env python3
"""Smoke gate for qemu_serial_payload_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_serial_payload_audit


def bench_row(*, tokens: int = 8, payload_tokens: int = 8) -> dict:
    payload = {
        "tokens": payload_tokens,
        "elapsed_us": 1000,
        "time_to_first_token_us": 100,
        "memory_bytes": 4096,
        "prompt_bytes": 12,
        "prompt_sha256": "abc123",
    }
    return {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt": "smoke",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": tokens,
        "elapsed_us": 1000,
        "ttft_us": 100,
        "memory_bytes": 4096,
        "prompt_bytes": 12,
        "prompt_sha256": "abc123",
        "stdout_tail": "BENCH_RESULT: " + json.dumps(payload) + "\n",
        "stderr_tail": "",
    }


def write_artifact(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        passing = tmp_path / "qemu_prompt_bench_passing.json"
        failing = tmp_path / "qemu_prompt_bench_failing.json"
        out = tmp_path / "out"
        write_artifact(passing, [bench_row()])
        write_artifact(failing, [bench_row(tokens=8, payload_tokens=7)])

        pass_status = qemu_serial_payload_audit.main(
            [
                str(passing),
                "--output-dir",
                str(out),
                "--output-stem",
                "qemu_serial_payload_audit_pass",
            ]
        )
        fail_status = qemu_serial_payload_audit.main(
            [
                str(failing),
                "--output-dir",
                str(out),
                "--output-stem",
                "qemu_serial_payload_audit_fail",
            ]
        )
        if pass_status != 0:
            raise SystemExit("expected passing serial payload audit to succeed")
        if fail_status == 0:
            raise SystemExit("expected mismatched serial payload audit to fail")

    output_dir = Path("bench/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    return qemu_serial_payload_audit.main(
        [
            "bench/results/qemu_prompt_bench_latest.json",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_serial_payload_audit_latest",
            "--min-rows",
            "1",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
