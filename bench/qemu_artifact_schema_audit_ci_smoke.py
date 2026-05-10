#!/usr/bin/env python3
"""Smoke gate for QEMU artifact schema audits."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_artifact_schema_audit


def row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "benchmark": "ci-airgap-smoke/Q4_0/smoke-short/1",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "prompt": "smoke-short",
        "prompt_sha256": "0" * 64,
        "command_sha256": "1" * 64,
        "exit_class": "ok",
        "timestamp": "2026-05-01T00:00:01Z",
        "tokens": 16,
        "elapsed_us": 80_000,
        "wall_elapsed_us": 96_000,
        "tok_per_s": 200.0,
        "wall_tok_per_s": 166.666667,
        "us_per_token": 5_000.0,
        "wall_us_per_token": 6_000.0,
        "returncode": 0,
        "timeout_seconds": 30.0,
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]], **overrides: object) -> None:
    payload: dict[str, object] = {
        "generated_at": "2026-05-01T00:00:02Z",
        "status": "pass",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "command": ["qemu-system-x86_64", "-nic", "none"],
        "command_sha256": "1" * 64,
        "command_airgap": {"ok": True, "violations": []},
        "prompt_suite": {"path": "bench/prompts/smoke.jsonl", "sha256": "2" * 64},
        "suite_summary": {"runs": len(rows), "ok_runs": len(rows), "failed_runs": 0},
        "warmups": [],
        "benchmarks": rows,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-artifact-schema-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        failing = root / "qemu_prompt_bench_fail.json"
        write_artifact(passing, [row()])
        write_artifact(failing, [row(tokens=0, timestamp="not-a-timestamp")], command_airgap="", generated_at="not-a-timestamp")
        pass_status = qemu_artifact_schema_audit.main(
            [
                str(passing),
                "--output-dir",
                str(root / "pass"),
                "--output-stem",
                "qemu_artifact_schema_audit_pass",
            ]
        )
        fail_status = qemu_artifact_schema_audit.main(
            [
                str(failing),
                "--output-dir",
                str(root / "fail"),
                "--output-stem",
                "qemu_artifact_schema_audit_fail",
            ]
        )
        if pass_status != 0:
            raise SystemExit("passing schema audit failed")
        if fail_status == 0:
            raise SystemExit("failing schema audit unexpectedly passed")
    print("qemu_artifact_schema_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
