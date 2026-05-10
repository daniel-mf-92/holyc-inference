#!/usr/bin/env python3
"""Stdlib-only smoke gate for qemu_launch_jsonl_parity_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_artifact(path: Path, *, corrupt_sidecar: bool = False) -> None:
    payload = {
        "generated_at": "2026-05-02T00:00:00Z",
        "status": "pass",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "command_sha256": "c" * 64,
        "command_airgap": {
            "ok": True,
            "explicit_nic_none": True,
            "legacy_net_none": False,
            "violations": [],
        },
        "launch_plan_sha256": "a" * 64,
        "expected_launch_sequence_sha256": "b" * 64,
        "observed_launch_sequence_sha256": "b" * 64,
        "prompt_suite": {"suite_sha256": "d" * 64},
        "warmups": [
            {
                "benchmark": "qemu_prompt",
                "phase": "warmup",
                "launch_index": 1,
                "iteration": 1,
                "prompt": "smoke-short",
                "prompt_sha256": "e" * 64,
                "tokens": 4,
                "elapsed_us": 1000,
                "wall_elapsed_us": 1200,
                "returncode": 0,
                "timed_out": False,
                "exit_class": "ok",
            }
        ],
        "benchmarks": [
            {
                "benchmark": "qemu_prompt",
                "phase": "measured",
                "launch_index": 2,
                "iteration": 1,
                "prompt": "smoke-short",
                "prompt_sha256": "e" * 64,
                "tokens": 4,
                "elapsed_us": 1000,
                "wall_elapsed_us": 1200,
                "returncode": 0,
                "timed_out": False,
                "exit_class": "ok",
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for row in payload["warmups"] + payload["benchmarks"]:
        merged = {
            "generated_at": payload["generated_at"],
            "status": payload["status"],
            "profile": payload["profile"],
            "model": payload["model"],
            "quantization": payload["quantization"],
            "commit": payload["commit"],
            "command_sha256": payload["command_sha256"],
            "command_airgap_ok": True,
            "command_has_explicit_nic_none": True,
            "command_has_legacy_net_none": False,
            "command_airgap_violations": [],
            "launch_plan_sha256": payload["launch_plan_sha256"],
            "expected_launch_sequence_sha256": payload["expected_launch_sequence_sha256"],
            "observed_launch_sequence_sha256": payload["observed_launch_sequence_sha256"],
            "prompt_suite_sha256": payload["prompt_suite"]["suite_sha256"],
            **row,
        }
        rows.append(merged)
    if corrupt_sidecar:
        rows[1]["tokens"] = 5
    sidecar = path.with_name("qemu_prompt_bench_launches_latest.jsonl")
    sidecar.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-launch-jsonl-parity-") as tmp:
        tmp_path = Path(tmp)
        passing = tmp_path / "pass" / "qemu_prompt_bench_latest.json"
        failing = tmp_path / "fail" / "qemu_prompt_bench_latest.json"
        passing.parent.mkdir()
        failing.parent.mkdir()
        write_artifact(passing)
        write_artifact(failing, corrupt_sidecar=True)

        out_dir = tmp_path / "out"
        pass_run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_launch_jsonl_parity_audit.py"),
                str(passing),
                "--output-dir",
                str(out_dir),
                "--output-stem",
                "pass",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if pass_run.returncode != 0:
            sys.stdout.write(pass_run.stdout)
            sys.stderr.write(pass_run.stderr)
            return pass_run.returncode

        fail_run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_launch_jsonl_parity_audit.py"),
                str(failing),
                "--output-dir",
                str(out_dir),
                "--output-stem",
                "fail",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if rc := require(fail_run.returncode == 1, "expected_corrupt_sidecar_failure"):
            return rc
        report = json.loads((out_dir / "fail.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "fail", "missing_fail_status"):
            return rc
        if rc := require(report["findings"][0]["kind"] == "value_mismatch", "missing_value_mismatch"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
