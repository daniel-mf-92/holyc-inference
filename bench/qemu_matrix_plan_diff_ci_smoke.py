#!/usr/bin/env python3
"""Smoke gate for qemu_matrix_plan_diff.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_matrix_plan_diff


def matrix_payload(*, launch_count: int = 2, command_hash: str = "cmd-a") -> dict[str, object]:
    launches = []
    for iteration in range(launch_count):
        launches.append(
            {
                "build": "base",
                "profile": "smoke",
                "model": "toy",
                "quantization": "Q8_0",
                "phase": "measured",
                "prompt_id": "alpha",
                "prompt_sha256": "prompt-a",
                "iteration": iteration,
            }
        )
    return {
        "generated_at": "2026-05-02T00:00:00Z",
        "status": "pass",
        "builds": [
            {
                "build": "base",
                "profile": "smoke",
                "model": "toy",
                "quantization": "Q8_0",
                "command_sha256": command_hash,
                "launch_plan_sha256": f"plan-{launch_count}",
                "launch_count": launch_count,
            }
        ],
        "launches": launches,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_smoke(output_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        baseline = tmp_path / "baseline.json"
        candidate_pass = tmp_path / "candidate_pass.json"
        candidate_fail = tmp_path / "candidate_fail.json"
        write_json(baseline, matrix_payload())
        write_json(candidate_pass, matrix_payload())
        write_json(candidate_fail, matrix_payload(launch_count=3, command_hash="cmd-b"))

        pass_report = qemu_matrix_plan_diff.compare_matrix_plans(baseline, candidate_pass)
        if pass_report["status"] != "pass":
            raise AssertionError("identical plans should pass")

        fail_report = qemu_matrix_plan_diff.compare_matrix_plans(baseline, candidate_fail)
        if fail_report["status"] != "fail":
            raise AssertionError("launch and command drift should fail")
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if "build_command_hash_drift" not in kinds or "extra_launch" not in kinds:
            raise AssertionError(f"missing expected drift findings: {sorted(kinds)}")

        status = qemu_matrix_plan_diff.main(
            [
                str(baseline),
                str(candidate_pass),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_matrix_plan_diff_smoke_latest",
            ]
        )
        if status != 0:
            raise AssertionError("CLI smoke report should pass")


def main() -> int:
    output_dir = Path("bench/results")
    run_smoke(output_dir)
    print(f"wrote {output_dir / 'qemu_matrix_plan_diff_smoke_latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
