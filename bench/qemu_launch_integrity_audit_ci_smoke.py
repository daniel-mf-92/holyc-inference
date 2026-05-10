#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU launch integrity audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_bench


def run_command(command: list[str], *, expected_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0 and not expected_failure:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def make_artifact(path: Path, *, stale_hash: bool = False) -> None:
    prompt = qemu_prompt_bench.PromptCase("smoke", "Smoke prompt", expected_tokens=4)
    plan = qemu_prompt_bench.dry_run_launch_plan([prompt], warmup=1, repeat=1)
    rows = [
        {
            "launch_index": row["launch_index"],
            "phase": row["phase"],
            "prompt": row["prompt_id"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_bytes": row["prompt_bytes"],
            "expected_tokens": row["expected_tokens"],
            "iteration": row["iteration"],
        }
        for row in plan
    ]
    expected = qemu_prompt_bench.launch_sequence_from_plan(plan)
    observed = [
        {
            "launch_index": row["launch_index"],
            "phase": row["phase"],
            "prompt_id": row["prompt"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_bytes": row["prompt_bytes"],
            "expected_tokens": row["expected_tokens"],
            "iteration": row["iteration"],
        }
        for row in rows
    ]
    integrity = qemu_prompt_bench.launch_sequence_integrity(expected, observed)
    payload = {
        "launch_plan": plan,
        "launch_plan_sha256": "bad" if stale_hash else qemu_prompt_bench.launch_plan_hash(plan),
        "expected_launch_sequence_sha256": integrity["expected_launch_sequence_sha256"],
        "observed_launch_sequence_sha256": integrity["observed_launch_sequence_sha256"],
        "launch_sequence_integrity": integrity,
        "warmups": [rows[0]],
        "benchmarks": [rows[1]],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-launch-integrity-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing = tmp_path / "qemu_prompt_bench_latest.json"
        make_artifact(passing)
        output_dir = tmp_path / "out"
        completed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_launch_integrity_audit.py"),
                str(passing),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_launch_integrity_audit_smoke",
                "--require-match",
            ]
        )
        if completed.returncode != 0:
            return completed.returncode
        report = json.loads((output_dir / "qemu_launch_integrity_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_launch_integrity_status"):
            return rc
        if rc := require(report["summary"]["matched_launches"] == 2, "unexpected_matched_launches"):
            return rc
        if rc := require(
            "QEMU Launch Integrity Audit" in (output_dir / "qemu_launch_integrity_audit_smoke.md").read_text(encoding="utf-8"),
            "missing_markdown",
        ):
            return rc
        junit = ET.parse(output_dir / "qemu_launch_integrity_audit_smoke_junit.xml").getroot()
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_junit_failure"):
            return rc

        failing = tmp_path / "failing_qemu_prompt_bench_latest.json"
        make_artifact(failing, stale_hash=True)
        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_launch_integrity_audit.py"),
                str(failing),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_launch_integrity_audit_failing",
                "--require-match",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "expected_launch_integrity_failure"):
            return rc
        failed_report = json.loads((output_dir / "qemu_launch_integrity_audit_failing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require("value_mismatch" in kinds, "missing_value_mismatch"):
            return rc

    print("qemu_launch_integrity_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
