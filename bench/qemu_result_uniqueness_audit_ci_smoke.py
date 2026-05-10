#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_result_uniqueness_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def row(prompt: str = "smoke-short", iteration: int = 1, launch_index: int = 1, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt": prompt,
        "prompt_sha256": f"sha-{prompt}",
        "phase": "measured",
        "iteration": iteration,
        "launch_index": launch_index,
        "commit": "abc123",
        "command_sha256": "cmd123",
        "tokens": 32,
        "elapsed_us": 30_000,
        "exit_class": "ok",
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_result_uniqueness_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_result_uniqueness_audit_latest",
            "--min-rows",
            "1",
            "--require-launch-index",
            "--require-iteration",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-result-unique-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps({"benchmarks": [row(iteration=1), row(iteration=2, launch_index=2)]}), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((pass_dir / "qemu_result_uniqueness_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_result_uniqueness_pass_status"):
            return rc
        if rc := require(report["summary"]["unique_identities"] == 2, "unexpected_unique_identity_count"):
            return rc
        if rc := require(
            "No result uniqueness findings." in (pass_dir / "qemu_result_uniqueness_audit_latest.md").read_text(encoding="utf-8"),
            "missing_result_uniqueness_markdown",
        ):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_result_uniqueness_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_result_uniqueness_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(json.dumps({"benchmarks": [row(), row()]}), encoding="utf-8")
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "duplicate_result_identity_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_result_uniqueness_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require("duplicate_result_identity" in kinds, "duplicate_result_identity_not_reported"):
            return rc

        missing = root / "qemu_prompt_bench_missing.json"
        missing.write_text(json.dumps({"benchmarks": [row(launch_index=None)]}), encoding="utf-8")
        missing_dir = root / "missing"
        missing_run = run_audit(missing, missing_dir)
        if rc := require(missing_run.returncode == 1, "missing_launch_index_not_rejected"):
            return rc
        missing_report = json.loads((missing_dir / "qemu_result_uniqueness_audit_latest.json").read_text(encoding="utf-8"))
        missing_kinds = {finding["kind"] for finding in missing_report["findings"]}
        if rc := require("missing_launch_index" in missing_kinds, "missing_launch_index_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
