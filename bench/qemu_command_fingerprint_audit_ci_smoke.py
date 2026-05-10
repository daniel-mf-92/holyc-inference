#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_command_fingerprint_audit.py."""

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


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def row(command: list[str], **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "iteration": 1,
        "launch_index": 1,
        "command": command,
        "command_sha256": qemu_prompt_bench.command_hash(command),
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_command_fingerprint_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_command_fingerprint_audit_latest",
            "--require-top-command",
            "--require-single-command-hash",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-command-fingerprint-") as tmp:
        root = Path(tmp)
        command = ["qemu-system-x86_64", "-nic", "none", "-m", "512M"]
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(
            json.dumps(
                {
                    "command": command,
                    "command_sha256": qemu_prompt_bench.command_hash(command),
                    "benchmarks": [row(command)],
                }
            ),
            encoding="utf-8",
        )
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((pass_dir / "qemu_command_fingerprint_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_command_fingerprint_pass_status"):
            return rc
        if rc := require(report["summary"]["unique_row_command_hashes"] == 1, "missing_command_fingerprint_summary"):
            return rc
        if rc := require(
            "No command fingerprint findings." in (pass_dir / "qemu_command_fingerprint_audit_latest.md").read_text(encoding="utf-8"),
            "missing_command_fingerprint_markdown",
        ):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_command_fingerprint_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_command_fingerprint_junit_failure"):
            return rc

        bad_command = ["qemu-system-x86_64", "-nic", "user"]
        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "command": command,
                    "command_sha256": qemu_prompt_bench.command_hash(command),
                    "benchmarks": [
                        row(command, prompt="bad-hash", command_sha256="bad"),
                        row(bad_command, prompt="networked", launch_index=2),
                    ],
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "bad_command_fingerprint_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_command_fingerprint_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require(
            {"command_sha256_mismatch", "command_airgap_violation", "row_command_hash_drift", "multiple_row_command_hashes"} <= kinds,
            "command_fingerprint_findings_not_reported",
        ):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
