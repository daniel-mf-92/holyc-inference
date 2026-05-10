#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_environment_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_bench


COMMAND = ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def artifact(commit: str = "abc123") -> dict[str, object]:
    command_airgap = qemu_prompt_bench.command_airgap_metadata(COMMAND)
    return {
        "status": "pass",
        "commit": commit,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "command": COMMAND,
        "command_sha256": qemu_prompt_bench.command_hash(COMMAND),
        "command_airgap": command_airgap,
        "environment": {
            "platform": "macOS-15-test",
            "machine": "arm64",
            "processor": "arm",
            "python": "3.14.0",
            "cpu_count": 10,
            "qemu_bin": "qemu-system-x86_64",
            "qemu_path": "/opt/homebrew/bin/qemu-system-x86_64",
            "qemu_version": "QEMU emulator version 9.2.0",
        },
        "benchmarks": [
            {
                "commit": commit,
                "command": COMMAND,
                "command_sha256": qemu_prompt_bench.command_hash(COMMAND),
                "command_airgap_ok": command_airgap["ok"],
                "command_has_explicit_nic_none": command_airgap["explicit_nic_none"],
                "command_has_legacy_net_none": command_airgap["legacy_net_none"],
                "command_airgap_violations": command_airgap["violations"],
            }
        ],
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-env-audit-") as tmp:
        root = Path(tmp)
        artifact_path = root / "qemu_prompt_bench_latest.json"
        artifact_path.write_text(json.dumps(artifact()), encoding="utf-8")
        output_dir = root / "out"
        command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_environment_audit.py"),
            str(artifact_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_environment_audit_latest",
            "--require-qemu-path",
            "--require-qemu-version",
            "--require-row-command-provenance",
            "--min-artifacts",
            "1",
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((output_dir / "qemu_environment_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_environment_audit_status"):
            return rc
        if rc := require(report["summary"]["artifacts"] == 1, "unexpected_environment_artifact_count"):
            return rc
        if rc := require(report["summary"]["artifacts_with_airgap_ok"] == 1, "missing_environment_airgap_rollup"):
            return rc
        if rc := require(
            report["summary"]["row_missing_command_provenance"] == 0,
            "unexpected_missing_row_command_provenance",
        ):
            return rc
        if rc := require(
            report["summary"]["row_command_airgap_mismatches"] == 0,
            "unexpected_row_airgap_mismatch",
        ):
            return rc
        if rc := require((output_dir / "qemu_environment_audit_latest.md").exists(), "missing_environment_markdown"):
            return rc
        if rc := require((output_dir / "qemu_environment_audit_latest.csv").exists(), "missing_environment_csv"):
            return rc
        if rc := require(
            (output_dir / "qemu_environment_audit_latest_junit.xml").exists(),
            "missing_environment_junit",
        ):
            return rc

        bad_artifact = artifact()
        bad_artifact["benchmarks"] = [
            {
                "commit": "abc123",
                "command": COMMAND,
                "command_sha256": "stale",
                "command_airgap_ok": False,
            },
            {"commit": "abc123"},
        ]
        bad_path = root / "qemu_prompt_bench_bad.json"
        bad_path.write_text(json.dumps(bad_artifact), encoding="utf-8")
        bad_out = root / "bad_out"
        bad_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_environment_audit.py"),
            str(bad_path),
            "--output-dir",
            str(bad_out),
            "--output-stem",
            "qemu_environment_audit_latest",
            "--require-row-command-provenance",
        ]
        completed = subprocess.run(bad_command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode == 0:
            print("bad_environment_audit_not_rejected=true", file=sys.stderr)
            return 1
        bad_report = json.loads((bad_out / "qemu_environment_audit_latest.json").read_text(encoding="utf-8"))
        finding_kinds = {finding["kind"] for finding in bad_report["findings"]}
        if rc := require("row_command_hash" in finding_kinds, "missing_bad_row_hash_finding"):
            return rc
        if rc := require("row_command_airgap_drift" in finding_kinds, "missing_bad_row_airgap_finding"):
            return rc
        if rc := require("row_command_missing" in finding_kinds, "missing_bad_row_command_finding"):
            return rc
        if rc := require(
            bad_report["summary"]["row_missing_command_provenance"] >= 1,
            "missing_bad_row_provenance_summary",
        ):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
