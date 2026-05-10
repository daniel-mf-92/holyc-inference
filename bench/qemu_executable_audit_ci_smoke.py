#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_executable_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def row(cmd: list[str], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": "smoke-short",
        "phase": "measured",
        "command": cmd,
        "exit_class": "ok",
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, command: list[str], rows: list[dict[str, object]], **overrides: object) -> None:
    payload: dict[str, object] = {
        "environment": {"qemu_bin": Path(command[0]).name, "qemu_path": command[0]},
        "command": command,
        "benchmarks": rows,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_executable_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_executable_audit_latest",
            "--min-rows",
            "1",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-executable-audit-") as tmp:
        root = Path(tmp)
        qemu = "/opt/homebrew/bin/qemu-system-x86_64"
        passing = root / "qemu_prompt_bench_pass.json"
        write_artifact(passing, [qemu, "-nic", "none"], [row([qemu, "-nic", "none"])])
        completed = run_audit(passing, ROOT / "bench" / "results")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((ROOT / "bench" / "results" / "qemu_executable_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_qemu_executable_audit_status"):
            return rc
        if rc := require(report["summary"]["row_commands"] == 1, "missing_qemu_executable_row_rollup"):
            return rc
        for suffix in (".md", ".csv", "_findings.csv", "_junit.xml"):
            if rc := require((ROOT / "bench" / "results" / f"qemu_executable_audit_latest{suffix}").exists(), f"missing_qemu_executable{suffix}"):
                return rc

        failing = root / "qemu_prompt_bench_fail.json"
        write_artifact(
            failing,
            ["bash", "-lc", "qemu-system-x86_64 -nic none"],
            [row(["qemu-system-aarch64", "-nic", "none"])],
            environment={"qemu_bin": "qemu-system-x86_64", "qemu_path": "/usr/bin/qemu-system-x86_64"},
        )
        failed = run_audit(failing, root / "fail")
        if rc := require(failed.returncode == 1, "qemu_executable_drift_not_rejected"):
            return rc
        fail_report = json.loads((root / "fail" / "qemu_executable_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {"wrapped_qemu_command", "non_qemu_executable", "qemu_bin_mismatch", "qemu_path_mismatch", "row_executable_drift"}
        if rc := require(expected <= kinds, "qemu_executable_findings_not_reported"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
