#!/usr/bin/env python3
"""Smoke gate for qemu_artifact_secret_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_artifact(path: Path, row: dict[str, object]) -> None:
    path.write_text(json.dumps({"benchmarks": [row]}, indent=2) + "\n", encoding="utf-8")


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_artifact_secret_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_artifact_secret_audit_smoke",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def base_row() -> dict[str, object]:
    return {
        "prompt": "smoke-short",
        "phase": "measured",
        "command": ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio", "-display", "none"],
        "stdout_tail": "BENCH_RESULT: {\"tokens\": 8, \"elapsed_us\": 100000}\n",
        "stderr_tail": "",
        "failure_reason": "",
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C"},
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-artifact-secret-") as tmp:
        root = Path(tmp)

        safe = root / "qemu_prompt_bench_safe.json"
        write_artifact(safe, base_row())
        safe_out = root / "safe_out"
        completed = run_audit(safe, safe_out)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        safe_report = json.loads((safe_out / "qemu_artifact_secret_audit_smoke.json").read_text(encoding="utf-8"))
        safe_junit = ET.parse(safe_out / "qemu_artifact_secret_audit_smoke_junit.xml").getroot()
        checks = [
            require(safe_report["status"] == "pass", "safe_secret_audit_not_pass=true"),
            require(safe_report["summary"]["rows"] == 1, "safe_secret_audit_missing_row=true"),
            require(safe_report["findings"] == [], "safe_secret_audit_has_findings=true"),
            require(safe_junit.attrib.get("failures") == "0", "safe_secret_audit_junit_failures=true"),
            require((safe_out / "qemu_artifact_secret_audit_smoke_findings.csv").exists(), "safe_secret_audit_missing_findings_csv=true"),
        ]
        if not all(checks):
            return 1

        unsafe = root / "qemu_prompt_bench_unsafe.json"
        unsafe_row = base_row()
        unsafe_row.update(
            {
                "stdout_tail": "auth failed for sk-proj-abcdefghijklmnopqrstuvwxyz123456",
                "stderr_tail": "clone https://user:pass@example.invalid/repo.git failed",
                "api_token": "ghp_abcdefghijklmnopqrstuvwxyz123456",
            }
        )
        write_artifact(unsafe, unsafe_row)
        unsafe_out = root / "unsafe_out"
        failed = run_audit(unsafe, unsafe_out)
        if failed.returncode == 0:
            print("unsafe_secret_audit_not_rejected=true", file=sys.stderr)
            return 1
        unsafe_report = json.loads((unsafe_out / "qemu_artifact_secret_audit_smoke.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in unsafe_report["findings"]}
        unsafe_junit = ET.parse(unsafe_out / "qemu_artifact_secret_audit_smoke_junit.xml").getroot()
        checks = [
            require(unsafe_report["status"] == "fail", "unsafe_secret_audit_not_fail=true"),
            require("openai_api_key" in kinds, "unsafe_secret_audit_missing_openai=true"),
            require("github_token" in kinds, "unsafe_secret_audit_missing_github=true"),
            require("url_embedded_credentials" in kinds, "unsafe_secret_audit_missing_url_creds=true"),
            require("sensitive_field_populated" in kinds, "unsafe_secret_audit_missing_sensitive_field=true"),
            require(int(unsafe_junit.attrib.get("failures", "0")) >= 1, "unsafe_secret_audit_junit_failures=true"),
        ]
        if not all(checks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
