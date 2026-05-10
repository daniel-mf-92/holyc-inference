#!/usr/bin/env python3
"""Smoke gate for qemu_artifact_path_audit.py."""

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


def run_audit(input_path: Path, output_dir: Path, fail_on_host_paths: bool = False) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bench" / "qemu_artifact_path_audit.py"),
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--output-stem",
        "qemu_artifact_path_audit_smoke",
    ]
    if fail_on_host_paths:
        command.append("--fail-on-host-paths")
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def base_row() -> dict[str, object]:
    return {
        "prompt": "smoke-short",
        "phase": "measured",
        "command": ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio", "-drive", "file=/tmp/TempleOS.synthetic.img,format=raw,if=ide"],
        "stdout_tail": "BENCH_RESULT: {\"tokens\": 8, \"elapsed_us\": 100000}\n",
        "stderr_tail": "",
        "failure_reason": "",
        "environment": {"qemu_path": "/opt/homebrew/bin/qemu-system-x86_64", "TMPDIR": "/private/tmp/holyc-bench"},
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-artifact-path-") as tmp:
        root = Path(tmp)

        safe = root / "qemu_prompt_bench_safe.json"
        write_artifact(safe, base_row())
        safe_out = root / "safe_out"
        completed = run_audit(safe, safe_out, fail_on_host_paths=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        safe_report = json.loads((safe_out / "qemu_artifact_path_audit_smoke.json").read_text(encoding="utf-8"))
        safe_junit = ET.parse(safe_out / "qemu_artifact_path_audit_smoke_junit.xml").getroot()
        checks = [
            require(safe_report["status"] == "pass", "safe_path_audit_not_pass=true"),
            require(safe_report["summary"]["absolute_paths_seen"] >= 2, "safe_path_audit_missing_allowed_paths=true"),
            require(safe_report["findings"] == [], "safe_path_audit_has_findings=true"),
            require(safe_junit.attrib.get("failures") == "0", "safe_path_audit_junit_failures=true"),
            require((safe_out / "qemu_artifact_path_audit_smoke_findings.csv").exists(), "safe_path_audit_missing_findings_csv=true"),
        ]
        if not all(checks):
            return 1

        warning = root / "qemu_prompt_bench_warning.json"
        warning_row = base_row()
        warning_row["command"] = ["/Users/alice/work/holyc/bench/fixtures/qemu_synthetic_bench.py", "-nic", "none"]
        write_artifact(warning, warning_row)
        warning_out = root / "warning_out"
        warned = run_audit(warning, warning_out)
        if warned.returncode != 0:
            sys.stdout.write(warned.stdout)
            sys.stderr.write(warned.stderr)
            return warned.returncode
        warning_report = json.loads((warning_out / "qemu_artifact_path_audit_smoke.json").read_text(encoding="utf-8"))
        warning_findings = warning_report["findings"]
        checks = [
            require(warning_report["status"] == "pass", "warning_path_audit_not_pass=true"),
            require(warning_findings and warning_findings[0]["severity"] == "warning", "warning_path_audit_missing_warning=true"),
            require(warning_findings[0]["kind"] == "host_path_leak", "warning_path_audit_wrong_kind=true"),
        ]
        if not all(checks):
            return 1

        unsafe = root / "qemu_prompt_bench_unsafe.json"
        unsafe_row = base_row()
        unsafe_row.update(
            {
                "command": ["/Users/alice/work/holyc/bench/fixtures/qemu_synthetic_bench.py", "-nic", "none"],
                "stderr_tail": "failed opening /Volumes/build-cache/TempleOS.img",
                "environment": {"MODEL_PATH": "/home/ci/models/tiny.bin"},
            }
        )
        write_artifact(unsafe, unsafe_row)
        unsafe_out = root / "unsafe_out"
        failed = run_audit(unsafe, unsafe_out, fail_on_host_paths=True)
        if failed.returncode == 0:
            print("unsafe_path_audit_not_rejected=true", file=sys.stderr)
            return 1
        unsafe_report = json.loads((unsafe_out / "qemu_artifact_path_audit_smoke.json").read_text(encoding="utf-8"))
        unsafe_junit = ET.parse(unsafe_out / "qemu_artifact_path_audit_smoke_junit.xml").getroot()
        unsafe_paths = {finding["path"] for finding in unsafe_report["findings"]}
        checks = [
            require(unsafe_report["status"] == "fail", "unsafe_path_audit_not_fail=true"),
            require("/Users/alice/work/holyc/bench/fixtures/qemu_synthetic_bench.py" in unsafe_paths, "unsafe_path_audit_missing_user_path=true"),
            require("/Volumes/build-cache/TempleOS.img" in unsafe_paths, "unsafe_path_audit_missing_volume_path=true"),
            require("/home/ci/models/tiny.bin" in unsafe_paths, "unsafe_path_audit_missing_home_path=true"),
            require(int(unsafe_junit.attrib.get("failures", "0")) >= 1, "unsafe_path_audit_junit_failures=true"),
        ]
        if not all(checks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
