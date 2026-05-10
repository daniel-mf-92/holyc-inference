#!/usr/bin/env python3
"""Smoke gate for QEMU block-device policy auditing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def row(command: list[str]) -> dict[str, object]:
    return {"prompt": "smoke-short", "phase": "measured", "launch_index": 1, "command": command}


def write_artifact(path: Path, command: list[str]) -> None:
    path.write_text(json.dumps({"command": command, "warmups": [], "benchmarks": [row(command)]}, indent=2) + "\n", encoding="utf-8")


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_block_device_policy_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_block_device_policy_audit_smoke",
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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-block-policy-") as tmp:
        tmp_path = Path(tmp)

        safe = tmp_path / "safe.json"
        safe_command = [
            "qemu-system-x86_64",
            "-nic",
            "none",
            "-serial",
            "stdio",
            "-display",
            "none",
            "-drive",
            "file=/tmp/TempleOS.synthetic.img,format=raw,if=ide",
        ]
        write_artifact(safe, safe_command)
        safe_out = tmp_path / "safe_out"
        completed = run_audit(safe, safe_out)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        safe_report = json.loads((safe_out / "qemu_block_device_policy_audit_smoke.json").read_text(encoding="utf-8"))
        safe_junit = ET.parse(safe_out / "qemu_block_device_policy_audit_smoke_junit.xml").getroot()
        checks = [
            require(safe_report["status"] == "pass", "safe_block_policy_not_pass=true"),
            require(safe_report["summary"]["drive_options"] == 2, "safe_block_policy_missing_drive_options=true"),
            require(safe_report["findings"] == [], "safe_block_policy_has_findings=true"),
            require(safe_junit.attrib.get("failures") == "0", "safe_block_policy_junit_failures=true"),
            require(
                "QEMU Block Device Policy Audit"
                in (safe_out / "qemu_block_device_policy_audit_smoke.md").read_text(encoding="utf-8"),
                "safe_block_policy_missing_markdown=true",
            ),
        ]
        if not all(checks):
            return 1

        unsafe = tmp_path / "unsafe.json"
        unsafe_command = [
            "qemu-system-x86_64",
            "-nic",
            "none",
            "-serial",
            "stdio",
            "-display",
            "none",
            "-drive",
            "file=https://example.invalid/TempleOS.img,format=raw,if=ide",
            "-drive",
            "file=/tmp/extra.qcow2,format=qcow2,if=virtio",
            "-blockdev",
            "driver=nbd,server.type=inet,server.host=127.0.0.1",
            "-cdrom",
            "/tmp/tools.iso",
        ]
        write_artifact(unsafe, unsafe_command)
        unsafe_out = tmp_path / "unsafe_out"
        completed = run_audit(unsafe, unsafe_out)
        if completed.returncode == 0:
            print("unsafe_block_policy_not_rejected=true", file=sys.stderr)
            return 1

        unsafe_report = json.loads((unsafe_out / "qemu_block_device_policy_audit_smoke.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in unsafe_report["findings"]}
        unsafe_junit = ET.parse(unsafe_out / "qemu_block_device_policy_audit_smoke_junit.xml").getroot()
        checks = [
            require(unsafe_report["status"] == "fail", "unsafe_block_policy_not_fail=true"),
            require("remote_block_transport" in kinds, "unsafe_block_policy_missing_remote=true"),
            require("non_canonical_drive" in kinds, "unsafe_block_policy_missing_noncanonical=true"),
            require("blockdev_graph" in kinds, "unsafe_block_policy_missing_blockdev=true"),
            require("extra_disk_media" in kinds, "unsafe_block_policy_missing_legacy_disk=true"),
            require(int(unsafe_junit.attrib.get("failures", "0")) >= 1, "unsafe_block_policy_junit_failures=true"),
        ]
        if not all(checks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
