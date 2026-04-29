#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU source air-gap auditing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bench" / "qemu_source_audit.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_dir / "qemu_source_audit_latest.json"),
        "--markdown",
        str(output_dir / "qemu_source_audit_latest.md"),
        "--csv",
        str(output_dir / "qemu_source_audit_latest.csv"),
        "--junit",
        str(output_dir / "qemu_source_audit_junit_latest.xml"),
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def write_safe_sources(path: Path) -> None:
    path.mkdir()
    (path / "runbook.md").write_text(
        "\n".join(
            [
                "# Air-gapped QEMU runbook",
                "",
                "```bash",
                "qemu-system-x86_64 -nic none -display none -drive file=/tmp/TempleOS.img,format=raw,if=ide",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (path / "safe.args").write_text("-display none -m 512M\n", encoding="utf-8")
    (path / "matrix.json").write_text(
        json.dumps(
            {
                "qemu_args": ["-m", "512M"],
                "qemu_args_file": "safe.args",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "profile.yaml").write_text(
        "\n".join(
            [
                "qemu_extra_args:",
                "  - -serial",
                "  - stdio",
                "qemu_args_files:",
                "  - safe.args",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_unsafe_sources(path: Path) -> None:
    path.mkdir()
    (path / "unsafe.args").write_text("-netdev user,id=n0\n", encoding="utf-8")
    (path / "runbook.md").write_text(
        "qemu-system-x86_64 -display none -drive file=/tmp/TempleOS.img,format=raw,if=ide\n",
        encoding="utf-8",
    )
    (path / "matrix.json").write_text(
        json.dumps(
            {
                "qemu_args": ["-netdev", "user,id=n0"],
                "qemu_args_file": "unsafe.args",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "profile.yaml").write_text(
        "\n".join(
            [
                "qemu_flags:",
                "  - -device",
                "  - e1000",
                "qemu_args_files:",
                "  - missing.args",
                "",
            ]
        ),
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-source-audit-ci-") as tmp:
        tmp_path = Path(tmp)

        safe_sources = tmp_path / "safe_sources"
        write_safe_sources(safe_sources)
        safe_output = tmp_path / "safe_output"
        completed = run_audit(safe_sources, safe_output)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        safe_report = json.loads((safe_output / "qemu_source_audit_latest.json").read_text(encoding="utf-8"))
        safe_junit = ET.parse(safe_output / "qemu_source_audit_junit_latest.xml").getroot()
        checks = [
            require(safe_report["status"] == "pass", "safe_source_audit_not_pass=true"),
            require(safe_report["commands_checked"] >= 4, "safe_source_audit_missing_coverage=true"),
            require(safe_report["findings"] == [], "safe_source_audit_has_findings=true"),
            require(safe_junit.attrib.get("failures") == "0", "safe_source_audit_junit_failures=true"),
            require(
                "QEMU Source Air-Gap Audit"
                in (safe_output / "qemu_source_audit_latest.md").read_text(encoding="utf-8"),
                "safe_source_audit_missing_markdown=true",
            ),
            require(
                "source,line,reason,command,text"
                in (safe_output / "qemu_source_audit_latest.csv").read_text(encoding="utf-8"),
                "safe_source_audit_missing_csv=true",
            ),
        ]
        if not all(checks):
            return 1

        unsafe_sources = tmp_path / "unsafe_sources"
        write_unsafe_sources(unsafe_sources)
        unsafe_output = tmp_path / "unsafe_output"
        completed = run_audit(unsafe_sources, unsafe_output)
        if completed.returncode == 0:
            print("unsafe_source_audit_not_rejected=true", file=sys.stderr)
            return 1

        unsafe_report = json.loads(
            (unsafe_output / "qemu_source_audit_latest.json").read_text(encoding="utf-8")
        )
        reasons = "\n".join(str(finding["reason"]) for finding in unsafe_report["findings"])
        unsafe_junit = ET.parse(unsafe_output / "qemu_source_audit_junit_latest.xml").getroot()
        checks = [
            require(unsafe_report["status"] == "fail", "unsafe_source_audit_not_fail=true"),
            require("missing explicit `-nic none`" in reasons, "unsafe_source_audit_missing_raw_qemu=true"),
            require("network backend `-netdev`" in reasons, "unsafe_source_audit_missing_json_fragment=true"),
            require("network device `e1000`" in reasons, "unsafe_source_audit_missing_yaml_fragment=true"),
            require(
                "referenced qemu args file not found: missing.args" in reasons,
                "unsafe_source_audit_missing_arg_file_ref=true",
            ),
            require(
                int(unsafe_junit.attrib.get("failures", "0")) >= 4,
                "unsafe_source_audit_junit_missing_failures=true",
            ),
        ]
        if not all(checks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
