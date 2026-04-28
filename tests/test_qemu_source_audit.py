#!/usr/bin/env python3
"""Tests for host-side QEMU source air-gap auditing."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_source_audit


def test_audit_checks_multiline_qemu_launches_and_json_arg_arrays(tmp_path: Path) -> None:
    source = tmp_path / "runbook.md"
    source.write_text(
        "\n".join(
            [
                "```bash",
                "qemu-system-x86_64 \\",
                "  -nic none \\",
                "  -m 512M \\",
                "  -drive file=/tmp/TempleOS.img,format=raw,if=ide",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    args_json = tmp_path / "qemu_args.json"
    args_json.write_text(json.dumps(["-m", "512M", "-smp", "2"]) + "\n", encoding="utf-8")

    commands_checked, findings = qemu_source_audit.audit([source, args_json], set())

    assert commands_checked == 2
    assert findings == []


def test_audit_rejects_standalone_json_args_file_with_network_backend(tmp_path: Path) -> None:
    args_json = tmp_path / "qemu_args.json"
    args_json.write_text(json.dumps(["-netdev", "user,id=n0", "-device", "e1000,netdev=n0"]) + "\n", encoding="utf-8")

    commands_checked, findings = qemu_source_audit.audit([args_json], set())

    assert commands_checked == 1
    assert len(findings) == 2
    assert findings[0].reason == "$: network backend `-netdev`"
    assert findings[1].reason == "$: network device `e1000,netdev=n0`"


def test_audit_rejects_json_nested_qemu_args_fragment(tmp_path: Path) -> None:
    matrix = tmp_path / "bench_matrix.json"
    matrix.write_text(
        json.dumps({"profiles": [{"name": "unsafe", "qemu_args": ["-nic", "user"]}]}) + "\n",
        encoding="utf-8",
    )

    commands_checked, findings = qemu_source_audit.audit([matrix], set())

    assert commands_checked == 1
    assert len(findings) == 1
    assert findings[0].reason == "$.profiles[0].qemu_args: non-air-gapped `-nic user`"


def test_cli_writes_reports_for_standalone_json_args_violation(tmp_path: Path) -> None:
    args_json = tmp_path / "qemu_args.json"
    args_json.write_text(json.dumps(["-device", "virtio-net-pci"]) + "\n", encoding="utf-8")
    output = tmp_path / "qemu_source_audit.json"
    markdown = tmp_path / "qemu_source_audit.md"
    csv_report = tmp_path / "qemu_source_audit.csv"
    junit = tmp_path / "qemu_source_audit.xml"

    status = qemu_source_audit.main(
        [
            "--input",
            str(args_json),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_report),
            "--junit",
            str(junit),
        ]
    )

    assert status == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["commands_checked"] == 1
    assert "$: network device `virtio-net-pci`" in markdown.read_text(encoding="utf-8")
    assert "$: network device `virtio-net-pci`" in csv_report.read_text(encoding="utf-8")
    junit_root = ET.parse(junit).getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_source_airgap_audit"
    assert junit_root.attrib["tests"] == "1"
    assert junit_root.attrib["failures"] == "1"
