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


def test_audit_rejects_legacy_net_none_in_qemu_args_fragments(tmp_path: Path) -> None:
    matrix = tmp_path / "bench_matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "profiles": [{"name": "legacy", "qemu_args": ["-net", "none"]}],
                "quantizations": [{"name": "legacy-equals", "qemu_flags": ["-net=none"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    commands_checked, findings = qemu_source_audit.audit([matrix], set())

    assert commands_checked == 2
    assert len(findings) == 2
    reasons = [finding.reason for finding in findings]
    assert "$.profiles[0].qemu_args: legacy `-net none` present; benchmark fragments must use injected `-nic none`" in reasons
    assert "$.quantizations[0].qemu_flags: legacy `-net=none` present; benchmark fragments must use injected `-nic none`" in reasons


def test_audit_rejects_args_fragment_escape_hatches(tmp_path: Path) -> None:
    args_file = tmp_path / "unsafe.args"
    args_file.write_text(
        "\n".join(
            [
                "qemu-system-x86_64 -nic none -m 512M",
                "@hidden-network.args",
                "-readconfig machine.cfg",
                "-chardev socket,id=mon,path=/tmp/qmp.sock",
                "-display vnc=127.0.0.1:1",
                "-object tls-creds-x509,id=tls0,endpoint=server,dir=/tmp/tls",
                "-smb /tmp/share",
                "",
            ]
        ),
        encoding="utf-8",
    )

    commands_checked, findings = qemu_source_audit.audit([args_file], set())

    assert commands_checked == 8
    reasons = [finding.reason for finding in findings]
    assert "embedded qemu-system executable; args fragments must not contain launch commands" in reasons
    assert "nested qemu args include `@hidden-network.args`" in reasons
    assert "qemu config include `-readconfig machine.cfg`" in reasons
    assert "socket endpoint `-chardev socket,id=mon,path=/tmp/qmp.sock`" in reasons
    assert "remote display socket `-display vnc=127.0.0.1:1`" in reasons
    assert "tls option `-object tls-creds-x509,id=tls0,endpoint=server,dir=/tmp/tls`" in reasons
    assert "user-mode network service `-smb`" in reasons


def test_audit_checks_yaml_qemu_args_fragments(tmp_path: Path) -> None:
    matrix = tmp_path / "bench_matrix.yml"
    matrix.write_text(
        "\n".join(
            [
                "profiles:",
                "  - name: safe",
                "    qemu_args:",
                "      - -m",
                "      - 512M",
                "  - name: unsafe",
                "    qemu_extra_args: ['-netdev', 'user,id=n0']",
                "quantizations:",
                "  - name: q4",
                "    qemu_flags: [-device, virtio-net-pci]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    commands_checked, findings = qemu_source_audit.audit([matrix], set())

    assert commands_checked == 3
    assert len(findings) == 2
    assert findings[0].reason == "$.qemu_extra_args: network backend `-netdev`"
    assert findings[1].reason == "$.qemu_flags: network device `virtio-net-pci`"


def test_audit_checks_toml_qemu_args_fragments_and_arg_files(tmp_path: Path) -> None:
    (tmp_path / "safe.args").write_text("-display none -m 512M\n", encoding="utf-8")
    matrix = tmp_path / "bench_matrix.toml"
    matrix.write_text(
        "\n".join(
            [
                "qemu_args_files = ['safe.args']",
                "",
                "[[profiles]]",
                "name = 'safe'",
                "qemu_args = ['-m', '512M']",
                "",
                "[[profiles]]",
                "name = 'unsafe'",
                "qemu_extra_args = ['-nic', 'user']",
                "",
                "[quantizations.q4]",
                "qemu_flags = ['-device', 'rtl8139']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    commands_checked, findings = qemu_source_audit.audit([matrix], set())

    assert commands_checked == 4
    assert len(findings) == 2
    reasons = [finding.reason for finding in findings]
    assert "$.profiles[1].qemu_extra_args: non-air-gapped `-nic user`" in reasons
    assert "$.quantizations.q4.qemu_flags: network device `rtl8139`" in reasons


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
