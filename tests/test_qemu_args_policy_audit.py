from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_args_policy_audit


def test_audit_accepts_safe_shell_and_json_args(tmp_path: Path) -> None:
    shell_args = tmp_path / "safe.args"
    json_args = tmp_path / "safe.json"
    shell_args.write_text("-display none -m 512M\n", encoding="utf-8")
    json_args.write_text(json.dumps(["-cpu", "max"]) + "\n", encoding="utf-8")

    records, findings = qemu_args_policy_audit.audit([tmp_path])

    assert findings == []
    assert len(records) == 2
    assert {record.format for record in records} == {"json-array", "shell-fragment"}
    assert {record.arg_count for record in records} == {2, 4}


def test_audit_rejects_networking_fragments_and_commands(tmp_path: Path) -> None:
    (tmp_path / "netdev.args").write_text("-netdev user,id=n0\n", encoding="utf-8")
    (tmp_path / "net.args").write_text("-net none\n", encoding="utf-8")
    (tmp_path / "device.json").write_text(json.dumps(["-device", "e1000"]) + "\n", encoding="utf-8")
    (tmp_path / "modern_device.args").write_text("-device igb\n-device vhost-vsock-pci\n-device rocker\n", encoding="utf-8")
    (tmp_path / "command.args").write_text("qemu-system-x86_64 -nic none -m 512M\n", encoding="utf-8")
    (tmp_path / "response.args").write_text("@hidden-networking.args\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps(["-readconfig", "machine.cfg"]) + "\n", encoding="utf-8")
    (tmp_path / "socket.args").write_text("-chardev socket,id=mon,path=/tmp/qmp.sock\n", encoding="utf-8")
    (tmp_path / "vnc.args").write_text("-vnc :1\n", encoding="utf-8")
    (tmp_path / "display.args").write_text("-display vnc=127.0.0.1:1\n", encoding="utf-8")
    (tmp_path / "spice.args").write_text("-spice port=5900,addr=127.0.0.1\n", encoding="utf-8")
    (tmp_path / "long_netdev.args").write_text("--netdev user,id=n0\n", encoding="utf-8")
    (tmp_path / "long_device.args").write_text("--device e1000\n", encoding="utf-8")
    (tmp_path / "long_vnc.args").write_text("--vnc :2\n", encoding="utf-8")
    (tmp_path / "smb.args").write_text("-smb /tmp/share\n", encoding="utf-8")
    (tmp_path / "tftp.args").write_text("-tftp=/tmp/tftp\n", encoding="utf-8")
    (tmp_path / "tls.args").write_text(
        "-object tls-creds-x509,id=tls0,endpoint=server,dir=/tmp/tls\n",
        encoding="utf-8",
    )
    (tmp_path / "duplicate_nic.args").write_text("-nic none -nic=none\n", encoding="utf-8")

    records, findings = qemu_args_policy_audit.audit([tmp_path])
    reasons = {finding.reason for finding in findings}

    assert len(records) == 18
    assert "network backend" in reasons
    assert "networking -net" in reasons
    assert "network device" in reasons
    assert "fragment includes -nic none" in reasons
    assert "duplicate -nic none" in reasons
    assert "qemu executable embedded in args file" in reasons
    assert "nested qemu args include" in reasons
    assert "qemu config include" in reasons
    assert "socket endpoint" in reasons
    assert "remote display socket" in reasons
    assert "user-mode network service" in reasons
    assert "tls option" in reasons


def test_cli_writes_reports_and_junit(tmp_path: Path) -> None:
    args_file = tmp_path / "safe.args"
    output_dir = tmp_path / "out"
    args_file.write_text("-display none\n", encoding="utf-8")

    status = qemu_args_policy_audit.main(
        [str(args_file), "--output-dir", str(output_dir), "--output-stem", "args_policy", "--min-files", "1"]
    )

    assert status == 0
    report = json.loads((output_dir / "args_policy.json").read_text(encoding="utf-8"))
    junit = ET.parse(output_dir / "args_policy_junit.xml").getroot()
    assert report["status"] == "pass"
    assert report["summary"]["files_checked"] == 1
    assert report["summary"]["duplicate_nic_none_fragments"] == 0
    assert junit.attrib["failures"] == "0"
    assert "QEMU Args Policy Audit" in (output_dir / "args_policy.md").read_text(encoding="utf-8")
