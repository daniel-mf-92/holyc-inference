#!/usr/bin/env python3
"""Stdlib-only smoke gate for QEMU args policy auditing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_args_policy_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_args_policy_audit_latest",
            "--min-files",
            "2",
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
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-args-policy-") as tmp:
        tmp_path = Path(tmp)

        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        (safe_dir / "display.args").write_text("-display none -m 512M\n", encoding="utf-8")
        (safe_dir / "cpu.json").write_text(json.dumps(["-cpu", "max"]) + "\n", encoding="utf-8")
        safe_out = tmp_path / "safe_out"
        completed = run_audit(safe_dir, safe_out)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        safe_report = json.loads((safe_out / "qemu_args_policy_audit_latest.json").read_text(encoding="utf-8"))
        safe_junit = ET.parse(safe_out / "qemu_args_policy_audit_latest_junit.xml").getroot()
        checks = [
            require(safe_report["status"] == "pass", "safe_args_policy_not_pass=true"),
            require(safe_report["summary"]["files_checked"] == 2, "safe_args_policy_missing_files=true"),
            require(safe_report["findings"] == [], "safe_args_policy_has_findings=true"),
            require(safe_junit.attrib.get("failures") == "0", "safe_args_policy_junit_failures=true"),
            require(
                "QEMU Args Policy Audit"
                in (safe_out / "qemu_args_policy_audit_latest.md").read_text(encoding="utf-8"),
                "safe_args_policy_missing_markdown=true",
            ),
            require(
                "path,exists,size_bytes,sha256,format,arg_count"
                in (safe_out / "qemu_args_policy_audit_latest.csv").read_text(encoding="utf-8"),
                "safe_args_policy_missing_csv=true",
            ),
        ]
        if not all(checks):
            return 1

        unsafe_dir = tmp_path / "unsafe"
        unsafe_dir.mkdir()
        (unsafe_dir / "netdev.args").write_text("-netdev user,id=n0\n", encoding="utf-8")
        (unsafe_dir / "long_netdev.args").write_text("--netdev user,id=n1\n", encoding="utf-8")
        (unsafe_dir / "legacy.args").write_text("-net none\n", encoding="utf-8")
        (unsafe_dir / "device.json").write_text(json.dumps(["-device", "virtio-net-pci"]) + "\n", encoding="utf-8")
        (unsafe_dir / "long_device.json").write_text(json.dumps(["--device", "e1000"]) + "\n", encoding="utf-8")
        (unsafe_dir / "command.args").write_text("qemu-system-x86_64 -nic none -m 512M\n", encoding="utf-8")
        (unsafe_dir / "response.args").write_text("@hidden-networking.args\n", encoding="utf-8")
        (unsafe_dir / "config.json").write_text(json.dumps(["-readconfig", "machine.cfg"]) + "\n", encoding="utf-8")
        (unsafe_dir / "socket.args").write_text("-serial tcp:127.0.0.1:4444,server=on\n", encoding="utf-8")
        (unsafe_dir / "long_socket.args").write_text("--serial tcp:127.0.0.1:4444,server=on\n", encoding="utf-8")
        (unsafe_dir / "vnc.args").write_text("-vnc :1\n", encoding="utf-8")
        (unsafe_dir / "display.args").write_text("-display vnc=127.0.0.1:1\n", encoding="utf-8")
        (unsafe_dir / "spice.args").write_text("-spice port=5900,addr=127.0.0.1\n", encoding="utf-8")
        (unsafe_dir / "tls.args").write_text(
            "-object tls-creds-x509,id=tls0,endpoint=server,dir=/tmp/tls\n",
            encoding="utf-8",
        )
        (unsafe_dir / "remote_disk.args").write_text(
            "-drive file=https://example.invalid/disk.img\n",
            encoding="utf-8",
        )
        (unsafe_dir / "remote_blockdev.json").write_text(
            json.dumps(["-blockdev", "driver=https,url=https://example.invalid/model.bin,node-name=n0"]) + "\n",
            encoding="utf-8",
        )
        (unsafe_dir / "duplicate_nic.args").write_text("-nic none -nic=none\n", encoding="utf-8")
        (unsafe_dir / "virtfs.args").write_text(
            "-virtfs local,path=/tmp/share,mount_tag=host0,security_model=none\n",
            encoding="utf-8",
        )
        (unsafe_dir / "fsdev.json").write_text(
            json.dumps(["-fsdev", "local,id=fs1,path=/tmp/fsdev,security_model=mapped-xattr"]) + "\n",
            encoding="utf-8",
        )
        (unsafe_dir / "fs_device.json").write_text(
            json.dumps(["-device", "virtio-9p-pci,fsdev=fs1,mount_tag=host1"]) + "\n",
            encoding="utf-8",
        )
        (unsafe_dir / "fs_marker.args").write_text(
            "-object memory-backend-file,id=mem0,mem-path=/tmp/guest,mount_tag=host2\n",
            encoding="utf-8",
        )
        (unsafe_dir / "vvfat_drive.args").write_text(
            "-drive file=fat:rw:/tmp/host-share,format=raw,if=ide\n",
            encoding="utf-8",
        )
        unsafe_out = tmp_path / "unsafe_out"
        completed = run_audit(unsafe_dir, unsafe_out)
        if completed.returncode == 0:
            print("unsafe_args_policy_not_rejected=true", file=sys.stderr)
            return 1

        unsafe_report = json.loads((unsafe_out / "qemu_args_policy_audit_latest.json").read_text(encoding="utf-8"))
        reasons = "\n".join(finding["reason"] for finding in unsafe_report["findings"])
        unsafe_junit = ET.parse(unsafe_out / "qemu_args_policy_audit_latest_junit.xml").getroot()
        checks = [
            require(unsafe_report["status"] == "fail", "unsafe_args_policy_not_fail=true"),
            require("network backend" in reasons, "unsafe_args_policy_missing_netdev=true"),
            require("networking -net" in reasons, "unsafe_args_policy_missing_legacy_net=true"),
            require("network device" in reasons, "unsafe_args_policy_missing_device=true"),
            require("qemu executable embedded" in reasons, "unsafe_args_policy_missing_command_like=true"),
            require("nested qemu args include" in reasons, "unsafe_args_policy_missing_response_include=true"),
            require("qemu config include" in reasons, "unsafe_args_policy_missing_config_include=true"),
            require("socket endpoint" in reasons, "unsafe_args_policy_missing_socket_endpoint=true"),
            require("remote display socket" in reasons, "unsafe_args_policy_missing_remote_display=true"),
            require("tls option" in reasons, "unsafe_args_policy_missing_tls_option=true"),
            require("remote resource" in reasons, "unsafe_args_policy_missing_remote_resource=true"),
            require("fragment includes -nic none" in reasons, "unsafe_args_policy_missing_fragment_nic=true"),
            require("duplicate -nic none" in reasons, "unsafe_args_policy_missing_duplicate_nic=true"),
            require("host filesystem share" in reasons, "unsafe_args_policy_missing_host_fs_share=true"),
            require("host filesystem share device" in reasons, "unsafe_args_policy_missing_host_fs_share_device=true"),
            require("host filesystem share marker" in reasons, "unsafe_args_policy_missing_host_fs_share_marker=true"),
            require(int(unsafe_junit.attrib.get("failures", "0")) >= 16, "unsafe_args_policy_junit_failures=true"),
        ]
        if not all(checks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
