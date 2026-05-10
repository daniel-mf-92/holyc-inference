#!/usr/bin/env python3
"""Tests for QEMU serial endpoint audit."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_serial_endpoint_audit


def command() -> list[str]:
    return ["qemu-system-x86_64", "-display", "none", "-nic", "none", "-serial", "stdio", "-m", "512M"]


def row(cmd: list[str], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": "smoke-short",
        "phase": "measured",
        "launch_index": 1,
        "command": cmd,
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]], top_command: list[str] | None = None) -> None:
    path.write_text(json.dumps({"command": top_command or command(), "warmups": [], "benchmarks": rows}) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_serial_endpoint_audit.build_parser().parse_args(extra)


def test_audit_accepts_stdio_serial_and_nographic(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    nographic = ["qemu-system-x86_64", "-display", "none", "-nic", "none", "-nographic"]
    write_artifact(artifact, [row(command()), row(nographic, launch_index=2)])
    args = parse_args([str(artifact), "--require-top-command"])

    artifact_record, commands, findings = qemu_serial_endpoint_audit.audit_artifact(
        artifact,
        require_top_command=args.require_top_command,
        require_serial_stdio=args.require_serial_stdio,
    )

    assert findings == []
    assert artifact_record.status == "pass"
    assert len(commands) == 3
    assert all(record.serial_stdio for record in commands)


def test_audit_rejects_missing_serial_and_socket_endpoints(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row(["qemu-system-x86_64", "-nic", "none"], launch_index=1),
            row(["qemu-system-x86_64", "-nic", "none", "-serial", "tcp:127.0.0.1:4444"], launch_index=2),
            row(["qemu-system-x86_64", "-nic", "none", "-serial", "stdio", "-qmp", "unix:/tmp/qmp.sock"], launch_index=3),
        ],
    )

    artifact_record, commands, findings = qemu_serial_endpoint_audit.audit_artifact(
        artifact,
        require_top_command=True,
        require_serial_stdio=True,
    )

    assert artifact_record.status == "fail"
    assert len(commands) == 4
    kinds = {finding.kind for finding in findings}
    assert {"missing_serial_stdio", "socket_endpoint"} <= kinds


def test_audit_can_disable_serial_stdio_requirement(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(["qemu-system-x86_64", "-nic", "none"])], top_command=["qemu-system-x86_64", "-nic", "none"])

    artifact_record, commands, findings = qemu_serial_endpoint_audit.audit_artifact(
        artifact,
        require_top_command=True,
        require_serial_stdio=False,
    )

    assert findings == []
    assert artifact_record.status == "pass"
    assert len(commands) == 2


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(command())])
    output_dir = tmp_path / "out"

    status = qemu_serial_endpoint_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "serial_endpoint", "--require-top-command"]
    )

    assert status == 0
    payload = json.loads((output_dir / "serial_endpoint.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["command_rows"] == 2
    assert "No QEMU serial endpoint findings." in (output_dir / "serial_endpoint.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "serial_endpoint.csv").open(encoding="utf-8")))
    assert rows[0]["serial_stdio"] == "True"
    finding_rows = list(csv.DictReader((output_dir / "serial_endpoint_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "serial_endpoint_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_serial_endpoint_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_without_command_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    artifact.write_text(json.dumps({"warmups": [], "benchmarks": []}) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    status = qemu_serial_endpoint_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "serial_endpoint"])

    assert status == 1
    payload = json.loads((output_dir / "serial_endpoint.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_command_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_stdio_serial_and_nographic(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_rejects_missing_serial_and_socket_endpoints(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_can_disable_serial_stdio_requirement(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_without_command_rows(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
