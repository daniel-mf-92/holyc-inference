#!/usr/bin/env python3
"""Tests for QEMU display policy audit."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_display_policy_audit


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
    return qemu_display_policy_audit.build_parser().parse_args(extra)


def test_audit_accepts_display_none_and_nographic(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    nographic = ["qemu-system-x86_64", "-nic", "none", "-nographic"]
    write_artifact(artifact, [row(command()), row(nographic, launch_index=2)])
    args = parse_args([str(artifact), "--require-top-command"])

    artifact_record, commands, findings = qemu_display_policy_audit.audit_artifact(
        artifact,
        require_top_command=args.require_top_command,
        require_headless_display=args.require_headless_display,
    )

    assert findings == []
    assert artifact_record.status == "pass"
    assert len(commands) == 3
    assert all(record.headless_display for record in commands)


def test_audit_rejects_missing_headless_and_display_backends(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row(["qemu-system-x86_64", "-nic", "none"], launch_index=1),
            row(["qemu-system-x86_64", "-display", "gtk", "-nic", "none"], launch_index=2),
            row(["qemu-system-x86_64", "-vnc", ":1", "-nic", "none"], launch_index=3),
            row(["qemu-system-x86_64", "-spice", "port=5900,addr=127.0.0.1", "-nic", "none"], launch_index=4),
        ],
    )

    artifact_record, commands, findings = qemu_display_policy_audit.audit_artifact(
        artifact,
        require_top_command=True,
        require_headless_display=True,
    )

    assert artifact_record.status == "fail"
    assert len(commands) == 5
    kinds = {finding.kind for finding in findings}
    assert {"missing_headless_display", "forbidden_display_backend"} <= kinds


def test_audit_can_disable_headless_requirement(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(["qemu-system-x86_64", "-nic", "none"])], top_command=["qemu-system-x86_64", "-nic", "none"])

    artifact_record, commands, findings = qemu_display_policy_audit.audit_artifact(
        artifact,
        require_top_command=True,
        require_headless_display=False,
    )

    assert findings == []
    assert artifact_record.status == "pass"
    assert len(commands) == 2


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(command())])
    output_dir = tmp_path / "out"

    status = qemu_display_policy_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "display_policy", "--require-top-command"]
    )

    assert status == 0
    payload = json.loads((output_dir / "display_policy.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["command_rows"] == 2
    assert "No QEMU display policy findings." in (output_dir / "display_policy.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "display_policy.csv").open(encoding="utf-8")))
    assert rows[0]["headless_display"] == "True"
    finding_rows = list(csv.DictReader((output_dir / "display_policy_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "display_policy_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_display_policy_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_without_command_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    artifact.write_text(json.dumps({"warmups": [], "benchmarks": []}) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    status = qemu_display_policy_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "display_policy"])

    assert status == 1
    payload = json.loads((output_dir / "display_policy.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_command_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_display_none_and_nographic(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_rejects_missing_headless_and_display_backends(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_can_disable_headless_requirement(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_without_command_rows(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
