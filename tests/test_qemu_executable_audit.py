#!/usr/bin/env python3
"""Tests for QEMU executable provenance audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_executable_audit


def row(command: list[str], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": "smoke-short",
        "phase": "measured",
        "command": command,
        "exit_class": "ok",
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, command: list[str], rows: list[dict[str, object]], **overrides: object) -> None:
    payload: dict[str, object] = {
        "environment": {
            "qemu_bin": Path(command[0]).name,
            "qemu_path": command[0],
        },
        "command": command,
        "warmups": [],
        "benchmarks": rows,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_executable_audit.build_parser().parse_args(extra)


def test_audit_accepts_direct_qemu_system_executable(tmp_path: Path) -> None:
    qemu = "/opt/homebrew/bin/qemu-system-x86_64"
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [qemu, "-nic", "none"], [row([qemu, "-nic", "none"])])
    args = parse_args([str(artifact), "--require-absolute"])
    args.pattern = ["qemu_prompt_bench*.json"]

    records, findings = qemu_executable_audit.audit([artifact], args)

    assert findings == []
    assert len(records) == 2
    assert records[0].argv0_basename == "qemu-system-x86_64"
    assert records[1].phase == "measured"


def test_audit_flags_wrappers_path_mismatch_and_row_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        ["env", "qemu-system-x86_64", "-nic", "none"],
        [row(["qemu-system-aarch64", "-nic", "none"])],
        environment={"qemu_bin": "qemu-system-x86_64", "qemu_path": "/usr/bin/qemu-system-x86_64"},
    )
    args = parse_args([str(artifact)])
    args.pattern = ["qemu_prompt_bench*.json"]

    records, findings = qemu_executable_audit.audit([artifact], args)

    assert len(records) == 2
    kinds = {finding.kind for finding in findings}
    assert {
        "wrapped_qemu_command",
        "non_qemu_executable",
        "qemu_bin_mismatch",
        "qemu_path_mismatch",
        "row_executable_drift",
    } <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    qemu = "qemu-system-x86_64"
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_artifact(artifact, [qemu, "-nic", "none"], [row([qemu, "-nic", "none"])])

    status = qemu_executable_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "qemu_exec"]
    )

    assert status == 0
    payload = json.loads((output_dir / "qemu_exec.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["row_commands"] == 1
    assert "No QEMU executable provenance findings." in (output_dir / "qemu_exec.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "qemu_exec.csv").open(encoding="utf-8")))
    assert rows[0]["argv0_basename"] == "qemu-system-x86_64"
    finding_rows = list(csv.DictReader((output_dir / "qemu_exec_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "qemu_exec_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_executable_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_without_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    status = qemu_executable_audit.main(
        [str(tmp_path), "--output-dir", str(output_dir), "--output-stem", "qemu_exec", "--min-artifacts", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "qemu_exec.json").read_text(encoding="utf-8"))
    assert any(finding["kind"] == "min_artifacts" for finding in payload["findings"])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-executable-test-") as tmp:
        test_audit_accepts_direct_qemu_system_executable(Path(tmp))
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-executable-test-") as tmp:
        test_audit_flags_wrappers_path_mismatch_and_row_drift(Path(tmp))
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-executable-test-") as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-executable-test-") as tmp:
        test_cli_fails_without_rows(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
