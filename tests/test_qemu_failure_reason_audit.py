#!/usr/bin/env python3
"""Tests for QEMU failure-reason consistency audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_failure_reason_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "returncode": 0,
        "timed_out": False,
        "failure_reason": None,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_failure_reason_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_consistent_ok_timeout_and_nonzero_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(),
            artifact_row(exit_class="timeout", returncode=124, timed_out=True, failure_reason="timeout after 1.0s"),
            artifact_row(exit_class="nonzero_exit", returncode=2, timed_out=False, failure_reason="qemu exited 2"),
            artifact_row(exit_class="launch_error", returncode=None, timed_out=False, failure_reason="missing qemu binary"),
        ],
    )
    args = parse_args([str(artifact)])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_failure_reason_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 4
    assert rows[0].checks == 5
    assert rows[1].exit_class == "timeout"


def test_audit_flags_inconsistent_failure_diagnosis(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(exit_class="ok", returncode=7, failure_reason="stale error"),
            artifact_row(exit_class="timeout", returncode=0, timed_out=False, failure_reason=""),
            artifact_row(exit_class="nonzero_exit", returncode=0, timed_out=True, failure_reason=""),
            artifact_row(exit_class="mystery", timed_out="maybe"),
        ],
    )
    args = parse_args([str(artifact)])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_failure_reason_audit.audit([artifact], args)

    assert len(rows) == 4
    kinds = {finding.kind for finding in findings}
    assert {
        "ok_nonzero_returncode",
        "ok_with_failure_reason",
        "timeout_without_timed_out",
        "timeout_zero_returncode",
        "missing_failure_reason",
        "nonzero_exit_timed_out",
        "missing_nonzero_returncode",
        "invalid_exit_class",
        "missing_timed_out",
    } <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_failure_reason_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "failure_reason",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "failure_reason.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No failure-reason findings." in (output_dir / "failure_reason.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "failure_reason.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    assert rows[0]["exit_class"] == "ok"
    finding_rows = list(csv.DictReader((output_dir / "failure_reason_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "failure_reason_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_failure_reason_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_failure_reason_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "failure_reason",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "failure_reason.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_ok_timeout_and_nonzero_rows(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_inconsistent_failure_diagnosis(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
