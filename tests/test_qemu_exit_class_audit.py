#!/usr/bin/env python3
"""Tests for QEMU exit-class consistency audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_exit_class_audit


def row(prompt: str = "smoke-short", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "returncode": 0,
        "timed_out": False,
        "exit_class": "ok",
        "failure_reason": None,
        "tokens": 32,
        "elapsed_us": 200000,
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_exit_class_audit.build_parser().parse_args(extra)


def test_audit_accepts_consistent_ok_timeout_and_nonzero_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row("ok"),
            row("timeout", returncode=-9, timed_out=True, exit_class="timeout", failure_reason="timeout", tokens=None, elapsed_us=0),
            row("nonzero", returncode=2, exit_class="nonzero_exit", failure_reason="returncode_2", tokens=None, elapsed_us=0),
        ],
    )
    args = parse_args([str(artifact), "--require-success-telemetry", "--require-failure-reason"])

    rows, findings = qemu_exit_class_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 3
    assert rows[1].computed_exit_class == "timeout"
    assert rows[2].computed_failure_reason == "returncode_2"


def test_audit_flags_exit_class_and_failure_reason_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row("bad-class", returncode=0, timed_out=False, exit_class="timeout", failure_reason="timeout"),
            row("bad-reason", returncode=127, exit_class="launch_error", failure_reason="launch_error"),
            row("missing-telemetry", tokens=0, elapsed_us=0),
            row("invalid", exit_class="weird"),
            row("missing-fields", returncode=None, timed_out=None),
        ],
    )
    args = parse_args([str(artifact), "--require-success-telemetry", "--require-failure-reason"])

    rows, findings = qemu_exit_class_audit.audit([artifact], args)
    kinds = {finding.kind for finding in findings}

    assert len(rows) == 5
    assert {
        "exit_class_mismatch",
        "failure_reason_mismatch",
        "ok_row_missing_tokens",
        "ok_row_missing_elapsed_us",
        "invalid_exit_class",
        "missing_returncode",
        "missing_timed_out",
    } <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row()])
    output_dir = tmp_path / "out"

    status = qemu_exit_class_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "exit_class", "--require-success-telemetry"]
    )

    assert status == 0
    payload = json.loads((output_dir / "exit_class.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["exit_class_ok_rows"] == 1
    assert "No exit-class findings." in (output_dir / "exit_class.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "exit_class.csv").open(encoding="utf-8")))
    assert rows[0]["computed_exit_class"] == "ok"
    finding_rows = list(csv.DictReader((output_dir / "exit_class_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "exit_class_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_exit_class_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-exit-class-test-") as tmp:
        test_audit_accepts_consistent_ok_timeout_and_nonzero_rows(Path(tmp) / "pass")
        test_audit_flags_exit_class_and_failure_reason_drift(Path(tmp) / "fail")
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp) / "cli")
    print("test_qemu_exit_class_audit=ok")
