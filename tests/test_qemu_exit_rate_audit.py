#!/usr/bin/env python3
"""Tests for QEMU exit-rate audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_exit_rate_audit


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
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_exit_rate_audit.build_parser().parse_args(extra)


def test_audit_accepts_zero_failure_rates(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row("a"), row("b")])
    args = parse_args([str(artifact), "--min-rows", "2"])

    rows, findings = qemu_exit_rate_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].failure_pct == 0.0
    assert rows[0].ok_rows == 2


def test_audit_flags_failure_timeout_and_launch_error_rates(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row("ok"),
            row("timeout", returncode=124, timed_out=True, exit_class="timeout"),
            row("nonzero", returncode=2, exit_class="nonzero_exit"),
            row("launch", returncode=None, exit_class="launch_error"),
            row("bad", exit_class="mystery"),
        ],
    )
    args = parse_args([str(artifact), "--max-failure-pct", "10", "--max-timeout-pct", "10", "--max-nonzero-exit-pct", "10", "--max-launch-error-pct", "10"])

    rows, findings = qemu_exit_rate_audit.audit([artifact], args)

    assert rows[0].failed_rows == 4
    assert {
        "invalid_exit_class",
        "max_failure_pct",
        "max_timeout_pct",
        "max_nonzero_exit_pct",
        "max_launch_error_pct",
    } <= {finding.kind for finding in findings}


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row()])
    output_dir = tmp_path / "out"

    status = qemu_exit_rate_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "exit_rate"]
    )

    assert status == 0
    payload = json.loads((output_dir / "exit_rate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No exit-rate findings." in (output_dir / "exit_rate.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "exit_rate.csv").open(encoding="utf-8")))
    assert rows[0]["failure_pct"] == "0.0"
    finding_rows = list(csv.DictReader((output_dir / "exit_rate_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "exit_rate_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_exit_rate_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_zero_failure_rates(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_failure_timeout_and_launch_error_rates(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
