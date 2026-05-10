#!/usr/bin/env python3
"""Tests for QEMU status consistency audits."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_status_audit


def row(prompt: str = "smoke-short", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "phase": "measured",
        "returncode": 0,
        "timed_out": False,
        "exit_class": "ok",
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, *, status: str, rows: list[dict[str, object]], **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = sum(1 for item in rows if item.get("exit_class") == "ok" and not item.get("timed_out") and item.get("returncode") == 0)
    payload: dict[str, object] = {
        "generated_at": "2026-05-01T00:00:00Z",
        "status": status,
        "warmups": [],
        "benchmarks": rows,
        "command_airgap": {"ok": True, "violations": []},
        "telemetry_findings": [],
        "variability_findings": [],
        "suite_summary": {
            "runs": len(rows),
            "ok_runs": ok_rows,
            "failed_runs": len(rows) - ok_rows,
            "timed_out_runs": sum(1 for item in rows if item.get("timed_out") is True),
            "nonzero_exit_runs": sum(1 for item in rows if item.get("returncode") not in (None, 0)),
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_audit_accepts_consistent_passing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_pass.json"
    write_artifact(artifact, status="pass", rows=[row("a"), row("b")])
    args = qemu_status_audit.build_parser().parse_args([str(artifact)])

    records, findings = qemu_status_audit.audit([artifact], args)

    assert findings == []
    assert records[0].status == "pass"
    assert records[0].expected_status == "pass"
    assert records[0].ok_rows == 2


def test_audit_flags_stale_pass_status_and_suite_counters(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_fail.json"
    write_artifact(
        artifact,
        status="pass",
        rows=[row("timeout", returncode=124, timed_out=True, exit_class="timeout")],
        command_airgap={"ok": False, "violations": ["missing -nic none"]},
        telemetry_findings=[{"metric": "wall_tok_per_s"}],
        suite_summary={"runs": 1, "ok_runs": 1, "failed_runs": 0, "timed_out_runs": 0, "nonzero_exit_runs": 0},
    )
    args = qemu_status_audit.build_parser().parse_args([str(artifact)])

    records, findings = qemu_status_audit.audit([artifact], args)
    kinds = {finding.kind for finding in findings}

    assert records[0].expected_status == "fail"
    assert {
        "pass_with_failed_rows",
        "pass_with_telemetry_findings",
        "pass_without_airgap_ok",
        "status_mismatch",
        "suite_ok_runs_mismatch",
        "suite_failed_runs_mismatch",
        "suite_timed_out_runs_mismatch",
        "suite_nonzero_exit_runs_mismatch",
    } <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_pass.json"
    write_artifact(artifact, status="pass", rows=[row("a")])
    output_dir = tmp_path / "out"

    status = qemu_status_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "status"])

    assert status == 0
    payload = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "QEMU Status Audit" in (output_dir / "status.md").read_text(encoding="utf-8")
    assert "expected_status" in (output_dir / "status.csv").read_text(encoding="utf-8")
    assert "kind" in (output_dir / "status_findings.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "status_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_status_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-status-test-") as tmp:
        test_audit_accepts_consistent_passing_artifact(Path(tmp) / "pass")
        test_audit_flags_stale_pass_status_and_suite_counters(Path(tmp) / "fail")
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp) / "cli")
    print("test_qemu_status_audit=ok")
