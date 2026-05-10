#!/usr/bin/env python3
"""Tests for perf CI gate tooling."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import perf_ci_gate


def write_dashboard(path: Path, *, status: str = "pass", findings: int = 0) -> None:
    path.write_text(
        json.dumps({"status": status, "summary": {"rows": 1, "findings": findings}}),
        encoding="utf-8",
    )
    path.with_suffix(".md").write_text("# dashboard\n", encoding="utf-8")
    path.with_suffix(".csv").write_text("name,status\nsmoke,pass\n", encoding="utf-8")
    path.with_name(f"{path.with_suffix('').name}_junit.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuite name="smoke" tests="1" failures="0" />',
        encoding="utf-8",
    )


def test_load_dashboard_accepts_passing_dashboard_with_sidecars(tmp_path: Path) -> None:
    dashboard = tmp_path / "perf_regression_latest.json"
    write_dashboard(dashboard)

    record, findings = perf_ci_gate.load_dashboard(dashboard, require_sidecars=True)

    assert findings == []
    assert record.name == "perf_regression"
    assert record.status == "pass"
    assert record.sidecars_checked == 3
    assert record.sidecars_missing == 0


def test_load_dashboard_flags_status_findings_and_missing_sidecars(tmp_path: Path) -> None:
    dashboard = tmp_path / "perf_slo_audit_latest.json"
    dashboard.write_text(json.dumps({"status": "fail", "summary": {"rows": 0, "findings": 2}}), encoding="utf-8")

    record, findings = perf_ci_gate.load_dashboard(dashboard, require_sidecars=True)

    assert record.status == "fail"
    assert record.sidecars_missing == 3
    assert {finding.kind for finding in findings} == {
        "dashboard_status",
        "dashboard_findings",
        "empty_dashboard",
        "missing_sidecar",
    }


def test_cli_writes_gate_outputs(tmp_path: Path) -> None:
    dashboard = tmp_path / "perf_regression_latest.json"
    output_dir = tmp_path / "out"
    write_dashboard(dashboard)

    status = perf_ci_gate.main([str(dashboard), "--output-dir", str(output_dir), "--output-stem", "gate"])

    assert status == 0
    payload = json.loads((output_dir / "gate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["dashboards"] == 1
    assert "No perf CI gate findings." in (output_dir / "gate.md").read_text(encoding="utf-8")
    assert "perf_regression" in (output_dir / "gate.csv").read_text(encoding="utf-8")
    assert (output_dir / "gate_findings.csv").exists()
    assert "holyc_perf_ci_gate" in (output_dir / "gate_junit.xml").read_text(encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_load_dashboard_accepts_passing_dashboard_with_sidecars(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_load_dashboard_flags_status_findings_and_missing_sidecars(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_gate_outputs(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
