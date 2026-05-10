#!/usr/bin/env python3
"""Tests for host-side dashboard digest tooling."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dashboard_digest


def test_load_dashboard_normalizes_status_and_summary_findings(tmp_path: Path) -> None:
    artifact = tmp_path / "perf_slo_audit_latest.json"
    artifact.write_text(
        json.dumps({"status": "ok", "summary": {"findings": "3"}, "generated_at": "2026-04-30T00:00:00Z"}),
        encoding="utf-8",
    )

    record = dashboard_digest.load_dashboard(artifact)

    assert record.name == "perf_slo_audit"
    assert record.status == "pass"
    assert record.findings == 3
    assert record.summary_keys == "findings"


def test_evaluate_flags_missing_and_failed_dashboards(tmp_path: Path) -> None:
    passing = tmp_path / "pass.json"
    failed = tmp_path / "fail.json"
    missing = tmp_path / "missing.json"
    passing.write_text(json.dumps({"status": "pass", "summary": {"findings": 0}}), encoding="utf-8")
    failed.write_text(json.dumps({"status": "fail", "summary": {"findings": 1}}), encoding="utf-8")

    records = [dashboard_digest.load_dashboard(path) for path in (passing, failed, missing)]
    findings = dashboard_digest.evaluate(
        records,
        fail_on_missing=True,
        fail_on_fail_status=True,
        min_dashboards=4,
    )

    assert {finding.gate for finding in findings} == {"min_dashboards", "dashboard_status", "missing"}


def test_cli_writes_digest_artifacts(tmp_path: Path) -> None:
    passing = tmp_path / "trend_latest.json"
    failed = tmp_path / "perf_regression_latest.json"
    output_dir = tmp_path / "dashboards"
    passing.write_text(json.dumps({"status": "pass", "summary": {"findings": 0}}), encoding="utf-8")
    failed.write_text(json.dumps({"status": "fail", "summary": {"findings": 2}}), encoding="utf-8")

    status = dashboard_digest.main(
        [
            str(passing),
            str(failed),
            "--output-dir",
            str(output_dir),
            "--fail-on-fail-status",
            "--min-dashboards",
            "2",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "dashboard_digest_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["summary"]["dashboards"] == 2
    assert payload["summary"]["total_dashboard_findings"] == 2
    assert "perf_regression status is fail" in (output_dir / "dashboard_digest_latest.md").read_text(
        encoding="utf-8"
    )
    assert "perf_regression" in (output_dir / "dashboard_digest_latest.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "dashboard_digest_latest_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_dashboard_digest"
    assert junit_root.attrib["failures"] == "1"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "load"
        path.mkdir()
        test_load_dashboard_normalizes_status_and_summary_findings(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluate"
        path.mkdir()
        test_evaluate_flags_missing_and_failed_dashboards(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cli"
        path.mkdir()
        test_cli_writes_digest_artifacts(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
