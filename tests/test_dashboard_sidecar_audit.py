#!/usr/bin/env python3
"""Tests for dashboard sidecar audit tooling."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dashboard_sidecar_audit


def write_dashboard(path: Path) -> None:
    path.write_text(json.dumps({"status": "pass", "summary": {"findings": 0}}), encoding="utf-8")
    path.with_suffix(".csv").write_text("name,status\nx,pass\n", encoding="utf-8")
    path.with_suffix(".md").write_text("# x\n", encoding="utf-8")


def test_audit_accepts_junit_latest_sidecar_name(tmp_path: Path) -> None:
    dashboard = tmp_path / "bench_trend_export_latest.json"
    write_dashboard(dashboard)
    (tmp_path / "bench_trend_export_junit_latest.xml").write_text("<testsuite/>", encoding="utf-8")

    record = dashboard_sidecar_audit.audit_dashboard(dashboard)

    assert record.status == "pass"
    assert record.junit_present is True
    assert record.junit_path.endswith("bench_trend_export_junit_latest.xml")


def test_audit_accepts_metric_csv_sidecar_names(tmp_path: Path) -> None:
    dashboard = tmp_path / "perf_regression_latest.json"
    dashboard.write_text(json.dumps({"status": "pass", "summary": {"findings": 0}}), encoding="utf-8")
    dashboard.with_suffix(".md").write_text("# perf\n", encoding="utf-8")
    dashboard.with_name("perf_regression_regressions_latest.csv").write_text("metric,value\nx,1\n", encoding="utf-8")
    dashboard.with_name("perf_regression_junit_latest.xml").write_text("<testsuite/>", encoding="utf-8")

    record = dashboard_sidecar_audit.audit_dashboard(dashboard)

    assert record.status == "pass"
    assert record.csv_present is True
    assert record.csv_path.endswith("perf_regression_regressions_latest.csv")


def test_evaluate_flags_missing_sidecars_and_invalid_json(tmp_path: Path) -> None:
    missing = tmp_path / "perf_slo_audit_latest.json"
    missing.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    invalid = tmp_path / "bad_latest.json"
    invalid.write_text("[1, 2, 3]", encoding="utf-8")

    records = [dashboard_sidecar_audit.audit_dashboard(missing), dashboard_sidecar_audit.audit_dashboard(invalid)]
    findings = dashboard_sidecar_audit.evaluate(records, min_dashboards=3)

    assert {finding.kind for finding in findings} == {"missing_sidecars", "invalid_json", "min_dashboards"}


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard_digest_latest.json"
    write_dashboard(dashboard)
    dashboard.with_name("dashboard_digest_latest_junit.xml").write_text("<testsuite/>", encoding="utf-8")
    output_dir = tmp_path / "out"

    status = dashboard_sidecar_audit.main([str(tmp_path), "--output-dir", str(output_dir)])

    assert status == 0
    payload = json.loads((output_dir / "dashboard_sidecar_audit_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["dashboards"] == 1
    assert "dashboard_digest_latest.json" in (output_dir / "dashboard_sidecar_audit_latest.md").read_text(
        encoding="utf-8"
    )
    assert "csv_present" in (output_dir / "dashboard_sidecar_audit_latest.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "dashboard_sidecar_audit_latest_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_dashboard_sidecar_audit"
    assert junit_root.attrib["failures"] == "0"
