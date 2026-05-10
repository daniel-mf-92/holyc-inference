#!/usr/bin/env python3
"""Tests for dashboard freshness audit tooling."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dashboard_freshness_audit


def write_dashboard(path: Path, generated_at: str) -> None:
    path.write_text(json.dumps({"status": "pass", "generated_at": generated_at}), encoding="utf-8")


def test_fresh_dashboard_passes(tmp_path: Path) -> None:
    dashboard = tmp_path / "perf_regression_latest.json"
    write_dashboard(dashboard, "2026-05-02T00:30:00Z")

    record, findings = dashboard_freshness_audit.audit_dashboard(
        dashboard,
        datetime(2026, 5, 2, 1, 0, 0, tzinfo=timezone.utc),
        max_age_hours=2,
        future_skew_minutes=5,
    )

    assert record.status == "pass"
    assert record.age_hours == 0.5
    assert findings == []


def test_stale_and_future_dashboards_fail(tmp_path: Path) -> None:
    stale = tmp_path / "stale_latest.json"
    future = tmp_path / "future_latest.json"
    write_dashboard(stale, "2026-05-01T00:00:00Z")
    write_dashboard(future, "2026-05-02T03:00:00Z")

    now = datetime(2026, 5, 2, 1, 0, 0, tzinfo=timezone.utc)
    _, stale_findings = dashboard_freshness_audit.audit_dashboard(stale, now, 12, 5)
    _, future_findings = dashboard_freshness_audit.audit_dashboard(future, now, 12, 5)

    assert stale_findings[0].kind == "stale_dashboard"
    assert future_findings[0].kind == "future_timestamp"


def test_cli_writes_outputs(tmp_path: Path) -> None:
    dashboard = tmp_path / "perf_ci_gate_latest.json"
    output_dir = tmp_path / "out"
    write_dashboard(dashboard, "2026-05-02T00:00:00Z")

    original_now = dashboard_freshness_audit.datetime

    class FixedDateTime(original_now):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return cls(2026, 5, 2, 1, 0, 0, tzinfo=tz)

    dashboard_freshness_audit.datetime = FixedDateTime
    try:
        status = dashboard_freshness_audit.main(
            [str(dashboard), "--output-dir", str(output_dir), "--max-age-hours", "2"]
        )
    finally:
        dashboard_freshness_audit.datetime = original_now

    assert status == 0
    payload = json.loads((output_dir / "dashboard_freshness_audit_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert "No dashboard freshness findings." in (output_dir / "dashboard_freshness_audit_latest.md").read_text(
        encoding="utf-8"
    )
    assert "perf_ci_gate_latest.json" in (output_dir / "dashboard_freshness_audit_latest.csv").read_text(
        encoding="utf-8"
    )
    assert (output_dir / "dashboard_freshness_audit_latest_findings.csv").exists()
    assert "holyc_dashboard_freshness_audit" in (
        output_dir / "dashboard_freshness_audit_latest_junit.xml"
    ).read_text(encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_fresh_dashboard_passes(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_stale_and_future_dashboards_fail(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_outputs(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
