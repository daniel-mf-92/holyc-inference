#!/usr/bin/env python3
"""Smoke test for dashboard sidecar audit artifact generation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dashboard_sidecar_audit


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dashboards = root / "dashboards"
        dashboards.mkdir()
        good = dashboards / "perf_regression_latest.json"
        good.write_text(json.dumps({"status": "pass", "summary": {"findings": 0}}), encoding="utf-8")
        good.with_suffix(".csv").write_text("metric,value\nstatus,pass\n", encoding="utf-8")
        good.with_suffix(".md").write_text("# pass\n", encoding="utf-8")
        (dashboards / "perf_regression_junit_latest.xml").write_text("<testsuite/>", encoding="utf-8")

        missing = dashboards / "perf_slo_audit_latest.json"
        missing.write_text(json.dumps({"status": "pass", "summary": {"findings": 0}}), encoding="utf-8")

        output_dir = root / "out"
        status = dashboard_sidecar_audit.main(
            [str(dashboards), "--output-dir", str(output_dir), "--min-dashboards", "2"]
        )
        if status != 1:
            raise AssertionError(f"expected failing status, got {status}")
        payload = json.loads((output_dir / "dashboard_sidecar_audit_latest.json").read_text(encoding="utf-8"))
        if payload["summary"]["dashboards"] != 2:
            raise AssertionError("dashboard count was not recorded")
        if payload["summary"]["findings"] != 1:
            raise AssertionError("missing sidecar finding was not recorded")
        for suffix in (".json", ".md", ".csv", "_junit.xml"):
            artifact = output_dir / f"dashboard_sidecar_audit_latest{suffix}"
            if not artifact.exists():
                raise AssertionError(f"missing artifact {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
