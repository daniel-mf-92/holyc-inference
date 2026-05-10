#!/usr/bin/env python3
"""Smoke test for dashboard digest artifact generation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dashboard_digest


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        passing = root / "perf_slo_audit_latest.json"
        failing = root / "perf_regression_latest.json"
        output_dir = root / "dashboards"
        passing.write_text(
            json.dumps({"status": "pass", "summary": {"findings": 0}, "generated_at": "2026-04-30T00:00:00Z"}),
            encoding="utf-8",
        )
        failing.write_text(
            json.dumps({"status": "fail", "summary": {"findings": 2}, "generated_at": "2026-04-30T00:01:00Z"}),
            encoding="utf-8",
        )
        status = dashboard_digest.main(
            [
                str(passing),
                str(failing),
                "--output-dir",
                str(output_dir),
                "--fail-on-fail-status",
                "--min-dashboards",
                "2",
            ]
        )
        if status != 1:
            raise AssertionError(f"expected failing status, got {status}")
        payload = json.loads((output_dir / "dashboard_digest_latest.json").read_text(encoding="utf-8"))
        if payload["summary"]["dashboards"] != 2:
            raise AssertionError("dashboard count was not recorded")
        if payload["summary"]["total_dashboard_findings"] != 2:
            raise AssertionError("dashboard findings total was not recorded")
        for suffix in (".json", ".md", ".csv", "_junit.xml"):
            artifact = output_dir / f"dashboard_digest_latest{suffix}"
            if not artifact.exists():
                raise AssertionError(f"missing artifact {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
