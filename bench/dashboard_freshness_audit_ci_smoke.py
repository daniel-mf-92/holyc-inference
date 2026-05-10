#!/usr/bin/env python3
"""Smoke test for dashboard freshness audit artifact generation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dashboard_freshness_audit


def write_dashboard(path: Path, generated_at: str) -> None:
    path.write_text(json.dumps({"status": "pass", "generated_at": generated_at}) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dashboard-freshness-") as tmp:
        root = Path(tmp)
        dashboards = root / "dashboards"
        dashboards.mkdir()
        write_dashboard(dashboards / "fresh_latest.json", "2026-05-02T00:00:00Z")
        write_dashboard(dashboards / "stale_latest.json", "2026-04-27T00:00:00Z")
        write_dashboard(dashboards / "future_latest.json", "2026-05-03T00:00:00Z")

        output_dir = root / "out"
        original_now = dashboard_freshness_audit.datetime

        class FixedDateTime(original_now):
            @classmethod
            def now(cls, tz=None):  # type: ignore[no-untyped-def]
                return cls(2026, 5, 2, 1, 0, 0, tzinfo=tz)

        dashboard_freshness_audit.datetime = FixedDateTime
        try:
            status = dashboard_freshness_audit.main(
                [
                    str(dashboards),
                    "--output-dir",
                    str(output_dir),
                    "--max-age-hours",
                    "48",
                    "--future-skew-minutes",
                    "5",
                    "--min-dashboards",
                    "3",
                ]
            )
        finally:
            dashboard_freshness_audit.datetime = original_now

        if status != 1:
            raise AssertionError(f"expected failing freshness status, got {status}")
        payload = json.loads((output_dir / "dashboard_freshness_audit_latest.json").read_text(encoding="utf-8"))
        if payload["summary"]["dashboards"] != 3:
            raise AssertionError("dashboard count was not recorded")
        kinds = {finding["kind"] for finding in payload["findings"]}
        if kinds != {"stale_dashboard", "future_timestamp"}:
            raise AssertionError(f"unexpected findings: {sorted(kinds)}")
        for suffix in (".json", ".md", ".csv", "_findings.csv", "_junit.xml"):
            artifact = output_dir / f"dashboard_freshness_audit_latest{suffix}"
            if not artifact.exists():
                raise AssertionError(f"missing artifact {artifact}")

    print("dashboard_freshness_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
