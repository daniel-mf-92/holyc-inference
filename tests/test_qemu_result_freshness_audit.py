#!/usr/bin/env python3
"""Tests for QEMU benchmark result freshness audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_result_freshness_audit


def write_artifact(path: Path, generated_at: str = "2026-05-01T10:00:00Z", status: str = "pass") -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "status": status,
                "warmups": [],
                "benchmarks": [{"phase": "measured", "prompt": "smoke"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def fixed_now() -> datetime:
    return datetime(2026, 5, 1, 11, 0, 0, tzinfo=timezone.utc)


def test_audit_accepts_fresh_latest_artifact(tmp_path: Path) -> None:
    latest = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(latest)

    record = qemu_result_freshness_audit.audit_latest(latest, now=fixed_now(), max_age_seconds=7200.0)
    findings = qemu_result_freshness_audit.evaluate([record], min_latest=1)

    assert record.status == "pass"
    assert record.age_seconds == 3600.0
    assert record.within_max_age is True
    assert record.rows == 1
    assert findings == []


def test_audit_flags_stale_and_invalid_artifacts(tmp_path: Path) -> None:
    stale = tmp_path / "qemu_prompt_bench_latest.json"
    invalid = tmp_path / "nested" / "qemu_prompt_bench_latest.json"
    invalid.parent.mkdir()
    write_artifact(stale, generated_at="2026-05-01T08:00:00Z")
    write_artifact(invalid, generated_at="not-a-time")

    stale_record = qemu_result_freshness_audit.audit_latest(stale, now=fixed_now(), max_age_seconds=7200.0)
    invalid_record = qemu_result_freshness_audit.audit_latest(invalid, now=fixed_now(), max_age_seconds=7200.0)
    findings = qemu_result_freshness_audit.evaluate([stale_record, invalid_record], min_latest=3)
    kinds = {finding.kind for finding in findings}

    assert stale_record.status == "fail"
    assert invalid_record.status == "fail"
    assert {"min_latest", "stale_artifact", "invalid_generated_at"} <= kinds


def test_audit_flags_future_generated_at(tmp_path: Path) -> None:
    future = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(future, generated_at="2026-05-01T11:05:00Z")

    record = qemu_result_freshness_audit.audit_latest(future, now=fixed_now(), max_age_seconds=7200.0)
    findings = qemu_result_freshness_audit.evaluate([record], min_latest=1)

    assert record.age_seconds == -300.0
    assert "future_generated_at" in {finding.kind for finding in findings}


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    latest = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_artifact(latest)

    status = qemu_result_freshness_audit.main(
        [
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "freshness",
            "--now",
            "2026-05-01T11:00:00Z",
            "--max-age-hours",
            "2",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "freshness.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["latest_artifacts"] == 1
    assert "QEMU Result Freshness Audit" in (output_dir / "freshness.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "freshness.csv").open(encoding="utf-8")))
    assert rows[0]["within_max_age"] == "True"
    finding_rows = list(csv.DictReader((output_dir / "freshness_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "freshness_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_result_freshness_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    output_dir = tmp_path / "out"
    empty.mkdir()

    status = qemu_result_freshness_audit.main(
        [str(empty), "--output-dir", str(output_dir), "--output-stem", "freshness", "--min-latest", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "freshness.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_latest"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_fresh_latest_artifact(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_stale_and_invalid_artifacts(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_future_generated_at(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_artifacts_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
