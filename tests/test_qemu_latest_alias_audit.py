#!/usr/bin/env python3
"""Tests for QEMU latest alias audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_latest_alias_audit


def write_artifact(path: Path, generated_at: str = "2026-05-01T10:00:00Z", prompt: str = "smoke") -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "status": "pass",
                "benchmarks": [{"phase": "measured", "prompt": prompt}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_accepts_matching_latest_alias(tmp_path: Path) -> None:
    latest = tmp_path / "qemu_prompt_bench_latest.json"
    stamped = tmp_path / "qemu_prompt_bench_20260501T100000Z.json"
    write_artifact(latest)
    write_artifact(stamped)

    record, findings = qemu_latest_alias_audit.audit_latest(latest)

    assert record.status == "pass"
    assert record.stamped_source == str(stamped)
    assert record.latest_sha256 == record.stamped_sha256
    assert findings == []


def test_audit_compares_latest_to_newest_stamped_sibling(tmp_path: Path) -> None:
    latest = tmp_path / "qemu_prompt_bench_latest.json"
    older = tmp_path / "qemu_prompt_bench_20260501T095959Z.json"
    newer = tmp_path / "qemu_prompt_bench_20260501T100000Z.json"
    write_artifact(latest, prompt="new")
    write_artifact(older, prompt="old")
    write_artifact(newer, prompt="new")

    record, findings = qemu_latest_alias_audit.audit_latest(latest)

    assert record.status == "pass"
    assert record.stamped_source == str(newer)
    assert record.stamped_candidates == 2
    assert findings == []


def test_audit_flags_payload_and_generated_at_drift(tmp_path: Path) -> None:
    latest = tmp_path / "qemu_prompt_bench_latest.json"
    stamped = tmp_path / "qemu_prompt_bench_20260501T100000Z.json"
    write_artifact(latest, generated_at="2026-05-01T09:59:59Z", prompt="old")
    write_artifact(stamped, generated_at="2026-05-01T10:00:00Z", prompt="new")

    record, findings = qemu_latest_alias_audit.audit_latest(latest)
    kinds = {finding.kind for finding in findings}

    assert record.status == "fail"
    assert "latest_alias_payload_drift" in kinds
    assert "latest_alias_generated_at_drift" in kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    latest = tmp_path / "qemu_prompt_bench_latest.json"
    stamped = tmp_path / "qemu_prompt_bench_20260501T100000Z.json"
    output_dir = tmp_path / "out"
    write_artifact(latest)
    write_artifact(stamped)

    status = qemu_latest_alias_audit.main([str(tmp_path), "--output-dir", str(output_dir), "--output-stem", "alias"])

    assert status == 0
    payload = json.loads((output_dir / "alias.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["latest_artifacts"] == 1
    assert "QEMU Latest Alias Audit" in (output_dir / "alias.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "alias.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "pass"
    finding_rows = list(csv.DictReader((output_dir / "alias_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "alias_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_latest_alias_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_without_stamped_sibling(tmp_path: Path) -> None:
    latest = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_artifact(latest)

    status = qemu_latest_alias_audit.main([str(tmp_path), "--output-dir", str(output_dir), "--output-stem", "alias"])

    assert status == 1
    payload = json.loads((output_dir / "alias.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "missing_stamped_sibling"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_matching_latest_alias(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_compares_latest_to_newest_stamped_sibling(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_payload_and_generated_at_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_without_stamped_sibling(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
