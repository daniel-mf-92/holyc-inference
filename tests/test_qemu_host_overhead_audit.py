#!/usr/bin/env python3
"""Tests for QEMU host overhead audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_host_overhead_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "iteration": 1,
        "exit_class": "ok",
        "elapsed_us": 100_000,
        "wall_elapsed_us": 125_000,
        "host_overhead_us": 25_000,
        "host_overhead_pct": 25.0,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_host_overhead_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_consistent_host_overhead_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    args = parse_args([str(artifact), "--max-ok-host-overhead-pct", "40", "--fail-negative-host-overhead"])

    rows, findings = qemu_host_overhead_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].expected_host_overhead_us == 25_000
    assert rows[0].expected_host_overhead_pct == 25.0


def test_audit_flags_formula_drift_and_high_overhead(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(prompt="pct-drift", host_overhead_pct=1.0),
            artifact_row(prompt="us-drift", host_overhead_us=1.0),
            artifact_row(prompt="inverted", elapsed_us=200_000, wall_elapsed_us=100_000),
            artifact_row(prompt="too-high", wall_elapsed_us=145_000, host_overhead_us=45_000, host_overhead_pct=45.0),
        ],
    )
    args = parse_args([str(artifact), "--max-ok-host-overhead-pct", "40", "--fail-negative-host-overhead"])

    rows, findings = qemu_host_overhead_audit.audit([artifact], args)

    assert len(rows) == 4
    kinds = {finding.kind for finding in findings}
    assert {"host_overhead_pct_drift", "host_overhead_us_drift", "wall_elapsed_before_guest_elapsed", "ok_host_overhead_too_high"} <= kinds


def test_audit_allows_negative_host_overhead_by_default(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [artifact_row(elapsed_us=200_000, wall_elapsed_us=100_000, host_overhead_us=-100_000, host_overhead_pct=-50.0)],
    )
    args = parse_args([str(artifact)])

    rows, findings = qemu_host_overhead_audit.audit([artifact], args)

    assert findings == []
    assert rows[0].expected_host_overhead_us == -100_000


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_host_overhead_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "host_overhead", "--max-ok-host-overhead-pct", "40"]
    )

    assert status == 0
    payload = json.loads((output_dir / "host_overhead.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["max_host_overhead_pct"] == 25.0
    assert payload["summary"]["median_host_overhead_pct"] == 25.0
    assert payload["summary"]["p95_host_overhead_pct"] == 25.0
    assert "No host overhead findings." in (output_dir / "host_overhead.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "host_overhead.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "host_overhead_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "host_overhead_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_host_overhead_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_host_overhead_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "host_overhead"])

    assert status == 1
    payload = json.loads((output_dir / "host_overhead.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def test_percentile_interpolates_sorted_host_overheads() -> None:
    assert qemu_host_overhead_audit.percentile([], 95) is None
    assert qemu_host_overhead_audit.percentile([25.0], 95) == 25.0
    assert qemu_host_overhead_audit.percentile([10.0, 20.0, 30.0, 40.0], 95) == 38.5


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_host_overhead_rows(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_formula_drift_and_high_overhead(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_allows_negative_host_overhead_by_default(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    test_percentile_interpolates_sorted_host_overheads()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
