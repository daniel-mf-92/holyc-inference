#!/usr/bin/env python3
"""Tests for QEMU TTFT audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_ttft_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "iteration": 1,
        "exit_class": "ok",
        "tokens": 32,
        "ttft_us": 12_000,
        "elapsed_us": 200_000,
        "wall_elapsed_us": 240_000,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_ttft_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_consistent_ttft_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(prompt="second", ttft_us=16_000, elapsed_us=320_000, wall_elapsed_us=400_000)])
    args = parse_args([str(artifact), "--min-rows", "2", "--max-ttft-us", "20000"])

    rows, findings = qemu_ttft_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 2
    assert rows[0].ttft_elapsed_pct == 6.0
    assert rows[0].ttft_wall_pct == 5.0


def test_audit_flags_missing_negative_late_and_high_ttft(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(prompt="missing", ttft_us=""),
            artifact_row(prompt="negative", ttft_us=-1),
            artifact_row(prompt="late-guest", ttft_us=250_000),
            artifact_row(prompt="late-wall", ttft_us=260_000, elapsed_us=300_000, wall_elapsed_us=240_000),
            artifact_row(prompt="too-high", ttft_us=80_000),
            artifact_row(prompt="zero-tokens", tokens=0),
        ],
    )
    args = parse_args([str(artifact), "--max-ttft-us", "50000", "--max-ttft-elapsed-pct", "30"])

    rows, findings = qemu_ttft_audit.audit([artifact], args)

    assert len(rows) == 6
    kinds = {finding.kind for finding in findings}
    assert {"missing_ttft_us", "negative_ttft_us", "ttft_after_guest_elapsed", "ttft_after_wall_elapsed", "max_ttft_us", "max_ttft_elapsed_pct", "missing_tokens"} <= kinds


def test_audit_skips_warmups_and_failed_rows_by_default(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(phase="warmup", ttft_us=""), artifact_row(exit_class="timeout", ttft_us="")])
    args = parse_args([str(artifact), "--min-rows", "1"])

    rows, findings = qemu_ttft_audit.audit([artifact], args)

    assert len(rows) == 1
    assert findings == []


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_ttft_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "ttft"])

    assert status == 0
    payload = json.loads((output_dir / "ttft.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["median_ttft_us"] == 12_000
    assert "No TTFT findings." in (output_dir / "ttft.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "ttft.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "ttft_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "ttft_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_ttft_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_ttft_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "ttft"])

    assert status == 1
    payload = json.loads((output_dir / "ttft.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def test_percentile_interpolates_sorted_ttft_values() -> None:
    assert qemu_ttft_audit.percentile([], 95) is None
    assert qemu_ttft_audit.percentile([12_000], 95) == 12_000
    assert qemu_ttft_audit.percentile([10.0, 20.0, 30.0, 40.0], 95) == 38.5


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_ttft_rows(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_negative_late_and_high_ttft(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_skips_warmups_and_failed_rows_by_default(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    test_percentile_interpolates_sorted_ttft_values()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
