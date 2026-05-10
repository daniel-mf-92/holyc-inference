#!/usr/bin/env python3
"""Tests for QEMU token latency audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_token_latency_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 32,
        "elapsed_us": 200_000,
        "wall_elapsed_us": 240_000,
        "us_per_token": 6_250.0,
        "wall_us_per_token": 7_500.0,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_token_latency_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_consistent_token_latency_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(prompt="second", tokens=16, elapsed_us=80_000, wall_elapsed_us=96_000, us_per_token=5_000.0, wall_us_per_token=6_000.0)])
    args = parse_args([str(artifact), "--min-rows", "2"])

    rows, findings = qemu_token_latency_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 2
    assert rows[0].expected_us_per_token == 6_250.0
    assert rows[0].expected_wall_us_per_token == 7_500.0


def test_audit_flags_missing_and_drifted_token_latency_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(us_per_token=1.0),
            artifact_row(wall_us_per_token=""),
            artifact_row(tokens=0),
            artifact_row(us_per_token=6_250.0, wall_us_per_token=7_500.0),
        ],
    )
    args = parse_args([str(artifact), "--max-us-per-token", "6000", "--max-wall-us-per-token", "7000"])

    rows, findings = qemu_token_latency_audit.audit([artifact], args)

    assert len(rows) == 4
    kinds = {finding.kind for finding in findings}
    assert {"us_per_token_drift", "missing_wall_us_per_token", "missing_tokens", "max_us_per_token", "max_wall_us_per_token"} <= kinds


def test_audit_skips_warmups_and_failed_rows_by_default(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(phase="warmup", us_per_token=0.0), artifact_row(exit_class="timeout", tokens=0)])
    args = parse_args([str(artifact), "--min-rows", "1"])

    rows, findings = qemu_token_latency_audit.audit([artifact], args)

    assert len(rows) == 1
    assert findings == []


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_token_latency_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "token_latency"]
    )

    assert status == 0
    payload = json.loads((output_dir / "token_latency.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No token latency findings." in (output_dir / "token_latency.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "token_latency.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "token_latency_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "token_latency_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_token_latency_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_rows_match(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(phase="warmup")])
    output_dir = tmp_path / "out"

    status = qemu_token_latency_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "token_latency", "--min-rows", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "token_latency.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_token_latency_metrics(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_and_drifted_token_latency_metrics(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_skips_warmups_and_failed_rows_by_default(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_rows_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
