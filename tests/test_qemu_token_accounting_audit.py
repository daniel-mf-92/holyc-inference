#!/usr/bin/env python3
"""Tests for QEMU token accounting audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_token_accounting_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 32,
        "expected_tokens": 32,
        "expected_tokens_match": True,
        "prompt_bytes": 16,
        "elapsed_us": 30_000,
        "wall_elapsed_us": 32_000,
        "timeout_seconds": 1.0,
        "wall_timeout_pct": 3.2,
        "host_overhead_us": 2_000,
        "host_overhead_pct": 2_000 * 100.0 / 30_000,
        "tok_per_s": 32 * 1_000_000.0 / 30_000,
        "wall_tok_per_s": 32 * 1_000_000.0 / 32_000,
        "prompt_bytes_per_s": 16 * 1_000_000.0 / 30_000,
        "wall_prompt_bytes_per_s": 16 * 1_000_000.0 / 32_000,
        "us_per_token": 30_000 / 32,
        "wall_us_per_token": 32_000 / 32,
        "tokens_per_prompt_byte": 2.0,
        "memory_bytes": 1024,
        "memory_bytes_per_token": 32.0,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_token_accounting_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_consistent_token_accounting(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    args = parse_args([str(artifact), "--require-expected-tokens", "--require-expected-tokens-match"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_token_accounting_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].checks >= 10
    assert rows[0].tokens == 32


def test_audit_flags_metric_and_expected_token_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(
                expected_tokens=31,
                expected_tokens_match=True,
                wall_tok_per_s=1.0,
                wall_us_per_token=1.0,
                host_overhead_us=1.0,
                host_overhead_pct=1.0,
                wall_timeout_pct=1.0,
                prompt_bytes_per_s=1.0,
                wall_prompt_bytes_per_s=1.0,
            )
        ],
    )
    args = parse_args([str(artifact), "--require-expected-tokens", "--require-expected-tokens-match"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_token_accounting_audit.audit([artifact], args)

    assert len(rows) == 1
    kinds = {finding.kind for finding in findings}
    assert {"metric_drift", "expected_tokens_match_drift"} <= kinds
    metrics = {finding.metric for finding in findings}
    assert {"host_overhead_us", "host_overhead_pct", "wall_timeout_pct", "prompt_bytes_per_s", "wall_prompt_bytes_per_s"} <= metrics


def test_audit_rejects_recorded_expected_token_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(expected_tokens=31, expected_tokens_match=False)])
    args = parse_args([str(artifact), "--require-expected-tokens", "--require-expected-tokens-match"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_token_accounting_audit.audit([artifact], args)

    assert len(rows) == 1
    assert {finding.kind for finding in findings} == {"expected_tokens_mismatch"}


def test_audit_accepts_negative_host_overhead_when_guest_timer_exceeds_wall(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(
                elapsed_us=40_000,
                wall_elapsed_us=30_000,
                host_overhead_us=-10_000,
                host_overhead_pct=-25.0,
                wall_timeout_pct=3.0,
                tok_per_s=32 * 1_000_000.0 / 40_000,
                wall_tok_per_s=32 * 1_000_000.0 / 30_000,
                prompt_bytes_per_s=16 * 1_000_000.0 / 40_000,
                wall_prompt_bytes_per_s=16 * 1_000_000.0 / 30_000,
                us_per_token=40_000 / 32,
                wall_us_per_token=30_000 / 32,
            )
        ],
    )
    args = parse_args([str(artifact)])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_token_accounting_audit.audit([artifact], args)

    assert len(rows) == 1
    assert findings == []
    assert rows[0].host_overhead_us == -10_000


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_token_accounting_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "token_accounting",
            "--require-expected-tokens",
            "--require-expected-tokens-match",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "token_accounting.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No token accounting findings." in (output_dir / "token_accounting.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "token_accounting.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    assert rows[0]["host_overhead_us"] == "2000.0"
    finding_rows = list(csv.DictReader((output_dir / "token_accounting_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "token_accounting_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_token_accounting_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_token_accounting_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "token_accounting",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "token_accounting.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_token_accounting(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_metric_and_expected_token_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_rejects_recorded_expected_token_mismatch(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_negative_host_overhead_when_guest_timer_exceeds_wall(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
