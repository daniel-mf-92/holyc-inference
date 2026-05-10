#!/usr/bin/env python3
"""Tests for QEMU prompt budget audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_budget_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 32,
        "expected_tokens": 32,
        "expected_tokens_match": True,
        "prompt_bytes": 16,
        "guest_prompt_bytes": 16,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_prompt_budget_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_rows_within_prompt_and_token_budgets(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(prompt="second", tokens=16, expected_tokens=16, prompt_bytes=24, guest_prompt_bytes=24)])
    args = parse_args([str(artifact), "--max-prompt-bytes", "32", "--max-tokens", "40", "--require-expected-tokens", "--require-guest-prompt-bytes"])

    rows, findings = qemu_prompt_budget_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 2
    assert rows[0].prompt_bytes_over_budget is None
    assert rows[0].tokens_over_budget is None


def test_audit_flags_budget_and_expected_token_failures(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(prompt_bytes=33),
            artifact_row(tokens=41),
            artifact_row(tokens=31, expected_tokens=32, expected_tokens_match=False),
            artifact_row(guest_prompt_bytes=17),
            artifact_row(expected_tokens=""),
        ],
    )
    args = parse_args([str(artifact), "--max-prompt-bytes", "32", "--max-tokens", "40", "--require-expected-tokens", "--require-guest-prompt-bytes"])

    rows, findings = qemu_prompt_budget_audit.audit([artifact], args)

    assert len(rows) == 5
    kinds = {finding.kind for finding in findings}
    assert {
        "prompt_bytes_over_budget",
        "tokens_over_budget",
        "expected_tokens_mismatch",
        "expected_tokens_match_false",
        "guest_prompt_bytes_mismatch",
        "missing_expected_tokens",
    } <= kinds


def test_audit_skips_warmups_and_failed_rows_by_default(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(phase="warmup", prompt_bytes=1000), artifact_row(exit_class="timeout", tokens=1000)])
    args = parse_args([str(artifact), "--max-prompt-bytes", "32", "--max-tokens", "40", "--min-rows", "1"])

    rows, findings = qemu_prompt_budget_audit.audit([artifact], args)

    assert len(rows) == 1
    assert findings == []


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_prompt_budget_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "prompt_budget", "--max-prompt-bytes", "32", "--max-tokens", "40"]
    )

    assert status == 0
    payload = json.loads((output_dir / "prompt_budget.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No prompt budget findings." in (output_dir / "prompt_budget.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "prompt_budget.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "prompt_budget_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "prompt_budget_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_prompt_budget_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_rows_match(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(phase="warmup")])
    output_dir = tmp_path / "out"

    status = qemu_prompt_budget_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "prompt_budget", "--min-rows", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "prompt_budget.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_rows_within_prompt_and_token_budgets(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_budget_and_expected_token_failures(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_skips_warmups_and_failed_rows_by_default(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_rows_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
