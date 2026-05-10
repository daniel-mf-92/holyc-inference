#!/usr/bin/env python3
"""Tests for QEMU failure taxonomy audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_failure_audit


def artifact_row(prompt: str = "smoke-short", exit_class: str = "ok", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "iteration": 1,
        "exit_class": exit_class,
        "timed_out": exit_class == "timeout",
        "returncode": 0 if exit_class == "ok" else 1,
        "failure_reason": "" if exit_class == "ok" else exit_class,
        "tokens": 16 if exit_class == "ok" else None,
        "wall_elapsed_us": 50000 if exit_class == "ok" else None,
        "wall_tok_per_s": 320.0 if exit_class == "ok" else None,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_failure_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_accepts_consistent_exit_classes(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        {"benchmarks": [artifact_row("ok"), artifact_row("timeout", "timeout"), artifact_row("nonzero", "nonzero_exit")]},
    )
    args = parse_args([str(artifact), "--max-failure-pct", "75"])

    rows, summaries, findings = qemu_failure_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 3
    summary_by_class = {summary.exit_class: summary.rows for summary in summaries}
    assert summary_by_class["ok"] == 1
    assert summary_by_class["timeout"] == 1


def test_audit_flags_mismatched_failure_taxonomy(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        {
            "benchmarks": [
                artifact_row("bad-timeout", timed_out=True),
                artifact_row("bad-nonzero", "nonzero_exit", returncode=0),
                artifact_row("bad-ok", tokens=0, failure_reason="unexpected"),
                artifact_row("bad-class", "mystery"),
            ]
        },
    )
    args = parse_args([str(artifact)])

    rows, summaries, findings = qemu_failure_audit.audit([artifact], args)

    assert len(rows) == 4
    assert summaries
    kinds = {finding.kind for finding in findings}
    assert {"timeout_exit_class_mismatch", "nonzero_returncode_mismatch", "ok_has_failure_reason", "ok_missing_tokens", "unknown_exit_class"} <= kinds


def test_cli_writes_json_markdown_csv_rows_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, {"benchmarks": [artifact_row()]})
    output_dir = tmp_path / "out"

    status = qemu_failure_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "failure"])

    assert status == 0
    payload = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["ok_rows"] == 1
    assert "No failure taxonomy findings." in (output_dir / "failure.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "failure.csv").open(encoding="utf-8")))
    assert any(row["exit_class"] == "ok" for row in rows)
    detail_rows = list(csv.DictReader((output_dir / "failure_rows.csv").open(encoding="utf-8")))
    assert detail_rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "failure_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "failure_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_failure_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_failure_rate_exceeds_gate(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, {"benchmarks": [artifact_row("ok"), artifact_row("timeout", "timeout")]})
    output_dir = tmp_path / "out"

    status = qemu_failure_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "failure", "--max-failure-pct", "25"]
    )

    assert status == 1
    payload = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert any(finding["kind"] == "max_failure_pct" for finding in payload["findings"])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_exit_classes(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_mismatched_failure_taxonomy(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_rows_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_failure_rate_exceeds_gate(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
