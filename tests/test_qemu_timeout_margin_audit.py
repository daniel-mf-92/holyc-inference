#!/usr/bin/env python3
"""Tests for QEMU timeout margin audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_timeout_margin_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "iteration": 1,
        "exit_class": "ok",
        "timed_out": False,
        "timeout_seconds": 1.0,
        "wall_elapsed_us": 250_000,
        "wall_timeout_pct": 25.0,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_timeout_margin_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_rows_with_timeout_headroom(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    args = parse_args([str(artifact), "--max-ok-timeout-pct", "80"])

    rows, findings = qemu_timeout_margin_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].timeout_margin_us == 750_000


def test_audit_flags_timeout_margin_and_formula_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(prompt="too-close", wall_elapsed_us=950_000, wall_timeout_pct=95.0),
            artifact_row(prompt="drift", wall_timeout_pct=1.0),
            artifact_row(prompt="missing", timeout_seconds=None),
            artifact_row(prompt="early-timeout", exit_class="timeout", timed_out=True, wall_elapsed_us=500_000, wall_timeout_pct=50.0),
        ],
    )
    args = parse_args([str(artifact), "--max-ok-timeout-pct", "80"])

    rows, findings = qemu_timeout_margin_audit.audit([artifact], args)

    assert len(rows) == 4
    kinds = {finding.kind for finding in findings}
    assert {"ok_timeout_margin_too_small", "wall_timeout_pct_drift", "missing_timeout_seconds", "timeout_budget_underused"} <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_timeout_margin_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "timeout_margin", "--max-ok-timeout-pct", "80"]
    )

    assert status == 0
    payload = json.loads((output_dir / "timeout_margin.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["min_timeout_margin_us"] == 750_000
    assert payload["summary"]["timeout_rows"] == 0
    assert "No timeout margin findings." in (output_dir / "timeout_margin.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "timeout_margin.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "timeout_margin_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "timeout_margin_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_timeout_margin_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_timeout_margin_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "timeout_margin"])

    assert status == 1
    payload = json.loads((output_dir / "timeout_margin.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_rows_with_timeout_headroom(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_timeout_margin_and_formula_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
