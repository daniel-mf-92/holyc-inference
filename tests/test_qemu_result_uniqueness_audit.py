#!/usr/bin/env python3
"""Tests for QEMU result uniqueness audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_result_uniqueness_audit


def artifact_row(prompt: str = "smoke-short", iteration: int = 1, launch_index: int = 1, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt_sha256": f"sha-{prompt}",
        "commit": "abc123",
        "phase": "measured",
        "iteration": iteration,
        "launch_index": launch_index,
        "command_sha256": "cmd123",
        "tokens": 32,
        "elapsed_us": 30_000,
        "exit_class": "ok",
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_result_uniqueness_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_unique_result_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(iteration=1), artifact_row(iteration=2, launch_index=2)])
    args = parse_args([str(artifact), "--require-launch-index", "--require-iteration"])

    rows, findings = qemu_result_uniqueness_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 2
    assert len({row.identity for row in rows}) == 2


def test_audit_flags_duplicate_result_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row()])
    args = parse_args([str(artifact), "--require-launch-index", "--require-iteration"])

    rows, findings = qemu_result_uniqueness_audit.audit([artifact], args)

    assert len(rows) == 2
    assert {finding.kind for finding in findings} == {"duplicate_result_identity"}
    assert "duplicates" in findings[0].detail


def test_audit_flags_missing_identity_fields_when_required(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(launch_index=None, iteration=None)])
    args = parse_args([str(artifact), "--require-launch-index", "--require-iteration"])

    rows, findings = qemu_result_uniqueness_audit.audit([artifact], args)

    assert len(rows) == 1
    assert {"missing_launch_index", "missing_iteration"} <= {finding.kind for finding in findings}


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_result_uniqueness_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "unique",
            "--require-launch-index",
            "--require-iteration",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "unique.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["unique_identities"] == 1
    assert "No result uniqueness findings." in (output_dir / "unique.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "unique.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "unique_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "unique_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_result_uniqueness_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_result_uniqueness_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "unique",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "unique.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_unique_result_rows(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_duplicate_result_identity(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_identity_fields_when_required(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
