#!/usr/bin/env python3
"""Tests for QEMU iteration coverage audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_iteration_coverage_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "phase": "measured",
        "iteration": 1,
        "launch_index": 1,
        "exit_class": "ok",
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_iteration_coverage_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]], warmups: list[dict[str, object]] | None = None) -> None:
    path.write_text(json.dumps({"warmups": warmups or [], "benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_contiguous_prompt_iterations(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [artifact_row(iteration=1, launch_index=2), artifact_row(iteration=2, launch_index=3)],
        [artifact_row(phase="warmup", iteration=1, launch_index=1)],
    )
    args = parse_args([str(artifact), "--min-measured-iterations-per-prompt", "2"])

    rows, groups, findings = qemu_iteration_coverage_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 3
    assert {group.phase for group in groups} == {"warmup", "measured"}
    measured = [group for group in groups if group.phase == "measured"][0]
    assert measured.max_iteration == 2


def test_audit_flags_iteration_gaps_duplicates_and_min_counts(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(prompt="gap", iteration=1, launch_index=1),
            artifact_row(prompt="gap", iteration=3, launch_index=2),
            artifact_row(prompt="dup", iteration=1, launch_index=3),
            artifact_row(prompt="dup", iteration=1, launch_index=4),
            artifact_row(prompt="few", iteration=1, launch_index=5, exit_class="fail"),
        ],
    )
    args = parse_args([str(artifact), "--min-measured-iterations-per-prompt", "2", "--count-only-ok"])

    rows, groups, findings = qemu_iteration_coverage_audit.audit([artifact], args)

    assert len(rows) == 5
    assert len(groups) == 3
    kinds = {finding.kind for finding in findings}
    assert {"iteration_gap", "duplicate_iteration", "min_measured_iterations_per_prompt"} <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_iteration_coverage_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "iteration"]
    )

    assert status == 0
    payload = json.loads((output_dir / "iteration.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["groups"] == 1
    assert "No iteration coverage findings." in (output_dir / "iteration.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "iteration.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "iteration_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "iteration_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_iteration_coverage_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_iteration_coverage_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "iteration"])

    assert status == 1
    payload = json.loads((output_dir / "iteration.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_contiguous_prompt_iterations(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_iteration_gaps_duplicates_and_min_counts(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
