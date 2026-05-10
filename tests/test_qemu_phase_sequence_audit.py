#!/usr/bin/env python3
"""Tests for QEMU phase sequence audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_phase_sequence_audit


def artifact_row(prompt: str = "smoke-short", phase: str = "measured", iteration: int = 1, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": phase,
        "iteration": iteration,
        "exit_class": "ok",
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_phase_sequence_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_accepts_warmup_then_measured_groups(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        {
            "warmups": [artifact_row(phase="warmup", iteration=1)],
            "benchmarks": [artifact_row(iteration=1), artifact_row(iteration=2)],
        },
    )
    args = parse_args([str(artifact), "--min-warmups-per-group", "1", "--min-measured-per-group", "2", "--require-measured-ok"])

    runs, groups, findings = qemu_phase_sequence_audit.audit([artifact], args)

    assert findings == []
    assert len(runs) == 3
    assert len(groups) == 1
    assert groups[0].warmups == 1
    assert groups[0].measured == 2


def test_audit_flags_bad_sequence_duplicate_and_failure(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        {
            "benchmarks": [
                artifact_row(iteration=1, exit_class="timeout"),
                artifact_row(phase="warmup", iteration=1),
                artifact_row(iteration=1),
            ]
        },
    )
    args = parse_args([str(artifact), "--min-warmups-per-group", "1", "--min-measured-per-group", "2", "--require-measured-ok"])

    runs, groups, findings = qemu_phase_sequence_audit.audit([artifact], args)

    assert len(runs) == 3
    assert len(groups) == 1
    kinds = {finding.kind for finding in findings}
    assert {"warmup_after_measured", "duplicate_iteration", "measured_not_ok"} <= kinds


def test_cli_writes_json_markdown_csv_runs_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, {"warmups": [artifact_row(phase="warmup")], "benchmarks": [artifact_row()]})
    output_dir = tmp_path / "out"

    status = qemu_phase_sequence_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "phase",
            "--min-warmups-per-group",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "phase.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["warmups"] == 1
    assert "No phase sequence findings." in (output_dir / "phase.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "phase.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    run_rows = list(csv.DictReader((output_dir / "phase_runs.csv").open(encoding="utf-8")))
    assert len(run_rows) == 2
    finding_rows = list(csv.DictReader((output_dir / "phase_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "phase_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_phase_sequence_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, {"benchmarks": []})
    output_dir = tmp_path / "out"

    status = qemu_phase_sequence_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "phase",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "phase.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_warmup_then_measured_groups(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_bad_sequence_duplicate_and_failure(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_runs_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
