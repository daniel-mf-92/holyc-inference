#!/usr/bin/env python3
"""Tests for QEMU launch integrity audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_launch_integrity_audit
import qemu_prompt_bench


def plan_rows() -> list[dict[str, object]]:
    return [
        {
            "launch_index": 1,
            "phase": "warmup",
            "prompt_index": 1,
            "prompt_id": "alpha",
            "prompt_sha256": qemu_prompt_bench.prompt_hash("Alpha prompt"),
            "prompt_bytes": qemu_prompt_bench.prompt_bytes("Alpha prompt"),
            "expected_tokens": 8,
            "iteration": 1,
        },
        {
            "launch_index": 2,
            "phase": "measured",
            "prompt_index": 1,
            "prompt_id": "alpha",
            "prompt_sha256": qemu_prompt_bench.prompt_hash("Alpha prompt"),
            "prompt_bytes": qemu_prompt_bench.prompt_bytes("Alpha prompt"),
            "expected_tokens": 8,
            "iteration": 1,
        },
    ]


def run_row(plan: dict[str, object]) -> dict[str, object]:
    return {
        "launch_index": plan["launch_index"],
        "phase": plan["phase"],
        "prompt": plan["prompt_id"],
        "prompt_sha256": plan["prompt_sha256"],
        "prompt_bytes": plan["prompt_bytes"],
        "expected_tokens": plan["expected_tokens"],
        "iteration": plan["iteration"],
    }


def write_artifact(path: Path, *, tamper_hash: bool = False, drop_measured: bool = False) -> None:
    plan = plan_rows()
    warmups = [run_row(plan[0])]
    benchmarks = [] if drop_measured else [run_row(plan[1])]
    expected = qemu_prompt_bench.launch_sequence_from_plan(plan)
    observed = qemu_launch_integrity_audit.observed_sequence_from_rows(warmups + benchmarks)
    integrity = qemu_prompt_bench.launch_sequence_integrity(expected, observed)
    payload = {
        "launch_plan": plan,
        "launch_plan_sha256": qemu_prompt_bench.launch_plan_hash(plan),
        "expected_launch_sequence_sha256": integrity["expected_launch_sequence_sha256"],
        "observed_launch_sequence_sha256": integrity["observed_launch_sequence_sha256"],
        "launch_sequence_integrity": integrity,
        "warmups": warmups,
        "benchmarks": benchmarks,
    }
    if tamper_hash:
        payload["launch_plan_sha256"] = "bad"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_audit_accepts_matching_launch_plan(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact)
    args = qemu_launch_integrity_audit.build_parser().parse_args([str(artifact), "--require-match"])

    record, findings = qemu_launch_integrity_audit.audit_artifact(artifact, args)

    assert findings == []
    assert record.launch_plan_rows == 2
    assert record.observed_rows == 2
    assert record.launch_sequence_match is True


def test_audit_flags_stale_hash_and_missing_launch(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, tamper_hash=True, drop_measured=True)
    args = qemu_launch_integrity_audit.build_parser().parse_args([str(artifact), "--require-match"])

    record, findings = qemu_launch_integrity_audit.audit_artifact(artifact, args)

    assert record.status == "fail"
    assert record.missing_launches == 1
    kinds = {finding.kind for finding in findings}
    assert {"value_mismatch", "launch_sequence_mismatch", "min_measured_rows"} <= kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact)
    output_dir = tmp_path / "out"

    status = qemu_launch_integrity_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "launch_integrity", "--require-match"]
    )

    assert status == 0
    payload = json.loads((output_dir / "launch_integrity.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["matched_launches"] == 2
    assert "QEMU Launch Integrity Audit" in (output_dir / "launch_integrity.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "launch_integrity.csv").open(encoding="utf-8")))
    assert rows[0]["launch_sequence_match"] == "True"
    finding_rows = list(csv.DictReader((output_dir / "launch_integrity_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "launch_integrity_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_launch_integrity_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    status = qemu_launch_integrity_audit.main(
        [str(tmp_path), "--output-dir", str(output_dir), "--output-stem", "launch_integrity", "--min-artifacts", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "launch_integrity.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_matching_launch_plan(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_stale_hash_and_missing_launch(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_artifacts_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
