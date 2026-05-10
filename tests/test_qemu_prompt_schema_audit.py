#!/usr/bin/env python3
"""Tests for QEMU prompt schema audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_bench
import qemu_prompt_schema_audit


COMMAND = ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"]


def bench_row(*, command: list[str] | None = None, exit_class: str = "ok") -> dict[str, object]:
    row_command = command or COMMAND
    airgap = qemu_prompt_bench.command_airgap_metadata(row_command)
    return {
        "benchmark": "qemu_prompt",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "launch_index": 1,
        "prompt": "alpha",
        "prompt_sha256": qemu_prompt_bench.prompt_hash("Alpha prompt"),
        "prompt_bytes": qemu_prompt_bench.prompt_bytes("Alpha prompt"),
        "iteration": 1,
        "timestamp": "2026-04-30T00:00:00Z",
        "tokens": 16,
        "elapsed_us": 100000,
        "wall_elapsed_us": 125000,
        "timeout_seconds": 5.0,
        "host_overhead_us": 25000,
        "tok_per_s": 160.0,
        "wall_tok_per_s": 128.0,
        "ttft_us": 8000,
        "memory_bytes": 67174400,
        "returncode": 0 if exit_class == "ok" else 1,
        "timed_out": False,
        "exit_class": exit_class,
        "command": row_command,
        "command_sha256": qemu_prompt_bench.command_hash(row_command),
        "command_airgap_ok": airgap["ok"],
        "command_has_explicit_nic_none": airgap["explicit_nic_none"],
        "command_has_legacy_net_none": airgap["legacy_net_none"],
        "command_airgap_violations": tuple(airgap["violations"]),
    }


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-30T00:00:00Z",
                "status": "pass",
                "command": COMMAND,
                "command_sha256": qemu_prompt_bench.command_hash(COMMAND),
                "command_airgap": qemu_prompt_bench.command_airgap_metadata(COMMAND),
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "prompt_suite": {"source": "bench/prompts/smoke.jsonl", "prompt_count": 1},
                "launch_plan": [],
                "launch_sequence_integrity": {},
                "planned_warmup_launches": 0,
                "planned_measured_launches": len(rows),
                "planned_total_launches": len(rows),
                "warmups": [],
                "suite_summary": {"runs": len(rows)},
                "phase_summaries": [],
                "benchmarks": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_accepts_valid_qemu_prompt_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [bench_row()])
    args = qemu_prompt_schema_audit.build_parser().parse_args([str(artifact_path), "--require-success"])

    artifact, findings = qemu_prompt_schema_audit.audit_artifact(artifact_path, args)

    assert artifact.status == "pass"
    assert artifact.measured_rows == 1
    assert artifact.command_airgap_ok is True
    assert findings == []


def test_audit_flags_legacy_network_and_bad_hash(tmp_path: Path) -> None:
    bad_command = ["qemu-system-x86_64", "-net", "none", "-serial", "stdio"]
    row = bench_row(command=bad_command)
    row["command_sha256"] = "bad"
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [row])
    args = qemu_prompt_schema_audit.build_parser().parse_args([str(artifact_path)])

    artifact, findings = qemu_prompt_schema_audit.audit_artifact(artifact_path, args)

    assert artifact.status == "fail"
    assert {"legacy_net_none", "missing_nic_none", "command_hash"} <= {finding.kind for finding in findings}


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [bench_row()])
    output_dir = tmp_path / "out"

    status = qemu_prompt_schema_audit.main(
        [
            str(artifact_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "schema",
            "--require-success",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "schema.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["measured_rows"] == 1
    assert "QEMU Prompt Schema Audit" in (output_dir / "schema.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "schema.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "pass"
    finding_rows = list(csv.DictReader((output_dir / "schema_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "schema_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_prompt_schema_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    status = qemu_prompt_schema_audit.main(
        [
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "schema",
            "--min-artifacts",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "schema.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pass"
        path.mkdir()
        test_audit_accepts_valid_qemu_prompt_artifact(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad"
        path.mkdir()
        test_audit_flags_legacy_network_and_bad_hash(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cli"
        path.mkdir()
        test_cli_writes_json_markdown_csv_and_junit(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty"
        path.mkdir()
        test_cli_fails_when_no_artifacts_match(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
