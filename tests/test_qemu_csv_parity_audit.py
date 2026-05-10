#!/usr/bin/env python3
"""Tests for QEMU CSV parity audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_csv_parity_audit


def row(prompt: str = "alpha", tokens: int = 16) -> dict[str, object]:
    return {
        "timestamp": "2026-05-01T00:00:00Z",
        "commit": "abc123",
        "benchmark": "qemu_prompt",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt": prompt,
        "prompt_sha256": "sha-" + prompt,
        "phase": "measured",
        "launch_index": 1,
        "iteration": 1,
        "tokens": tokens,
        "expected_tokens": tokens,
        "elapsed_us": 100000,
        "wall_elapsed_us": 125000,
        "returncode": 0,
        "timed_out": False,
        "exit_class": "ok",
        "command_sha256": "cmd-sha",
        "command_airgap_ok": True,
        "command_has_explicit_nic_none": True,
        "command_has_legacy_net_none": False,
    }


def write_pair(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qemu_csv_parity_audit.DEFAULT_COMPARE_FIELDS), lineterminator="\n")
        writer.writeheader()
        for item in rows:
            writer.writerow({field: qemu_csv_parity_audit.normalize_json_value(item.get(field)) for field in writer.fieldnames})


def test_audit_accepts_matching_json_and_csv(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_pair(artifact_path, [row()])
    args = qemu_csv_parity_audit.build_parser().parse_args([str(artifact_path)])

    artifact, findings = qemu_csv_parity_audit.compare_artifact(artifact_path, args)

    assert artifact.status == "pass"
    assert artifact.json_rows == 1
    assert artifact.csv_rows == 1
    assert findings == []


def test_audit_flags_value_and_row_count_drift(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_pair(artifact_path, [row("alpha", 16), row("beta", 24)])
    csv_path = artifact_path.with_suffix(".csv")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    rows[0]["tokens"] = "15"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerow(rows[0])
    args = qemu_csv_parity_audit.build_parser().parse_args([str(artifact_path)])

    artifact, findings = qemu_csv_parity_audit.compare_artifact(artifact_path, args)

    assert artifact.status == "fail"
    assert {"value_mismatch", "row_count_mismatch"} <= {finding.kind for finding in findings}


def test_cli_writes_reports(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_pair(artifact_path, [row()])

    status = qemu_csv_parity_audit.main(
        [
            str(artifact_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "parity",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "parity.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["compared_values"] == len(qemu_csv_parity_audit.DEFAULT_COMPARE_FIELDS)
    assert "QEMU CSV Parity Audit" in (output_dir / "parity.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "parity.csv").open(encoding="utf-8", newline="")))
    assert rows[0]["status"] == "pass"
    finding_rows = list(csv.DictReader((output_dir / "parity_findings.csv").open(encoding="utf-8", newline="")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "parity_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_csv_parity_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    status = qemu_csv_parity_audit.main(
        [
            str(tmp_path / "empty"),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "parity",
            "--min-artifacts",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "parity.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_matching_json_and_csv(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_value_and_row_count_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_reports(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_artifacts_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
