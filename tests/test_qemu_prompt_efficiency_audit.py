#!/usr/bin/env python3
"""Tests for QEMU prompt efficiency audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_efficiency_audit


def row(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "prompt": "short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "exit_class": "ok",
        "timed_out": False,
        "tokens": 32,
        "prompt_bytes": 16,
        "elapsed_us": 8_000,
        "wall_elapsed_us": 10_000,
        "tokens_per_prompt_byte": 2.0,
        "prompt_bytes_per_s": 2_000.0,
        "wall_prompt_bytes_per_s": 1_600.0,
    }
    item.update(overrides)
    return item


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_prompt_efficiency_audit.build_parser().parse_args(extra)


def test_audit_accepts_consistent_prompt_efficiency_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row(),
            row(
                prompt="long",
                tokens=64,
                prompt_bytes=32,
                elapsed_us=16_000,
                wall_elapsed_us=20_000,
                prompt_bytes_per_s=2_000.0,
                wall_prompt_bytes_per_s=1_600.0,
            ),
        ],
    )
    args = parse_args([str(artifact), "--min-rows", "2", "--min-tokens-per-prompt-byte", "1.5"])

    rows, findings = qemu_prompt_efficiency_audit.audit([artifact], args)

    assert len(rows) == 2
    assert findings == []
    assert rows[0].tokens_per_prompt_byte == 2.0


def test_audit_flags_missing_and_drifted_efficiency_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row(tokens_per_prompt_byte=1.0),
            row(prompt_bytes_per_s=""),
            row(wall_elapsed_us=20_000, wall_prompt_bytes_per_s=1_600.0),
            row(tokens=8, tokens_per_prompt_byte=0.5),
        ],
    )
    args = parse_args([str(artifact), "--min-tokens-per-prompt-byte", "1.0"])

    rows, findings = qemu_prompt_efficiency_audit.audit([artifact], args)

    assert len(rows) == 4
    kinds = {finding.kind for finding in findings}
    assert {
        "tokens_per_prompt_byte_drift",
        "missing_prompt_bytes_per_s",
        "wall_prompt_bytes_per_s_drift",
        "min_tokens_per_prompt_byte",
    } <= kinds


def test_audit_skips_warmups_and_failed_rows_by_default(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(), row(phase="warmup", tokens_per_prompt_byte=0.0), row(exit_class="nonzero_exit", prompt_bytes=0)])
    args = parse_args([str(artifact), "--min-rows", "1"])

    rows, findings = qemu_prompt_efficiency_audit.audit([artifact], args)

    assert len(rows) == 1
    assert findings == []


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row()])
    output_dir = tmp_path / "out"

    status = qemu_prompt_efficiency_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "efficiency"]
    )

    assert status == 0
    payload = json.loads((output_dir / "efficiency.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No prompt efficiency findings." in (output_dir / "efficiency.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "efficiency.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "short"
    findings = list(csv.DictReader((output_dir / "efficiency_findings.csv").open(encoding="utf-8")))
    assert findings == []
    junit_root = ET.parse(output_dir / "efficiency_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_prompt_efficiency_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_rows_match(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(phase="warmup")])
    output_dir = tmp_path / "out"

    status = qemu_prompt_efficiency_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "efficiency", "--min-rows", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "efficiency.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_prompt_efficiency_metrics(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_and_drifted_efficiency_metrics(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_skips_warmups_and_failed_rows_by_default(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_rows_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
