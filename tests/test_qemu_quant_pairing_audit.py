#!/usr/bin/env python3
"""Tests for QEMU quant pairing audit."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_quant_pairing_audit


def row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "prompt": "smoke-short",
        "phase": "measured",
        "iteration": 1,
        "commit": "abc123",
        "quantization": "Q4_0",
        "exit_class": "ok",
        "timed_out": False,
        "failure_reason": None,
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_quant_pairing_audit.build_parser().parse_args(extra)


def test_complete_pair_passes(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(quantization="Q4_0"), row(quantization="Q8_0")])
    args = parse_args([str(artifact), "--require-success"])

    rows, findings = qemu_quant_pairing_audit.collect_rows(args.inputs, args.pattern, only_measured=True)
    report = qemu_quant_pairing_audit.build_report(rows, findings, args)

    assert report["status"] == "pass"
    assert report["summary"]["complete_pairs"] == 1
    assert report["pairs"][0]["present_quantizations"] == "Q4_0,Q8_0"


def test_missing_quant_and_failed_pair_are_reported(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row(quantization="Q4_0"),
            row(prompt="other", quantization="Q8_0"),
            row(prompt="failed", quantization="Q4_0"),
            row(prompt="failed", quantization="Q8_0", exit_class="timeout", timed_out=True),
        ],
    )
    args = parse_args([str(artifact), "--require-success"])

    rows, findings = qemu_quant_pairing_audit.collect_rows(args.inputs, args.pattern, only_measured=True)
    report = qemu_quant_pairing_audit.build_report(rows, findings, args)

    kinds = {finding["kind"] for finding in report["findings"]}
    assert report["status"] == "fail"
    assert "missing_quant_pair" in kinds
    assert "incomplete_success_pair" in kinds


def test_warmups_are_ignored_by_default(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(quantization="Q4_0"), row(quantization="Q8_0", phase="warmup")])
    args = parse_args([str(artifact)])

    rows, findings = qemu_quant_pairing_audit.collect_rows(args.inputs, args.pattern, only_measured=not args.include_warmups)
    report = qemu_quant_pairing_audit.build_report(rows, findings, args)

    assert report["status"] == "fail"
    assert report["summary"]["complete_pairs"] == 0


def test_success_falls_back_to_zero_returncode_for_older_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row(quantization="Q4_0", exit_class="", returncode=0),
            row(quantization="Q8_0", exit_class="", returncode=0),
        ],
    )
    args = parse_args([str(artifact), "--require-success"])

    rows, findings = qemu_quant_pairing_audit.collect_rows(args.inputs, args.pattern, only_measured=True)
    report = qemu_quant_pairing_audit.build_report(rows, findings, args)

    assert report["status"] == "pass"
    assert report["pairs"][0]["successful_rows"] == 2


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(quantization="Q4_0"), row(quantization="Q8_0")])
    output_dir = tmp_path / "out"

    status = qemu_quant_pairing_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "pairing"])

    assert status == 0
    payload = json.loads((output_dir / "pairing.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert "QEMU Quant Pairing Audit" in (output_dir / "pairing.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "pairing.csv").open(encoding="utf-8")))
    assert rows[0]["missing_quantizations"] == ""
    finding_rows = list(csv.DictReader((output_dir / "pairing_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "pairing_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_quant_pairing_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_complete_pair_passes(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_missing_quant_and_failed_pair_are_reported(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_warmups_are_ignored_by_default(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_success_falls_back_to_zero_returncode_for_older_artifacts(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
