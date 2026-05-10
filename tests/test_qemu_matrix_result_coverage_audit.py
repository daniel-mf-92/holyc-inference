#!/usr/bin/env python3
"""Tests for QEMU matrix result coverage audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_matrix_result_coverage_audit


def launch(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "build": "baseline",
        "command_sha256": "cmd-a",
        "profile": "unit",
        "model": "smoke",
        "quantization": "Q8_0",
        "phase": "measured",
        "prompt_id": "alpha",
        "prompt_sha256": "a" * 64,
        "iteration": 1,
    }
    row.update(overrides)
    return row


def write_matrix(path: Path, launches: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "builds": [{"build": "baseline", "command_sha256": "cmd-a"}],
                "launches": launches,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}, indent=2) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_matrix_result_coverage_audit.build_parser().parse_args(extra)


def test_audit_passes_when_matrix_launches_are_observed(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    result = tmp_path / "qemu_prompt_bench_latest.json"
    write_matrix(matrix, [launch(), launch(phase="warmup", iteration=1)])
    write_results(result, [launch(prompt="alpha"), launch(phase="warmup", prompt="alpha")])
    args = parse_args([str(matrix), str(result)])

    report = qemu_matrix_result_coverage_audit.audit(matrix, [result], args)

    assert report["status"] == "pass"
    assert report["summary"]["covered_launch_keys"] == 2
    assert report["findings"] == []


def test_audit_flags_missing_duplicate_and_unexpected_launches(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    result = tmp_path / "qemu_prompt_bench_latest.json"
    write_matrix(matrix, [launch(), launch(prompt_id="beta", prompt_sha256="b" * 64)])
    write_results(result, [launch(prompt="alpha"), launch(prompt="alpha"), launch(prompt="gamma", prompt_id="gamma", prompt_sha256="c" * 64)])
    args = parse_args([str(matrix), str(result)])

    report = qemu_matrix_result_coverage_audit.audit(matrix, [result], args)

    assert report["status"] == "fail"
    kinds = {finding["kind"] for finding in report["findings"]}
    assert {"missing_launch", "duplicate_launch", "unexpected_launch"} <= kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    result = tmp_path / "qemu_prompt_bench_latest.json"
    out = tmp_path / "out"
    write_matrix(matrix, [launch()])
    write_results(result, [launch(prompt="alpha")])

    status = qemu_matrix_result_coverage_audit.main([str(matrix), str(result), "--output-dir", str(out), "--output-stem", "coverage"])

    assert status == 0
    payload = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    rows = list(csv.DictReader((out / "coverage.csv").open(encoding="utf-8")))
    assert rows[0]["prompt_id"] == "alpha"
    assert (out / "coverage_findings.csv").read_text(encoding="utf-8").startswith("severity,kind,field,detail")
    assert "No matrix result coverage findings." in (out / "coverage.md").read_text(encoding="utf-8")
    junit_root = ET.parse(out / "coverage_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_matrix_result_coverage_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_passes_when_matrix_launches_are_observed(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_duplicate_and_unexpected_launches(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
