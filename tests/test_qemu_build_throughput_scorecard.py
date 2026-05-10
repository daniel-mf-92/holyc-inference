#!/usr/bin/env python3
"""Tests for QEMU build throughput scorecard."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_build_throughput_scorecard


def write_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    artifact.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "build": "base",
                        "profile": "unit",
                        "model": "smoke",
                        "quantization": "Q8_0",
                        "phase": "measured",
                        "exit_class": "ok",
                        "timed_out": False,
                        "tokens": 10,
                        "elapsed_us": 100000,
                        "wall_elapsed_us": 200000,
                        "tok_per_s": 100.0,
                        "wall_tok_per_s": 50.0,
                    },
                    {
                        "build": "base",
                        "profile": "unit",
                        "model": "smoke",
                        "quantization": "Q8_0",
                        "phase": "measured",
                        "exit_class": "ok",
                        "timed_out": False,
                        "tokens": 20,
                        "elapsed_us": 100000,
                        "wall_elapsed_us": 250000,
                        "tok_per_s": 200.0,
                        "wall_tok_per_s": 80.0,
                    },
                    {
                        "build": "base",
                        "profile": "unit",
                        "model": "smoke",
                        "quantization": "Q8_0",
                        "phase": "warmup",
                        "exit_class": "ok",
                        "tokens": 99,
                        "tok_per_s": 999.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_scorecard_summarizes_measured_rows(tmp_path: Path) -> None:
    artifact = write_artifact(tmp_path)

    payload = qemu_build_throughput_scorecard.build_scorecard([artifact], ["*.json"], min_rows=1)

    assert payload["status"] == "pass"
    assert payload["summary"]["groups"] == 1
    assert payload["summary"]["measured_rows"] == 2
    row = payload["rows"][0]
    assert row["build"] == "base"
    assert row["total_tokens"] == 30
    assert row["mean_tok_per_s"] == 150.0
    assert row["median_tok_per_s"] == 150.0
    assert row["stdev_tok_per_s"] == 50.0
    assert row["cv_tok_per_s"] == 50.0 / 150.0
    assert row["mean_wall_tok_per_s"] == 65.0
    assert row["stdev_wall_tok_per_s"] == 15.0
    assert row["cv_wall_tok_per_s"] == 15.0 / 65.0


def test_scorecard_reports_min_row_failures(tmp_path: Path) -> None:
    artifact = write_artifact(tmp_path)

    payload = qemu_build_throughput_scorecard.build_scorecard([artifact], ["*.json"], min_rows=3)

    assert payload["status"] == "fail"
    assert {finding["kind"] for finding in payload["findings"]} == {"min_rows"}


def test_scorecard_reports_stability_gate_failures(tmp_path: Path) -> None:
    artifact = write_artifact(tmp_path)

    payload = qemu_build_throughput_scorecard.build_scorecard(
        [artifact],
        ["*.json"],
        min_rows=1,
        max_cv=0.1,
        max_wall_cv=0.1,
    )

    assert payload["status"] == "fail"
    assert {finding["kind"] for finding in payload["findings"]} == {"max_cv", "max_wall_cv"}


def test_cli_writes_scorecard_artifacts(tmp_path: Path) -> None:
    artifact = write_artifact(tmp_path)
    output_dir = tmp_path / "out"

    status = qemu_build_throughput_scorecard.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "scorecard"])

    assert status == 0
    payload = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert payload["summary"]["total_tokens"] == 30
    rows = list(csv.DictReader((output_dir / "scorecard.csv").open(encoding="utf-8")))
    assert rows[0]["mean_tok_per_s"] == "150.0"
    assert rows[0]["stdev_tok_per_s"] == "50.0"
    assert rows[0]["cv_wall_tok_per_s"] == str(15.0 / 65.0)
    findings = list(csv.DictReader((output_dir / "scorecard_findings.csv").open(encoding="utf-8")))
    assert findings == []
    assert "QEMU Build Throughput Scorecard" in (output_dir / "scorecard.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "scorecard_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_build_throughput_scorecard"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_scorecard_summarizes_measured_rows(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_scorecard_reports_min_row_failures(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_scorecard_reports_stability_gate_failures(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_scorecard_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
