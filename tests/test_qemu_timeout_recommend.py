#!/usr/bin/env python3
"""Tests for QEMU timeout recommendation reporting."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_timeout_recommend


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "benchmark": "qemu_prompt",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "prompt": "smoke-short",
        "wall_elapsed_us": 10_000_000,
        "timeout_seconds": 60,
        "returncode": 0,
        "timed_out": False,
        "exit_class": "ok",
    }
    payload.update(overrides)
    return payload


def test_recommends_timeout_from_p95_wall_time(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row(prompt="a", wall_elapsed_us=10_000_000),
            row(prompt="b", wall_elapsed_us=20_000_000),
            row(prompt="c", wall_elapsed_us=30_000_000),
        ],
    )
    args = qemu_timeout_recommend.build_parser().parse_args(
        [str(artifact), "--min-timeout-seconds", "1", "--p95-multiplier", "2", "--additive-seconds", "5"]
    )

    report = qemu_timeout_recommend.audit([artifact], args)

    assert report["status"] == "pass"
    recommendation = report["recommendations"][0]
    assert recommendation["samples"] == 3
    assert recommendation["prompts"] == 3
    assert recommendation["p95_wall_s"] == 29.0
    assert recommendation["recommended_timeout_s"] == 63


def test_flags_low_sample_count_and_missing_timeout_telemetry(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(timeout_seconds=None)])
    args = qemu_timeout_recommend.build_parser().parse_args(
        [str(artifact), "--min-samples", "2", "--require-timeout-telemetry"]
    )

    report = qemu_timeout_recommend.audit([artifact], args)
    kinds = {finding["kind"] for finding in report["findings"]}

    assert report["status"] == "fail"
    assert {"low_sample_count", "missing_timeout_telemetry"} <= kinds


def test_filters_failed_rows_by_default(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row(prompt="ok", wall_elapsed_us=1_000_000),
            row(prompt="timeout", wall_elapsed_us=100_000_000, timed_out=True, returncode=124, exit_class="timeout"),
        ],
    )
    args = qemu_timeout_recommend.build_parser().parse_args([str(artifact), "--min-timeout-seconds", "1"])

    report = qemu_timeout_recommend.audit([artifact], args)

    assert report["status"] == "pass"
    assert report["recommendations"][0]["samples"] == 1
    assert report["recommendations"][0]["max_wall_s"] == 1.0


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row()])
    output_dir = tmp_path / "out"

    status = qemu_timeout_recommend.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "timeout"])

    assert status == 0
    payload = json.loads((output_dir / "timeout.json").read_text(encoding="utf-8"))
    assert payload["summary"]["groups"] == 1
    assert "QEMU Timeout Recommendations" in (output_dir / "timeout.md").read_text(encoding="utf-8")
    assert "recommended_timeout_s" in (output_dir / "timeout.csv").read_text(encoding="utf-8")
    assert "kind" in (output_dir / "timeout_findings.csv").read_text(encoding="utf-8")
    root = ET.parse(output_dir / "timeout_junit.xml").getroot()
    assert root.attrib["name"] == "holyc_qemu_timeout_recommend"
    assert root.attrib["failures"] == "0"
