#!/usr/bin/env python3
"""Tests for host-side benchmark artifact manifests."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import bench_artifact_manifest
import bench_result_index


def make_summary(source: Path, **overrides: object) -> bench_result_index.ArtifactSummary:
    values = {
        "source": str(source),
        "artifact_type": "qemu_prompt",
        "status": "pass",
        "generated_at": "2026-04-28T02:50:00Z",
        "generated_age_seconds": 60,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt_suite_sha256": "a" * 64,
        "command_sha256": "b" * 64,
        "prompts": 2,
        "measured_runs": 2,
        "warmup_runs": 0,
        "median_tok_per_s": 128.0,
        "max_memory_bytes": 4096,
        "telemetry_status": "pass",
        "telemetry_findings": [],
        "command_hash_status": "pass",
        "command_hash_findings": [],
        "command_airgap_status": "pass",
        "command_findings": [],
        "commit": "abc123",
        "current_commit": "abc123",
        "current_commit_match": True,
        "commit_status": "pass",
        "commit_findings": [],
        "freshness_status": "unchecked",
        "freshness_findings": [],
    }
    values.update(overrides)
    return bench_result_index.ArtifactSummary(**values)  # type: ignore[arg-type]


def test_manifest_includes_commit_metadata_in_json_markdown_and_csv(tmp_path: Path) -> None:
    source = tmp_path / "qemu_prompt_bench_latest.json"
    source.write_text('{"status":"pass"}\n', encoding="utf-8")

    output_path, status, _history = bench_artifact_manifest.write_manifest([make_summary(source)], tmp_path)

    assert status == "pass"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    latest = payload["latest_artifacts"][0]
    assert latest["commit"] == "abc123"
    assert latest["current_commit"] == "abc123"
    assert latest["current_commit_match"] is True
    assert latest["commit_status"] == "pass"

    markdown = (tmp_path / "bench_artifact_manifest_latest.md").read_text(encoding="utf-8")
    assert "Commit" in markdown
    assert "pass:abc123" in markdown

    rows = list(csv.DictReader((tmp_path / "bench_artifact_manifest_latest.csv").open(encoding="utf-8")))
    assert rows[0]["commit"] == "abc123"
    assert rows[0]["current_commit_match"] == "True"
    history_rows = list(
        csv.DictReader((tmp_path / "bench_artifact_manifest_history_latest.csv").open(encoding="utf-8"))
    )
    assert history_rows[0]["source"] == str(source)


def test_manifest_status_and_junit_fail_on_inconsistent_commit_metadata(tmp_path: Path) -> None:
    source = tmp_path / "qemu_prompt_bench_latest.json"
    source.write_text('{"status":"pass"}\n', encoding="utf-8")
    summary = make_summary(
        source,
        commit_status="fail",
        commit="abc123,def456",
        commit_findings=["qemu_prompt: mixed commits: abc123,def456"],
    )

    output_path, status, _history = bench_artifact_manifest.write_manifest([summary], tmp_path)

    assert status == "fail"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["latest_artifacts"][0]["commit_status"] == "fail"

    root = ET.parse(tmp_path / "bench_artifact_manifest_junit_latest.xml").getroot()
    assert root.attrib["name"] == "holyc_bench_artifact_manifest"
    assert root.attrib["tests"] == "7"
    assert root.attrib["failures"] == "1"
    failure = root.find("./testcase[@name='commit_metadata']/failure")
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_commit_metadata_failure"


def test_manifest_history_csv_keeps_superseded_artifacts(tmp_path: Path) -> None:
    older_source = tmp_path / "qemu_prompt_bench_older.json"
    newer_source = tmp_path / "qemu_prompt_bench_latest.json"
    older_source.write_text('{"status":"pass","generation":"older"}\n', encoding="utf-8")
    newer_source.write_text('{"status":"pass","generation":"newer"}\n', encoding="utf-8")

    bench_artifact_manifest.write_manifest(
        [
            make_summary(older_source, generated_at="2026-04-28T02:40:00Z", median_tok_per_s=100.0),
            make_summary(newer_source, generated_at="2026-04-28T02:50:00Z", median_tok_per_s=128.0),
        ],
        tmp_path,
    )

    latest_rows = list(csv.DictReader((tmp_path / "bench_artifact_manifest_latest.csv").open(encoding="utf-8")))
    history_rows = list(
        csv.DictReader((tmp_path / "bench_artifact_manifest_history_latest.csv").open(encoding="utf-8"))
    )

    assert [row["source"] for row in latest_rows] == [str(newer_source)]
    assert [row["source"] for row in history_rows] == [str(older_source), str(newer_source)]


def test_manifest_stale_commit_gate_is_opt_in(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    report = input_dir / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-28T02:50:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "b" * 64, "prompt_count": 1},
                "suite_summary": {"tok_per_s_median": 123.0},
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "ci-airgap-smoke",
                        "model": "synthetic-smoke",
                        "quantization": "Q8_0",
                        "commit": "definitely-stale",
                        "command": ["qemu-system-x86_64", "-nic", "none"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert bench_artifact_manifest.main(["--input", str(input_dir), "--output-dir", str(output_dir)]) == 0
    assert (
        bench_artifact_manifest.main(
            [
                "--input",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--fail-on-stale-commit",
            ]
        )
        == 1
    )
