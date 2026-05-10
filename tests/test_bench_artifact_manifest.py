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
        "launch_plan_sha256": "c" * 64,
        "environment_sha256": "d" * 64,
        "host_platform": "ci",
        "host_machine": "host",
        "qemu_version": "synthetic",
        "qemu_bin": "qemu-system-x86_64",
        "prompts": 2,
        "expected_token_prompts": 2,
        "expected_tokens_total": 64,
        "expected_tokens_matches": 2,
        "expected_tokens_mismatches": 0,
        "measured_prompt_bytes_total": 32,
        "prompt_bytes_min": 16,
        "prompt_bytes_max": 16,
        "prompt_bytes_per_s_median": 64.0,
        "wall_prompt_bytes_per_s_median": None,
        "tokens_per_prompt_byte_median": 2.0,
        "total_tokens": 64,
        "total_elapsed_us": 500000,
        "measured_runs": 2,
        "warmup_runs": 0,
        "median_tok_per_s": 128.0,
        "wall_tok_per_s_median": None,
        "ttft_us_p95": None,
        "host_overhead_pct_median": None,
        "host_child_cpu_us_median": None,
        "host_child_cpu_pct_median": None,
        "host_child_tok_per_cpu_s_median": None,
        "host_child_peak_rss_bytes_max": None,
        "us_per_token_median": None,
        "wall_us_per_token_median": None,
        "max_memory_bytes": 4096,
        "memory_bytes_per_token_median": None,
        "memory_bytes_per_token_max": None,
        "serial_output_bytes_total": None,
        "serial_output_bytes_max": None,
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

    output_path, status, *_ = bench_artifact_manifest.write_manifest([make_summary(source)], tmp_path)

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


def test_manifest_carries_host_child_efficiency_and_rss(tmp_path: Path) -> None:
    source = tmp_path / "qemu_prompt_bench_latest.json"
    source.write_text('{"status":"pass"}\n', encoding="utf-8")

    output_path, status, *_ = bench_artifact_manifest.write_manifest(
        [
            make_summary(
                source,
                host_child_tok_per_cpu_s_median=640.5,
                host_child_peak_rss_bytes_max=123456,
            )
        ],
        tmp_path,
    )

    assert status == "pass"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    latest = payload["latest_artifacts"][0]
    assert latest["host_child_tok_per_cpu_s_median"] == 640.5
    assert latest["host_child_peak_rss_bytes_max"] == 123456

    markdown = (tmp_path / "bench_artifact_manifest_latest.md").read_text(encoding="utf-8")
    assert "Host child tok/CPU s" in markdown
    assert "640.500" in markdown

    rows = list(csv.DictReader((tmp_path / "bench_artifact_manifest_latest.csv").open(encoding="utf-8")))
    assert rows[0]["host_child_tok_per_cpu_s_median"] == "640.5"
    assert rows[0]["host_child_peak_rss_bytes_max"] == "123456"
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

    output_path, status, *_ = bench_artifact_manifest.write_manifest([summary], tmp_path)

    assert status == "fail"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["latest_artifacts"][0]["commit_status"] == "fail"

    root = ET.parse(tmp_path / "bench_artifact_manifest_junit_latest.xml").getroot()
    assert root.attrib["name"] == "holyc_bench_artifact_manifest"
    assert root.attrib["tests"] == "12"
    assert root.attrib["failures"] == "1"
    failure = root.find("./testcase[@name='commit_metadata']/failure")
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_commit_metadata_failure"


def test_manifest_sample_coverage_gates_runs_and_tokens(tmp_path: Path) -> None:
    source = tmp_path / "qemu_prompt_bench_latest.json"
    source.write_text('{"status":"pass"}\n', encoding="utf-8")

    output_path, status, _history, _history_coverage, sample_coverage, *_ = bench_artifact_manifest.write_manifest(
        [make_summary(source, measured_runs=1, total_tokens=16)],
        tmp_path,
        min_measured_runs=3,
        min_total_tokens=64,
    )

    assert status == "fail"
    assert [(item.metric, item.observed, item.required) for item in sample_coverage] == [
        ("measured_runs", 1, 3),
        ("total_tokens", 16, 64),
    ]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["min_measured_runs"] == 3
    assert payload["min_total_tokens"] == 64
    assert len(payload["sample_coverage_violations"]) == 2

    markdown = (tmp_path / "bench_artifact_manifest_latest.md").read_text(encoding="utf-8")
    assert "Sample Coverage Violations" in markdown
    rows = list(csv.DictReader((tmp_path / "bench_artifact_manifest_sample_coverage_latest.csv").open(encoding="utf-8")))
    assert [row["metric"] for row in rows] == ["measured_runs", "total_tokens"]

    root = ET.parse(tmp_path / "bench_artifact_manifest_junit_latest.xml").getroot()
    assert root.attrib["failures"] == "1"
    failure = root.find("./testcase[@name='sample_coverage']/failure")
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_manifest_sample_coverage"


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


def test_manifest_timestamp_collision_gate_reports_ambiguous_latest_artifacts(tmp_path: Path) -> None:
    source_a = tmp_path / "qemu_prompt_bench_a.json"
    source_b = tmp_path / "qemu_prompt_bench_b.json"
    source_a.write_text('{"status":"pass","source":"a"}\n', encoding="utf-8")
    source_b.write_text('{"status":"pass","source":"b"}\n', encoding="utf-8")

    output_path, status, _history, _history_coverage, _sample_coverage, collisions, *_ = (
        bench_artifact_manifest.write_manifest(
            [
                make_summary(source_a, generated_at="2026-04-28T02:50:00Z"),
                make_summary(source_b, generated_at="2026-04-28T02:50:00Z"),
            ],
            tmp_path,
            require_unique_timestamps=True,
        )
    )

    assert status == "fail"
    assert len(collisions) == 1
    assert collisions[0].key == "ci-airgap-smoke/synthetic-smoke/Q4_0/" + "a" * 64
    assert collisions[0].generated_at == "2026-04-28T02:50:00Z"
    assert collisions[0].sources == sorted([str(source_a), str(source_b)])
    assert len(collisions[0].sha256s) == 2

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["require_unique_timestamps"] is True
    assert len(payload["timestamp_collisions"]) == 1

    markdown = (tmp_path / "bench_artifact_manifest_latest.md").read_text(encoding="utf-8")
    assert "Timestamp Collisions" in markdown

    rows = list(
        csv.DictReader((tmp_path / "bench_artifact_manifest_timestamp_collisions_latest.csv").open(encoding="utf-8"))
    )
    assert rows[0]["source_count"] == "2"
    assert rows[0]["sha256_count"] == "2"

    root = ET.parse(tmp_path / "bench_artifact_manifest_junit_latest.xml").getroot()
    assert root.attrib["failures"] == "1"
    failure = root.find("./testcase[@name='timestamp_uniqueness']/failure")
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_manifest_timestamp_collision"


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
