#!/usr/bin/env python3
"""Tests for host-side benchmark result indexing."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import bench_result_index


def test_indexes_qemu_prompt_report_with_airgap_status(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "a" * 64, "prompt_count": 1},
                "suite_summary": {
                    "tok_per_s_median": 123.0,
                    "memory_bytes_max": 4096,
                    "host_child_cpu_us_median": 5000,
                    "host_child_cpu_pct_median": 75.5,
                },
                "warmups": [],
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([tmp_path])

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.artifact_type == "qemu_prompt"
    assert summary.command_airgap_status == "pass"
    assert summary.profile == "secure"
    assert summary.quantization == "Q4_0"
    assert summary.generated_age_seconds is not None
    assert summary.freshness_status == "unchecked"
    assert summary.median_tok_per_s == 123.0
    assert summary.host_child_cpu_us_median == 5000.0
    assert summary.host_child_cpu_pct_median == 75.5
    assert summary.max_memory_bytes == 4096


def test_qemu_prompt_report_status_reflects_failed_runs(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "d" * 64, "prompt_count": 1},
                "suite_summary": {"tok_per_s_median": 123.0},
                "warmups": [{"returncode": 0, "timed_out": False}],
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "returncode": 124,
                        "timed_out": True,
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([tmp_path])

    assert summaries[0].status == "fail"
    assert summaries[0].command_airgap_status == "pass"
    assert bench_result_index.index_status(summaries) == "fail"


def test_qemu_prompt_report_status_reflects_variability_findings(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "e" * 64, "prompt_count": 1},
                "suite_summary": {"tok_per_s_median": 123.0},
                "variability_findings": [
                    {"scope": "suite", "metric": "tok_per_s_cv_pct", "value": 20.0, "limit": 5.0}
                ],
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "returncode": 0,
                        "timed_out": False,
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([tmp_path])

    assert summaries[0].status == "fail"
    assert bench_result_index.index_status(summaries) == "fail"


def test_qemu_prompt_report_requires_measured_telemetry(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "1" * 64, "prompt_count": 1},
                "suite_summary": {},
                "benchmarks": [],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([tmp_path])

    assert summaries[0].telemetry_status == "fail"
    assert "qemu_prompt: non-positive measured run count: 0" in summaries[0].telemetry_findings
    assert "qemu_prompt: missing median tok/s" in summaries[0].telemetry_findings
    assert bench_result_index.index_status(summaries) == "fail"


def test_qemu_prompt_report_checks_every_recorded_command(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "f" * 64, "prompt_count": 2},
                "suite_summary": {"tok_per_s_median": 123.0},
                "warmups": [
                    {
                        "returncode": 0,
                        "timed_out": False,
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "returncode": 0,
                        "timed_out": False,
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    },
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "returncode": 0,
                        "timed_out": False,
                        "command": [
                            "qemu-system-x86_64",
                            "-device",
                            "rtl8139",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([tmp_path])

    assert summaries[0].command_airgap_status == "fail"
    assert any(finding.startswith("measured[1]:") for finding in summaries[0].command_findings)
    assert bench_result_index.index_status(summaries) == "fail"


def test_qemu_prompt_report_marks_stale_commit_and_cli_can_fail(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    stale_commit = "deadbeef0000"
    report = input_dir / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "a" * 64, "prompt_count": 1},
                "suite_summary": {"tok_per_s_median": 123.0},
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "commit": stale_commit,
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([input_dir])

    assert summaries[0].commit == stale_commit
    assert summaries[0].commit_status == "pass"
    assert summaries[0].current_commit_match is False
    assert summaries[0] in bench_result_index.commit_drift(summaries)
    assert (
        bench_result_index.main(
            ["--input", str(input_dir), "--output-dir", str(output_dir), "--fail-on-stale-commit"]
        )
        == 1
    )
    payload = json.loads((output_dir / "bench_result_index_latest.json").read_text(encoding="utf-8"))
    assert payload["artifacts"][0]["current_commit_match"] is False


def test_mixed_commit_metadata_fails_index_and_junit(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "b" * 64, "prompt_count": 2},
                "suite_summary": {"tok_per_s_median": 123.0},
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "commit": "abc123",
                        "command": ["qemu-system-x86_64", "-nic", "none"],
                    },
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "commit": "def456",
                        "command": ["qemu-system-x86_64", "-nic", "none"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([tmp_path])
    report_payload = {
        "artifacts": [summary.__dict__ for summary in summaries],
        "prompt_suite_drift": [],
    }
    root = ET.fromstring(bench_result_index.junit_report(report_payload))

    assert summaries[0].commit_status == "fail"
    assert bench_result_index.index_status(summaries) == "fail"
    assert root.attrib["tests"] == "8"
    assert root.attrib["failures"] == "1"
    failure = root.find("./testcase[@name='commit_metadata']/failure")
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_commit_metadata_failure"


def test_indexes_matrix_cells_and_flags_network_devices(tmp_path: Path) -> None:
    report = tmp_path / "bench_matrix_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "cells": [
                    {
                        "status": "pass",
                        "profile": "unsafe",
                        "model": "tiny",
                        "quantization": "Q8_0",
                        "prompt_suite_sha256": "b" * 64,
                        "prompts": 2,
                        "measured_runs": 4,
                        "warmup_runs": 1,
                        "median_tok_per_s": 99.5,
                        "max_memory_bytes": 8192,
                        "command": [
                            "qemu-system-x86_64",
                            "-device",
                            "virtio-net-pci",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([report])

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.artifact_type == "bench_matrix_cell"
    assert summary.command_airgap_status == "fail"
    assert any("network device" in finding for finding in summary.command_findings)
    assert bench_result_index.index_status(summaries) == "fail"


def test_reports_prompt_suite_drift_for_comparable_artifacts(tmp_path: Path) -> None:
    for index, suite_hash in enumerate(("a" * 64, "b" * 64), 1):
        report_dir = tmp_path / f"run{index}"
        report_dir.mkdir()
        (report_dir / "qemu_prompt_bench_latest.json").write_text(
            json.dumps(
                {
                    "generated_at": f"2026-04-27T20:0{index}:00Z",
                    "status": "pass",
                    "prompt_suite": {"suite_sha256": suite_hash, "prompt_count": 1},
                    "suite_summary": {"tok_per_s_median": 100 + index},
                    "benchmarks": [
                        {
                            "profile": "secure",
                            "model": "tiny",
                            "quantization": "Q4_0",
                            "command": [
                                "qemu-system-x86_64",
                                "-nic",
                                "none",
                                "-drive",
                                "file=TempleOS.img,format=raw,if=ide",
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    drift = bench_result_index.prompt_suite_drift(bench_result_index.load_summaries([tmp_path]))

    assert len(drift) == 1
    assert drift[0].key == "secure/tiny/Q4_0"
    assert drift[0].hashes == ["a" * 64, "b" * 64]


def test_cli_writes_json_markdown_and_csv(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "qemu_prompt_bench_latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "c" * 64, "prompt_count": 1},
                "suite_summary": {"tok_per_s_median": 160.0},
                "benchmarks": [
                    {
                        "profile": "synthetic",
                        "model": "smoke",
                        "quantization": "Q4_0",
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status = bench_result_index.main(
        ["--input", str(input_dir), "--output-dir", str(output_dir), "--fail-on-airgap"]
    )

    assert status == 0
    payload = json.loads((output_dir / "bench_result_index_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "bench_result_index_latest.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "bench_result_index_latest.csv").open(encoding="utf-8")))
    junit_root = ET.parse(output_dir / "bench_result_index_junit_latest.xml").getroot()

    assert payload["status"] == "pass"
    assert payload["artifacts"][0]["prompt_suite_sha256"] == "c" * 64
    assert payload["prompt_suite_drift"] == []
    assert payload["command_drift"] == []
    assert "Benchmark Result Index" in markdown
    assert "Prompt suite drift: none detected." in markdown
    assert "Command drift: none detected." in markdown
    assert "command_sha256" in rows[0]
    assert rows[0]["command_airgap_status"] == "pass"
    assert rows[0]["telemetry_status"] == "pass"
    assert rows[0]["freshness_status"] == "unchecked"
    assert rows[0]["generated_age_seconds"] != ""
    assert junit_root.attrib["name"] == "holyc_bench_result_index"
    assert junit_root.attrib["failures"] == "0"


def test_cli_airgap_gate_ignores_telemetry_only_failures(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "qemu_prompt_bench_latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "d" * 64, "prompt_count": 1},
                "suite_summary": {},
                "benchmarks": [
                    {
                        "profile": "synthetic",
                        "model": "smoke",
                        "quantization": "Q4_0",
                        "command": ["qemu-system-x86_64", "-nic", "none"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    airgap_status = bench_result_index.main(
        ["--input", str(input_dir), "--output-dir", str(output_dir), "--fail-on-airgap"]
    )
    telemetry_status = bench_result_index.main(
        ["--input", str(input_dir), "--output-dir", str(output_dir), "--fail-on-telemetry"]
    )

    assert airgap_status == 0
    assert telemetry_status == 1


def test_cli_writes_drift_csv_and_can_fail_on_drift(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for index, suite_hash in enumerate(("a" * 64, "b" * 64), 1):
        report_dir = input_dir / f"run{index}"
        report_dir.mkdir()
        (report_dir / "qemu_prompt_bench_latest.json").write_text(
            json.dumps(
                {
                    "generated_at": f"2026-04-27T20:1{index}:00Z",
                    "status": "pass",
                    "prompt_suite": {"suite_sha256": suite_hash, "prompt_count": 1},
                    "suite_summary": {"tok_per_s_median": 160.0},
                    "benchmarks": [
                        {
                            "profile": "synthetic",
                            "model": "smoke",
                            "quantization": "Q4_0",
                            "command": [
                                "qemu-system-x86_64",
                                "-nic",
                                "none",
                                "-drive",
                                "file=TempleOS.img,format=raw,if=ide",
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    status = bench_result_index.main(
        ["--input", str(input_dir), "--output-dir", str(output_dir), "--fail-on-drift"]
    )

    assert status == 1
    rows = list(
        csv.DictReader(
            (output_dir / "bench_result_index_prompt_suite_drift_latest.csv").open(encoding="utf-8")
        )
    )
    assert rows[0]["key"] == "synthetic/smoke/Q4_0"
    assert rows[0]["hash_count"] == "2"


def test_command_hash_drift_is_indexed_and_can_fail_gate(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for index, command_hash in enumerate(("a" * 64, "b" * 64), 1):
        report_dir = input_dir / f"run{index}"
        report_dir.mkdir()
        (report_dir / "qemu_prompt_bench_latest.json").write_text(
            json.dumps(
                {
                    "generated_at": f"2026-04-27T20:2{index}:00Z",
                    "status": "pass",
                    "command_sha256": command_hash,
                    "prompt_suite": {"suite_sha256": "c" * 64, "prompt_count": 1},
                    "suite_summary": {"tok_per_s_median": 160.0},
                    "benchmarks": [
                        {
                            "profile": "synthetic",
                            "model": "smoke",
                            "quantization": "Q4_0",
                            "command_sha256": command_hash,
                            "command": [
                                "qemu-system-x86_64",
                                "-nic",
                                "none",
                                "-drive",
                                "file=TempleOS.img,format=raw,if=ide",
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    summaries = bench_result_index.load_summaries([input_dir])
    drift = bench_result_index.command_drift(summaries)
    status = bench_result_index.main(
        ["--input", str(input_dir), "--output-dir", str(output_dir), "--fail-on-command-drift"]
    )

    assert len(drift) == 1
    assert drift[0].key == f"synthetic/smoke/Q4_0/{'c' * 64}"
    assert drift[0].hashes == ["a" * 64, "b" * 64]
    assert status == 1
    payload = json.loads((output_dir / "bench_result_index_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "bench_result_index_latest.md").read_text(encoding="utf-8")
    rows = list(
        csv.DictReader((output_dir / "bench_result_index_command_drift_latest.csv").open(encoding="utf-8"))
    )
    junit_root = ET.parse(output_dir / "bench_result_index_junit_latest.xml").getroot()

    assert payload["command_drift"][0]["key"] == f"synthetic/smoke/Q4_0/{'c' * 64}"
    assert "## Command Drift" in markdown
    assert rows[0]["hash_count"] == "2"
    failure = junit_root.find("./testcase[@name='command_drift']/failure")
    assert failure is not None
    assert failure.attrib["type"] == "command_drift"


def test_mixed_command_hash_metadata_fails_index_and_junit(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "command_sha256": "a" * 64,
                "prompt_suite": {"suite_sha256": "d" * 64, "prompt_count": 2},
                "suite_summary": {"tok_per_s_median": 123.0},
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "command_sha256": "a" * 64,
                        "command": ["qemu-system-x86_64", "-nic", "none"],
                    },
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "command_sha256": "b" * 64,
                        "command": ["qemu-system-x86_64", "-nic", "none"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([tmp_path])
    report_payload = {
        "artifacts": [summary.__dict__ for summary in summaries],
        "prompt_suite_drift": [],
        "command_drift": [],
    }
    root = ET.fromstring(bench_result_index.junit_report(report_payload))

    assert summaries[0].command_hash_status == "fail"
    assert bench_result_index.index_status(summaries) == "fail"
    assert root.attrib["tests"] == "8"
    assert root.attrib["failures"] == "1"
    failure = root.find("./testcase[@name='command_hash_metadata']/failure")
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_command_hash_metadata_failure"


def test_junit_report_marks_artifact_airgap_and_drift_failures() -> None:
    report = {
        "artifacts": [
            {
                "source": "qemu_prompt_bench_latest.json",
                "status": "fail",
                "command_airgap_status": "pass",
            },
            {
                "source": "bench_matrix_latest.json",
                "status": "pass",
                "command_airgap_status": "fail",
            },
        ],
        "prompt_suite_drift": [{"key": "secure/tiny/Q4_0"}],
        "command_drift": [{"key": "secure/tiny/Q4_0/aaaaaaaa"}],
    }

    root = ET.fromstring(bench_result_index.junit_report(report))

    assert root.attrib["name"] == "holyc_bench_result_index"
    assert root.attrib["tests"] == "8"
    assert root.attrib["failures"] == "4"
    failures = root.findall("./testcase/failure")
    assert {failure.attrib["type"] for failure in failures} == {
        "benchmark_artifact_failure",
        "airgap_violation",
        "prompt_suite_drift",
        "command_drift",
    }


def test_junit_report_marks_missing_telemetry_failure() -> None:
    report = {
        "artifacts": [
            {
                "source": "qemu_prompt_bench_latest.json",
                "status": "pass",
                "command_airgap_status": "pass",
                "telemetry_status": "fail",
                "telemetry_findings": ["qemu_prompt: missing median tok/s"],
            }
        ],
        "prompt_suite_drift": [],
    }

    root = ET.fromstring(bench_result_index.junit_report(report))

    assert root.attrib["name"] == "holyc_bench_result_index"
    assert root.attrib["failures"] == "1"
    failure = root.find("./testcase[@name='telemetry_coverage']/failure")
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_telemetry_missing"


def test_artifact_freshness_gate_marks_old_reports(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "qemu_prompt_bench_latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "e" * 64, "prompt_count": 1},
                "suite_summary": {"tok_per_s_median": 160.0},
                "benchmarks": [
                    {
                        "profile": "synthetic",
                        "model": "smoke",
                        "quantization": "Q4_0",
                        "command": ["qemu-system-x86_64", "-nic", "none"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries(
        [input_dir],
        max_artifact_age_seconds=3600,
        now=datetime(2026, 4, 27, 22, 0, 0, tzinfo=timezone.utc),
    )
    status = bench_result_index.main(
        [
            "--input",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--max-artifact-age-hours",
            "1",
            "--fail-on-stale-artifact",
        ]
    )

    assert summaries[0].generated_age_seconds == 7200
    assert summaries[0].freshness_status == "fail"
    assert "exceeds 3600s" in summaries[0].freshness_findings[0]
    assert bench_result_index.index_status(summaries) == "fail"
    assert status == 1
    payload = json.loads((output_dir / "bench_result_index_latest.json").read_text(encoding="utf-8"))
    assert payload["artifacts"][0]["freshness_status"] == "fail"
    junit_root = ET.parse(output_dir / "bench_result_index_junit_latest.xml").getroot()
    failure = junit_root.find("./testcase[@name='artifact_freshness']/failure")
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_artifact_stale"


def test_stale_artifact_gate_requires_age_threshold(tmp_path: Path) -> None:
    status = bench_result_index.main(
        ["--input", str(tmp_path), "--output-dir", str(tmp_path / "out"), "--fail-on-stale-artifact"]
    )

    assert status == 2
