#!/usr/bin/env python3
"""Tests for host-side performance regression dashboard tooling."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import perf_regression


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_load_records_ignores_non_perf_json(tmp_path: Path) -> None:
    result = tmp_path / "quant_audit_latest.json"
    result.write_text('{"status": "pass", "source_audit": {"files_scanned": 1}}\n', encoding="utf-8")

    records = perf_regression.load_records([tmp_path])

    assert records == []


def test_detects_tok_per_s_and_memory_regressions(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 100.0,
                "memory_bytes": 1000,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 90.0,
                "memory_bytes": 1200,
            },
        ],
    )

    records = perf_regression.load_records([result])
    regressions = perf_regression.detect_regressions(records, 5.0, 10.0)

    assert [regression.metric for regression in regressions] == ["tok_per_s", "memory_bytes"]
    assert regressions[0].candidate_commit == "head"


def test_detects_optional_wall_tok_per_s_regressions(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "wall_tok_per_s": 80.0,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "wall_tok_per_s": 70.0,
            },
        ],
    )

    records = perf_regression.load_records([result])
    assert records[0].wall_tok_per_s == 80.0
    assert perf_regression.detect_regressions(records, 5.0, 10.0) == []

    regressions = perf_regression.detect_regressions(
        records,
        5.0,
        10.0,
        wall_tok_threshold_pct=5.0,
    )

    assert len(regressions) == 1
    assert regressions[0].metric == "wall_tok_per_s"
    assert regressions[0].baseline_value == 80.0
    assert regressions[0].candidate_value == 70.0


def test_detects_optional_p05_tok_per_s_regressions(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T10:01:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T10:02:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 40.0,
            },
            {
                "timestamp": "2026-04-27T11:01:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T11:02:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 100.0,
            },
        ],
    )

    records = perf_regression.load_records([result])
    assert perf_regression.detect_regressions(records, 5.0, 10.0) == []

    regressions = perf_regression.detect_regressions(
        records,
        5.0,
        10.0,
        p05_tok_threshold_pct=5.0,
    )

    assert len(regressions) == 1
    assert regressions[0].metric == "tok_per_s_p05"
    assert regressions[0].baseline_value == 100.0
    assert regressions[0].candidate_value == 46.0


def test_detects_optional_emitted_token_drop_regressions(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tokens": 128,
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "output_tokens": 96,
                "tok_per_s": 100.0,
            },
        ],
    )

    records = perf_regression.load_records([result])
    assert [record.tokens for record in records] == [128, 96]
    assert perf_regression.detect_regressions(records, 5.0, 10.0) == []

    regressions = perf_regression.detect_regressions(
        records,
        5.0,
        10.0,
        token_drop_threshold_pct=20.0,
    )

    assert len(regressions) == 1
    assert regressions[0].metric == "tokens"
    assert regressions[0].baseline_value == 128.0
    assert regressions[0].candidate_value == 96.0


def test_detects_optional_ttft_us_latency_regressions(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "ttft_ms": 50.0,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "time_to_first_token_us": 60000,
            },
        ],
    )

    records = perf_regression.load_records([result])
    assert [record.ttft_us for record in records] == [50000, 60000]
    assert perf_regression.detect_regressions(records, 5.0, 10.0) == []

    regressions = perf_regression.detect_regressions(
        records,
        5.0,
        10.0,
        ttft_threshold_pct=10.0,
    )

    assert len(regressions) == 1
    assert regressions[0].metric == "ttft_us"
    assert regressions[0].baseline_value == 50000.0
    assert regressions[0].candidate_value == 60000.0


def test_regression_detection_compares_latest_distinct_commits(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 90.0,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 92.0,
            },
        ],
    )

    records = perf_regression.load_records([result])
    regressions = perf_regression.detect_regressions(records, 5.0, 10.0)

    assert len(regressions) == 1
    assert regressions[0].baseline_commit == "base"
    assert regressions[0].candidate_commit == "head"
    assert regressions[0].candidate_value == 91.0


def test_explicit_baseline_and_candidate_commits(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "old",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 80.0,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T12:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 93.0,
            },
        ],
    )

    records = perf_regression.load_records([result])
    regressions = perf_regression.detect_regressions(
        records,
        5.0,
        10.0,
        baseline_commit="base",
        candidate_commit="head",
    )

    assert len(regressions) == 1
    assert regressions[0].baseline_commit == "base"


def test_memory_regression_uses_commit_point_max_memory(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q8_0",
                "memory_bytes": 1000,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q8_0",
                "memory_bytes": 1150,
            },
        ],
    )

    records = perf_regression.load_records([result])
    regressions = perf_regression.detect_regressions(records, 5.0, 10.0)

    assert len(regressions) == 1
    assert regressions[0].metric == "memory_bytes"
    assert regressions[0].baseline_value == 1000.0
    assert regressions[0].candidate_value == 1150.0


def test_explicit_comparison_commits_must_exist_for_each_key() -> None:
    records = [
        perf_regression.PerfRecord(
            source="fixture.jsonl",
            commit="base",
            timestamp="2026-04-28T00:00:00Z",
            benchmark="qemu_prompt",
            profile="ci",
            model="synthetic",
            quantization="Q4_0",
            prompt="short",
            tok_per_s=100.0,
            wall_tok_per_s=90.0,
            memory_bytes=1024,
        ),
        perf_regression.PerfRecord(
            source="fixture.jsonl",
            commit="head",
            timestamp="2026-04-28T00:01:00Z",
            benchmark="qemu_prompt",
            profile="ci",
            model="synthetic",
            quantization="Q4_0",
            prompt="short",
            tok_per_s=99.0,
            wall_tok_per_s=89.0,
            memory_bytes=1024,
        ),
        perf_regression.PerfRecord(
            source="fixture.jsonl",
            commit="head",
            timestamp="2026-04-28T00:01:00Z",
            benchmark="qemu_prompt",
            profile="ci",
            model="synthetic",
            quantization="Q8_0",
            prompt="short",
            tok_per_s=101.0,
            wall_tok_per_s=91.0,
            memory_bytes=1024,
        ),
    ]

    report = perf_regression.build_report(
        records,
        tok_threshold_pct=5.0,
        memory_threshold_pct=10.0,
        baseline_commit="base",
        candidate_commit="head",
    )

    assert report["status"] == "fail"
    assert report["comparison_coverage_violations"] == [
        {
            "key": "qemu_prompt/ci/synthetic/Q8_0/short",
            "baseline_commit": "base",
            "candidate_commit": "head",
            "missing_commits": "baseline:base",
        }
    ]
    assert "Comparison coverage violations: 1" in perf_regression.markdown_report(report)

    junit_root = ET.fromstring(perf_regression.junit_report(report))
    assert junit_root.attrib["failures"] == "1"
    failure = junit_root.find("./testcase/failure")
    assert failure is not None
    assert failure.attrib["type"] == "comparison_coverage"


def test_report_includes_baseline_candidate_comparison_rows() -> None:
    records = [
        perf_regression.PerfRecord(
            source="fixture.jsonl",
            commit="base",
            timestamp="2026-04-28T00:00:00Z",
            benchmark="qemu_prompt",
            profile="ci",
            model="synthetic",
            quantization="Q4_0",
            prompt="short",
            tok_per_s=100.0,
            wall_tok_per_s=90.0,
            memory_bytes=1000,
            ttft_us=50000,
        ),
        perf_regression.PerfRecord(
            source="fixture.jsonl",
            commit="head",
            timestamp="2026-04-28T00:01:00Z",
            benchmark="qemu_prompt",
            profile="ci",
            model="synthetic",
            quantization="Q4_0",
            prompt="short",
            tok_per_s=95.0,
            wall_tok_per_s=81.0,
            memory_bytes=1100,
            ttft_us=55000,
        ),
    ]

    report = perf_regression.build_report(records, 10.0, 20.0)

    assert report["status"] == "pass"
    assert report["comparisons"] == [
        {
            "key": "qemu_prompt/ci/synthetic/Q4_0/short",
            "baseline_commit": "base",
            "candidate_commit": "head",
            "baseline_latest_timestamp": "2026-04-28T00:00:00Z",
            "candidate_latest_timestamp": "2026-04-28T00:01:00Z",
            "baseline_records": 1,
            "candidate_records": 1,
            "median_tok_per_s_baseline": 100.0,
            "median_tok_per_s_candidate": 95.0,
            "median_tok_per_s_delta_pct": 5.0,
            "p05_tok_per_s_baseline": 100.0,
            "p05_tok_per_s_candidate": 95.0,
            "p05_tok_per_s_delta_pct": 5.0,
            "median_wall_tok_per_s_baseline": 90.0,
            "median_wall_tok_per_s_candidate": 81.0,
            "median_wall_tok_per_s_delta_pct": 10.0,
            "median_us_per_token_baseline": None,
            "median_us_per_token_candidate": None,
            "median_us_per_token_delta_pct": None,
            "median_wall_us_per_token_baseline": None,
            "median_wall_us_per_token_candidate": None,
            "median_wall_us_per_token_delta_pct": None,
            "max_memory_bytes_baseline": 1000,
            "max_memory_bytes_candidate": 1100,
            "max_memory_bytes_delta_pct": 10.0,
            "median_tokens_baseline": None,
            "median_tokens_candidate": None,
            "median_tokens_delta_pct": None,
            "min_tokens_baseline": None,
            "min_tokens_candidate": None,
            "min_tokens_delta_pct": None,
            "median_ttft_us_baseline": 50000,
            "median_ttft_us_candidate": 55000,
            "median_ttft_us_delta_pct": 10.0,
            "p95_ttft_us_baseline": 50000.0,
            "p95_ttft_us_candidate": 55000.0,
            "p95_ttft_us_delta_pct": 10.0,
            "median_host_overhead_pct_baseline": None,
            "median_host_overhead_pct_candidate": None,
            "median_host_overhead_pct_delta_pct": None,
        }
    ]
    assert "## Comparisons" in perf_regression.markdown_report(report)
    assert "| qemu_prompt/ci/synthetic/Q4_0/short | base | head | 5.00% |" in (
        perf_regression.markdown_report(report)
    )


def test_prompt_suite_drift_fails_comparable_perf_key() -> None:
    records = [
        perf_regression.PerfRecord(
            source="base.json",
            commit="base",
            timestamp="2026-04-28T00:00:00Z",
            benchmark="qemu_prompt",
            profile="ci",
            model="synthetic",
            quantization="Q4_0",
            prompt="short",
            tok_per_s=100.0,
            wall_tok_per_s=90.0,
            memory_bytes=1024,
            prompt_suite_sha256="a" * 64,
        ),
        perf_regression.PerfRecord(
            source="head.json",
            commit="head",
            timestamp="2026-04-28T00:01:00Z",
            benchmark="qemu_prompt",
            profile="ci",
            model="synthetic",
            quantization="Q4_0",
            prompt="short",
            tok_per_s=101.0,
            wall_tok_per_s=91.0,
            memory_bytes=1024,
            prompt_suite_sha256="b" * 64,
        ),
    ]

    report = perf_regression.build_report(records, 5.0, 10.0)

    assert report["status"] == "fail"
    assert report["prompt_suite_drift_violations"] == [
        {
            "key": "qemu_prompt/ci/synthetic/Q4_0/short",
            "hashes": ["a" * 64, "b" * 64],
            "commits": ["base", "head"],
            "sources": ["base.json", "head.json"],
        }
    ]
    assert "Prompt-suite drift violations: 1" in perf_regression.markdown_report(report)

    junit_root = ET.fromstring(perf_regression.junit_report(report))
    assert junit_root.attrib["failures"] == "1"
    failure = junit_root.find("./testcase/failure")
    assert failure is not None
    assert failure.attrib["type"] == "prompt_suite_drift"


def test_load_records_extracts_nested_prompt_suite_hash(tmp_path: Path) -> None:
    result = tmp_path / "qemu_prompt_bench_latest.json"
    result.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-28T00:00:00Z",
                "prompt_suite": {"suite_sha256": "c" * 64},
                "benchmarks": [
                    {
                        "commit": "head",
                        "benchmark": "qemu_prompt",
                        "profile": "ci",
                        "model": "synthetic",
                        "quantization": "Q4_0",
                        "prompt": "short",
                        "tok_per_s": 100.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    records = perf_regression.load_records([result])

    assert len(records) == 1
    assert records[0].prompt_suite_sha256 == "c" * 64


def test_cli_writes_comparison_coverage_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "perf.jsonl"
    output_dir = tmp_path / "dashboards"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "commit": "head",
                        "timestamp": "2026-04-28T00:01:00Z",
                        "benchmark": "qemu_prompt",
                        "profile": "ci",
                        "model": "synthetic",
                        "quantization": "Q4_0",
                        "prompt": "short",
                        "tok_per_s": 100.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status = perf_regression.main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--baseline-commit",
            "base",
            "--candidate-commit",
            "head",
            "--fail-on-regression",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "perf_regression_latest.json").read_text(encoding="utf-8"))
    csv_text = (
        output_dir / "perf_regression_comparison_coverage_violations_latest.csv"
    ).read_text(encoding="utf-8")
    assert payload["comparison_coverage_violations"][0]["missing_commits"] == "baseline:base"
    assert "key,baseline_commit,candidate_commit,missing_commits" in csv_text


def test_min_records_per_point_flags_under_sampled_commit_points(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T11:00:00Z",
                "commit": "base",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 101.0,
            },
            {
                "timestamp": "2026-04-27T12:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "tok_per_s": 102.0,
            },
        ],
    )

    records = perf_regression.load_records([result])
    report = perf_regression.build_report(records, 5.0, 10.0, min_records_per_point=2)

    assert report["status"] == "fail"
    assert report["commit_points"][0]["p05_tok_per_s"] == 100.05
    assert report["thresholds"]["min_records_per_point"] == 2
    assert report["sample_violations"] == [
        {
            "key": "decode/secure-local/-/Q4_0/-",
            "commit": "head",
            "records": 1,
            "minimum_records": 2,
        }
    ]


def test_max_tok_cv_pct_flags_noisy_commit_points(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "prompt": "short",
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T10:01:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "prompt": "short",
                "tok_per_s": 140.0,
            },
            {
                "timestamp": "2026-04-27T10:02:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "prompt": "short",
                "tok_per_s": 80.0,
            },
        ],
    )

    records = perf_regression.load_records([result])
    report = perf_regression.build_report(records, 5.0, 10.0, max_tok_cv_pct=10.0)

    assert report["status"] == "fail"
    assert report["thresholds"]["max_tok_cv_pct"] == 10.0
    assert len(report["variability_violations"]) == 1
    assert report["variability_violations"][0]["key"] == "decode/secure-local/-/Q4_0/short"
    assert report["variability_violations"][0]["tok_per_s_cv_pct"] > 10.0


def test_min_commits_per_key_flags_missing_baselines(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "prompt": "short",
                "tok_per_s": 100.0,
            },
            {
                "timestamp": "2026-04-27T10:01:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "prompt": "short",
                "tok_per_s": 101.0,
            },
        ],
    )

    records = perf_regression.load_records([result])
    report = perf_regression.build_report(records, 5.0, 10.0, min_commits_per_key=2)

    assert report["status"] == "fail"
    assert report["thresholds"]["min_commits_per_key"] == 2
    assert report["commit_coverage_violations"] == [
        {
            "key": "decode/secure-local/-/Q4_0/short",
            "commits": 1,
            "minimum_commits": 2,
            "latest_commit": "head",
        }
    ]


def test_required_telemetry_flags_missing_commit_point_metrics(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "prompt": "short",
                "tok_per_s": 100.0,
            }
        ],
    )

    records = perf_regression.load_records([result])
    report = perf_regression.build_report(
        records,
        5.0,
        10.0,
        require_tok_per_s=True,
        require_wall_tok_per_s=True,
        require_memory=True,
        require_tokens=True,
        require_ttft_us=True,
    )

    assert report["status"] == "fail"
    assert report["thresholds"]["require_tok_per_s"] is True
    assert report["thresholds"]["require_wall_tok_per_s"] is True
    assert report["thresholds"]["require_memory"] is True
    assert report["thresholds"]["require_tokens"] is True
    assert report["thresholds"]["require_ttft_us"] is True
    assert report["commit_points"][0]["tok_per_s_records"] == 1
    assert report["commit_points"][0]["wall_tok_per_s_records"] == 0
    assert report["commit_points"][0]["memory_records"] == 0
    assert report["commit_points"][0]["token_records"] == 0
    assert report["commit_points"][0]["ttft_us_records"] == 0
    assert report["telemetry_coverage_violations"] == [
        {
            "key": "decode/secure-local/-/Q4_0/short",
            "commit": "head",
            "metric": "wall_tok_per_s",
            "records": 1,
            "present_records": 0,
        },
        {
            "key": "decode/secure-local/-/Q4_0/short",
            "commit": "head",
            "metric": "memory_bytes",
            "records": 1,
            "present_records": 0,
        },
        {
            "key": "decode/secure-local/-/Q4_0/short",
            "commit": "head",
            "metric": "tokens",
            "records": 1,
            "present_records": 0,
        },
        {
            "key": "decode/secure-local/-/Q4_0/short",
            "commit": "head",
            "metric": "ttft_us",
            "records": 1,
            "present_records": 0,
        },
    ]
    assert "Telemetry coverage violations: 4" in perf_regression.markdown_report(report)

    junit_root = ET.fromstring(perf_regression.junit_report(report))
    assert junit_root.attrib["failures"] == "4"
    assert [failure.attrib["type"] for failure in junit_root.findall(".//failure")] == [
        "telemetry_coverage",
        "telemetry_coverage",
        "telemetry_coverage",
        "telemetry_coverage",
    ]


def test_required_environment_flags_missing_commit_point_metadata(tmp_path: Path) -> None:
    result = tmp_path / "perf.jsonl"
    write_jsonl(
        result,
        [
            {
                "timestamp": "2026-04-27T10:00:00Z",
                "commit": "head",
                "benchmark": "decode",
                "profile": "secure-local",
                "quantization": "Q4_0",
                "prompt": "short",
                "tok_per_s": 100.0,
                "environment": {
                    "platform": "macOS-15.4-arm64",
                    "machine": "arm64",
                },
            }
        ],
    )

    records = perf_regression.load_records([result])
    report = perf_regression.build_report(
        records,
        5.0,
        10.0,
        require_environment_sha256=True,
        require_host_platform=True,
        require_host_machine=True,
        require_qemu_version=True,
        require_qemu_bin=True,
    )

    assert report["status"] == "fail"
    assert report["thresholds"]["require_environment_sha256"] is True
    assert report["thresholds"]["require_qemu_bin"] is True
    assert report["commit_points"][0]["environment_sha256"]
    assert report["environment_coverage_violations"] == [
        {
            "key": "decode/secure-local/-/Q4_0/short",
            "commit": "head",
            "field": "qemu_version",
            "records": 1,
            "present_records": 0,
        },
        {
            "key": "decode/secure-local/-/Q4_0/short",
            "commit": "head",
            "field": "qemu_bin",
            "records": 1,
            "present_records": 0,
        },
    ]
    assert "Environment coverage violations: 2" in perf_regression.markdown_report(report)

    junit_root = ET.fromstring(perf_regression.junit_report(report))
    assert junit_root.attrib["failures"] == "2"
    assert [failure.attrib["type"] for failure in junit_root.findall(".//failure")] == [
        "environment_coverage",
        "environment_coverage",
    ]


def test_junit_report_marks_perf_failures() -> None:
    report = {
        "generated_at": "2026-04-27T20:00:00Z",
        "regressions": [
            {
                "key": "decode/secure-local/-/Q4_0/-",
                "metric": "tok_per_s",
                "baseline_commit": "base",
                "candidate_commit": "head",
                "baseline_value": 100.0,
                "candidate_value": 90.0,
                "delta_pct": 10.0,
                "threshold_pct": 5.0,
            }
        ],
        "sample_violations": [
            {
                "key": "decode/secure-local/-/Q4_0/-",
                "commit": "head",
                "records": 1,
                "minimum_records": 3,
            }
        ],
        "variability_violations": [
            {
                "key": "decode/secure-local/-/Q4_0/-",
                "commit": "head",
                "records": 3,
                "tok_per_s_cv_pct": 14.5,
                "threshold_pct": 10.0,
            }
        ],
        "commit_coverage_violations": [
            {
                "key": "decode/secure-local/-/Q4_0/-",
                "commits": 1,
                "minimum_commits": 2,
                "latest_commit": "head",
            }
        ],
    }

    root = ET.fromstring(perf_regression.junit_report(report))
    failures = root.findall(".//failure")

    assert root.attrib["tests"] == "4"
    assert root.attrib["failures"] == "4"
    assert failures[0].attrib["type"] == "perf_regression"
    assert "tok_per_s changed 10.00%" in failures[0].attrib["message"]
    assert failures[1].attrib["type"] == "sample_coverage"
    assert failures[2].attrib["type"] == "tok_per_s_variability"
    assert failures[3].attrib["type"] == "commit_coverage"


def test_write_dashboard_outputs_includes_junit(tmp_path: Path) -> None:
    report = perf_regression.build_report([], 5.0, 10.0)

    perf_regression.write_dashboard_outputs(report, tmp_path)

    root = ET.parse(tmp_path / "perf_regression_junit_latest.xml").getroot()
    assert root.attrib["name"] == "holyc_perf_regression"
    assert root.attrib["failures"] == "0"
    assert root.find("./testcase").attrib["name"] == "dashboard_pass"


def test_csv_tok_per_s_milli_is_normalized(tmp_path: Path) -> None:
    result = tmp_path / "perf.csv"
    result.write_text(
        "profile,iter,tokens,prompt_tokens,elapsed_us,tok_per_s_milli,wall_tok_per_s_milli,hardening\n"
        "secure-local,1,256,128,1000,250000,200000,attestation=on\n",
        encoding="utf-8",
    )

    records = perf_regression.load_records([result])

    assert len(records) == 1
    assert records[0].tok_per_s == 250.0
    assert records[0].wall_tok_per_s == 200.0
    assert records[0].benchmark == "perf"


def test_cli_writes_dashboard_files(tmp_path: Path) -> None:
    result = tmp_path / "perf.json"
    output_dir = tmp_path / "dashboards"
    result.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T10:00:00Z",
                "results": [
                    {
                        "commit": "abc",
                        "benchmark": "prompt",
                        "profile": "dev-local",
                        "tok_per_s": 42.0,
                        "ttft_us": 12000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status = perf_regression.main(["--input", str(result), "--output-dir", str(output_dir)])

    assert status == 0
    assert (output_dir / "perf_regression_latest.json").exists()
    assert (output_dir / "perf_regression_commit_points_latest.csv").exists()
    assert (output_dir / "perf_regression_regressions_latest.csv").exists()
    assert (output_dir / "perf_regression_comparisons_latest.csv").exists()
    assert (output_dir / "perf_regression_sample_violations_latest.csv").exists()
    assert (output_dir / "perf_regression_variability_violations_latest.csv").exists()
    assert (output_dir / "perf_regression_commit_coverage_violations_latest.csv").exists()
    assert (
        output_dir / "perf_regression_comparison_coverage_violations_latest.csv"
    ).exists()
    assert (output_dir / "perf_regression_prompt_suite_drift_latest.csv").exists()
    assert (
        output_dir / "perf_regression_environment_coverage_violations_latest.csv"
    ).exists()
    markdown = (output_dir / "perf_regression_latest.md").read_text(encoding="utf-8")
    assert "Perf Regression Dashboard" in markdown
    assert "Commit Points" in markdown
    assert "Sample Coverage" in markdown
    assert "Variability" in markdown
    assert "Commit Coverage" in markdown
    assert "Comparison Coverage" in markdown
    assert "Comparisons" in markdown
    assert "Prompt Suite Drift" in markdown
    assert "Environment Coverage" in markdown
    assert "Telemetry Coverage" in markdown
    assert "prompt/dev-local/-/-/-" in markdown
    commit_points_csv = (output_dir / "perf_regression_commit_points_latest.csv").read_text(
        encoding="utf-8"
    )
    regressions_csv = (output_dir / "perf_regression_regressions_latest.csv").read_text(
        encoding="utf-8"
    )
    comparisons_csv = (output_dir / "perf_regression_comparisons_latest.csv").read_text(
        encoding="utf-8"
    )
    sample_violations_csv = (output_dir / "perf_regression_sample_violations_latest.csv").read_text(
        encoding="utf-8"
    )
    variability_violations_csv = (
        output_dir / "perf_regression_variability_violations_latest.csv"
    ).read_text(encoding="utf-8")
    commit_coverage_violations_csv = (
        output_dir / "perf_regression_commit_coverage_violations_latest.csv"
    ).read_text(encoding="utf-8")
    comparison_coverage_violations_csv = (
        output_dir / "perf_regression_comparison_coverage_violations_latest.csv"
    ).read_text(encoding="utf-8")
    prompt_suite_drift_csv = (
        output_dir / "perf_regression_prompt_suite_drift_latest.csv"
    ).read_text(encoding="utf-8")
    environment_coverage_csv = (
        output_dir / "perf_regression_environment_coverage_violations_latest.csv"
    ).read_text(encoding="utf-8")
    telemetry_coverage_csv = (
        output_dir / "perf_regression_telemetry_coverage_violations_latest.csv"
    ).read_text(encoding="utf-8")
    assert (
        "key,commit,latest_timestamp,records,tok_per_s_records,wall_tok_per_s_records,us_per_token_records,wall_us_per_token_records,memory_records,token_records,ttft_us_records"
        in commit_points_csv
    )
    assert "prompt/dev-local/-/-/-,abc,2026-04-27T10:00:00Z,1,1,0,0,0,0,0,1,0,42.0,42.0" in commit_points_csv
    assert "key,metric,baseline_commit,candidate_commit,baseline_value,candidate_value" in regressions_csv
    assert "key,baseline_commit,candidate_commit,baseline_latest_timestamp" in comparisons_csv
    assert "key,commit,records,minimum_records" in sample_violations_csv
    assert "key,commit,records,tok_per_s_cv_pct,threshold_pct" in variability_violations_csv
    assert "key,commits,minimum_commits,latest_commit" in commit_coverage_violations_csv
    assert "key,baseline_commit,candidate_commit,missing_commits" in comparison_coverage_violations_csv
    assert "key,hashes,commits,sources" in prompt_suite_drift_csv
    assert "key,commit,field,records,present_records" in environment_coverage_csv
    assert "key,commit,metric,records,present_records" in telemetry_coverage_csv
