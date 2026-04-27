#!/usr/bin/env python3
"""Tests for host-side performance regression dashboard tooling."""

from __future__ import annotations

import json
import sys
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


def test_csv_tok_per_s_milli_is_normalized(tmp_path: Path) -> None:
    result = tmp_path / "perf.csv"
    result.write_text(
        "profile,iter,tokens,prompt_tokens,elapsed_us,tok_per_s_milli,hardening\n"
        "secure-local,1,256,128,1000,250000,attestation=on\n",
        encoding="utf-8",
    )

    records = perf_regression.load_records([result])

    assert len(records) == 1
    assert records[0].tok_per_s == 250.0
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
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status = perf_regression.main(["--input", str(result), "--output-dir", str(output_dir)])

    assert status == 0
    assert (output_dir / "perf_regression_latest.json").exists()
    markdown = (output_dir / "perf_regression_latest.md").read_text(encoding="utf-8")
    assert "Perf Regression Dashboard" in markdown
    assert "Commit Points" in markdown
    assert "prompt/dev-local/-/-/-" in markdown
