#!/usr/bin/env python3
"""Tests for absolute performance SLO audit tooling."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import perf_slo_audit


def test_load_rows_filters_warmups_and_flattens_benchmarks(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "profile": "ci",
                "model": "tiny",
                "quantization": "Q4_0",
                "commit": "abc",
                "warmups": [{"prompt": "warm", "phase": "warmup", "tok_per_s": 1.0}],
                "benchmarks": [{"prompt": "p0", "phase": "measured", "tok_per_s": 42.0}],
            }
        ),
        encoding="utf-8",
    )

    rows = perf_slo_audit.load_rows([report])

    assert len(rows) == 1
    assert rows[0].prompt == "p0"
    assert rows[0].profile == "ci"
    assert rows[0].metrics["tok_per_s"] == 42.0


def test_evaluate_flags_metric_slos_and_failures(tmp_path: Path) -> None:
    report = tmp_path / "bench.csv"
    report.write_text(
        "\n".join(
            [
                "prompt,exit_class,returncode,timed_out,tok_per_s,memory_bytes",
                "slow,ok,0,false,9.5,2048",
                "bad,timeout,1,true,100,512",
                "",
            ]
        ),
        encoding="utf-8",
    )
    args = perf_slo_audit.build_parser().parse_args(
        [
            str(report),
            "--require-success",
            "--min-tok-per-s",
            "10",
            "--max-memory-bytes",
            "1024",
            "--max-failure-pct",
            "0",
        ]
    )

    rows = perf_slo_audit.load_rows([report])
    findings = perf_slo_audit.evaluate(rows, args)

    assert {finding.kind for finding in findings} == {"slo_violation", "run_failure", "failure_rate"}
    assert any(finding.metric == "tok_per_s" for finding in findings)
    assert any(finding.metric == "memory_bytes" for finding in findings)
    assert any(finding.metric == "failure_pct" for finding in findings)


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "dashboards"
    report.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "prompt": "p0",
                        "phase": "measured",
                        "exit_class": "ok",
                        "returncode": 0,
                        "timed_out": False,
                        "tok_per_s": 50.0,
                        "memory_bytes": 4096,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    status = perf_slo_audit.main(
        [
            str(report),
            "--output-dir",
            str(output_dir),
            "--min-rows",
            "1",
            "--require-success",
            "--min-tok-per-s",
            "100",
            "--max-memory-bytes",
            "1024",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "perf_slo_audit_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["summary"]["findings"] == 2
    assert "tok_per_s 50.0 < 100.0" in (output_dir / "perf_slo_audit_latest.md").read_text(encoding="utf-8")
    assert "memory_bytes" in (output_dir / "perf_slo_audit_latest.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "perf_slo_audit_latest_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_perf_slo_audit"
    assert junit_root.attrib["failures"] == "2"
