#!/usr/bin/env python3
"""Tests for host-side build benchmark comparison tooling."""

from __future__ import annotations

import json
import sys
import csv
import xml.etree.ElementTree as ET
from tempfile import TemporaryDirectory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import build_compare


def write_report(
    path: Path,
    commit: str,
    tok_per_s: float,
    elapsed_us: int,
    wall_tok_per_s: float | None = None,
    ttft_us: int | None = None,
    memory_bytes: int = 4096,
    prompt_suite_sha256: str = "suite-a",
) -> None:
    row = {
        "commit": commit,
        "benchmark": "qemu_prompt",
        "profile": "secure-local",
        "model": "tiny",
        "quantization": "Q4_0",
        "prompt": "smoke",
        "tokens": 32,
        "elapsed_us": elapsed_us,
        "tok_per_s": tok_per_s,
        "memory_bytes": memory_bytes,
        "returncode": 0,
        "timed_out": False,
    }
    if wall_tok_per_s is not None:
        row["wall_tok_per_s"] = wall_tok_per_s
    if ttft_us is not None:
        row["ttft_us"] = ttft_us
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T10:00:00Z",
                "prompt_suite": {"suite_sha256": prompt_suite_sha256},
                "benchmarks": [row],
            }
        ),
        encoding="utf-8",
    )


def write_multi_report(
    path: Path,
    commit: str,
    tok_per_s_values: list[float],
    wall_tok_per_s_values: list[float] | None = None,
    elapsed_us: int = 200000,
) -> None:
    rows = []
    for index, tok_per_s in enumerate(tok_per_s_values):
        row = {
            "commit": commit,
            "benchmark": "qemu_prompt",
            "profile": "secure-local",
            "model": "tiny",
            "quantization": "Q4_0",
            "prompt": "smoke",
            "tokens": 32,
            "elapsed_us": elapsed_us + index,
            "tok_per_s": tok_per_s,
            "memory_bytes": 4096,
            "returncode": 0,
            "timed_out": False,
        }
        if wall_tok_per_s_values is not None:
            row["wall_tok_per_s"] = wall_tok_per_s_values[index]
        rows.append(row)
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T10:00:00Z",
                "prompt_suite": {"suite_sha256": "suite-a"},
                "benchmarks": rows,
            }
        ),
        encoding="utf-8",
    )


def write_metric_rows_report(path: Path, commit: str, rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-28T16:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "suite-a"},
                "benchmarks": [
                    {
                        "commit": commit,
                        "benchmark": "qemu_prompt",
                        "profile": "secure-local",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "prompt": "smoke",
                        "tokens": 32,
                        "elapsed_us": 200000,
                        "memory_bytes": 4096,
                        "returncode": 0,
                        "timed_out": False,
                        **row,
                    }
                    for row in rows
                ],
            }
        ),
        encoding="utf-8",
    )


def test_compare_builds_computes_tok_per_s_and_elapsed_deltas(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    write_report(baseline, "base", 100.0, 200000, wall_tok_per_s=80.0, ttft_us=40000)
    write_report(candidate, "head", 125.0, 160000, wall_tok_per_s=100.0, ttft_us=36000, memory_bytes=5120)

    metrics = build_compare.load_build_metrics([f"base={baseline}", f"head={candidate}"])
    deltas = build_compare.compare_builds(metrics, "base")

    assert len(metrics) == 2
    assert len(deltas) == 1
    assert deltas[0].candidate_build == "head"
    assert deltas[0].tok_per_s_delta_pct == 25.0
    assert deltas[0].baseline_tok_per_s_p05 == 100.0
    assert deltas[0].candidate_tok_per_s_p05 == 125.0
    assert deltas[0].tok_per_s_p05_delta_pct == 25.0
    assert deltas[0].wall_tok_per_s_delta_pct == 25.0
    assert deltas[0].baseline_wall_tok_per_s_p05 == 80.0
    assert deltas[0].candidate_wall_tok_per_s_p05 == 100.0
    assert deltas[0].wall_tok_per_s_p05_delta_pct == 25.0
    assert deltas[0].elapsed_delta_pct == -20.0
    assert deltas[0].baseline_ttft_us == 40000
    assert deltas[0].candidate_ttft_us == 36000
    assert deltas[0].ttft_delta_pct == -10.0
    assert deltas[0].baseline_memory_bytes == 4096
    assert deltas[0].candidate_memory_bytes == 5120
    assert deltas[0].memory_delta_pct == 25.0
    assert deltas[0].key == "qemu_prompt/secure-local/tiny/Q4_0/smoke"


def test_cli_writes_json_markdown_and_csv_reports(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_report(baseline, "base", 100.0, 200000, wall_tok_per_s=50.0, ttft_us=40000)
    write_report(candidate, "head", 90.0, 220000, wall_tok_per_s=45.0, ttft_us=44000, memory_bytes=6144)

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "build_compare_latest.md").read_text(encoding="utf-8")
    csv_rows = list(csv.DictReader((output_dir / "build_compare_latest.csv").open(newline="", encoding="utf-8")))
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()

    assert payload["status"] == "fail"
    assert payload["baseline_build"] == "base"
    assert payload["deltas"][0]["tok_per_s_delta_pct"] == -10.0
    assert payload["deltas"][0]["tok_per_s_p05_delta_pct"] == -10.0
    assert payload["deltas"][0]["wall_tok_per_s_delta_pct"] == -10.0
    assert payload["deltas"][0]["wall_tok_per_s_p05_delta_pct"] == -10.0
    assert payload["deltas"][0]["ttft_delta_pct"] == 10.0
    assert payload["deltas"][0]["memory_delta_pct"] == 50.0
    assert payload["regressions"][0]["candidate_build"] == "head"
    assert payload["regressions"][0]["metric"] == "tok_per_s"
    assert csv_rows[0]["tok_per_s_delta_pct"] == "-10.0"
    assert csv_rows[0]["tok_per_s_p05_delta_pct"] == "-10.0"
    assert csv_rows[0]["wall_tok_per_s_delta_pct"] == "-10.0"
    assert csv_rows[0]["wall_tok_per_s_p05_delta_pct"] == "-10.0"
    assert csv_rows[0]["ttft_delta_pct"] == "10.0"
    assert csv_rows[0]["memory_delta_pct"] == "50.0"
    assert junit_root.attrib["name"] == "holyc_build_compare"
    assert junit_root.attrib["tests"] == "1"
    assert junit_root.attrib["failures"] == "1"
    assert junit_root.find("./testcase/failure") is not None
    assert "Build Benchmark Compare" in markdown
    assert "Status: fail" in markdown
    assert (
        "| head | qemu_prompt/secure-local/tiny/Q4_0/smoke | 100.000 | 90.000 | -10.000 | "
        "100.000 | 90.000 | -10.000 | 50.000 | 45.000 | -10.000 | "
        "50.000 | 45.000 | -10.000 | "
        "200000.000 | 220000.000 | 10.000 | "
        "40000.000 | 44000.000 | 10.000 |"
    ) in markdown


def test_cli_can_gate_memory_growth(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_report(baseline, "base", 100.0, 200000, memory_bytes=4096)
    write_report(candidate, "head", 100.0, 200000, memory_bytes=5120)

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--max-memory-growth-pct",
            "10",
            "--fail-on-regression",
        ]
    )

    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()

    assert status == 1
    assert payload["status"] == "fail"
    assert payload["regressions"][0]["metric"] == "memory_bytes"
    assert payload["regressions"][0]["delta_pct"] == 25.0
    assert "memory_bytes changed by 25.000%" in junit_root.find("./testcase/failure").attrib["message"]


def test_cli_can_gate_wall_clock_throughput_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_report(baseline, "base", 100.0, 200000, wall_tok_per_s=80.0)
    write_report(candidate, "head", 100.0, 200000, wall_tok_per_s=70.0)

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--max-wall-tok-regression-pct",
            "5",
            "--fail-on-regression",
        ]
    )

    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()

    assert status == 1
    assert payload["status"] == "fail"
    assert payload["max_wall_tok_regression_pct"] == 5.0
    assert payload["regressions"][0]["metric"] == "wall_tok_per_s"
    assert payload["regressions"][0]["delta_pct"] == -12.5
    assert "wall_tok_per_s changed by -12.500%" in junit_root.find("./testcase/failure").attrib["message"]


def test_cli_can_gate_p05_throughput_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_multi_report(baseline, "base", [100.0, 110.0, 120.0])
    write_multi_report(candidate, "head", [80.0, 115.0, 130.0])

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--max-tok-regression-pct",
            "10",
            "--max-p05-tok-regression-pct",
            "10",
            "--fail-on-regression",
        ]
    )

    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()

    assert status == 1
    assert payload["status"] == "fail"
    assert payload["max_p05_tok_regression_pct"] == 10.0
    assert payload["deltas"][0]["tok_per_s_delta_pct"] > 0.0
    assert payload["deltas"][0]["tok_per_s_p05_delta_pct"] < -10.0
    assert payload["regressions"][0]["metric"] == "tok_per_s_p05"
    assert "tok_per_s_p05 changed by" in junit_root.find("./testcase/failure").attrib["message"]


def test_cli_can_gate_p05_wall_clock_throughput_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_multi_report(baseline, "base", [100.0, 110.0, 120.0], wall_tok_per_s_values=[90.0, 100.0, 110.0])
    write_multi_report(candidate, "head", [105.0, 115.0, 125.0], wall_tok_per_s_values=[70.0, 105.0, 115.0])

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--max-tok-regression-pct",
            "10",
            "--max-p05-wall-tok-regression-pct",
            "10",
            "--fail-on-regression",
        ]
    )

    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()

    assert status == 1
    assert payload["status"] == "fail"
    assert payload["max_p05_wall_tok_regression_pct"] == 10.0
    assert payload["deltas"][0]["wall_tok_per_s_delta_pct"] > 0.0
    assert payload["deltas"][0]["wall_tok_per_s_p05_delta_pct"] < -10.0
    assert payload["regressions"][0]["metric"] == "wall_tok_per_s_p05"
    assert "wall_tok_per_s_p05 changed by" in junit_root.find("./testcase/failure").attrib["message"]


def test_cli_can_gate_ttft_growth(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_report(baseline, "base", 100.0, 200000, ttft_us=40000)
    write_report(candidate, "head", 100.0, 200000, ttft_us=46000)

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--max-ttft-growth-pct",
            "10",
            "--fail-on-regression",
        ]
    )

    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()

    assert status == 1
    assert payload["status"] == "fail"
    assert payload["max_ttft_growth_pct"] == 10.0
    assert payload["regressions"][0]["metric"] == "ttft_us"
    assert payload["regressions"][0]["delta_pct"] == 15.0
    assert "ttft_us changed by 15.000%" in junit_root.find("./testcase/failure").attrib["message"]


def test_cli_can_gate_host_latency_cpu_and_rss_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_metric_rows_report(
        baseline,
        "base",
        [
            {
                "tok_per_s": 100.0,
                "wall_tok_per_s": 95.0,
                "us_per_token": 10000.0,
                "wall_us_per_token": 10500.0,
                "host_child_cpu_us": 500000,
                "host_child_cpu_pct": 50.0,
                "host_child_tok_per_cpu_s": 200.0,
                "host_child_peak_rss_bytes": 1000000,
            },
            {
                "tok_per_s": 102.0,
                "wall_tok_per_s": 97.0,
                "us_per_token": 9800.0,
                "wall_us_per_token": 10300.0,
                "host_child_cpu_us": 520000,
                "host_child_cpu_pct": 52.0,
                "host_child_tok_per_cpu_s": 205.0,
                "host_child_peak_rss_bytes": 1100000,
            },
        ],
    )
    write_metric_rows_report(
        candidate,
        "head",
        [
            {
                "tok_per_s": 99.0,
                "wall_tok_per_s": 93.0,
                "us_per_token": 10800.0,
                "wall_us_per_token": 11300.0,
                "host_child_cpu_us": 580000,
                "host_child_cpu_pct": 59.0,
                "host_child_tok_per_cpu_s": 175.0,
                "host_child_peak_rss_bytes": 1300000,
            },
            {
                "tok_per_s": 101.0,
                "wall_tok_per_s": 94.0,
                "us_per_token": 11000.0,
                "wall_us_per_token": 11500.0,
                "host_child_cpu_us": 600000,
                "host_child_cpu_pct": 60.0,
                "host_child_tok_per_cpu_s": 180.0,
                "host_child_peak_rss_bytes": 1400000,
            },
        ],
    )

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--max-us-per-token-growth-pct",
            "5",
            "--max-wall-us-per-token-growth-pct",
            "5",
            "--max-host-child-cpu-growth-pct",
            "10",
            "--max-host-child-cpu-pct-growth-pct",
            "10",
            "--max-host-child-tok-per-cpu-s-regression-pct",
            "10",
            "--max-host-child-rss-growth-pct",
            "20",
            "--fail-on-regression",
        ]
    )

    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader((output_dir / "build_compare_latest.csv").open(newline="", encoding="utf-8")))
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()
    markdown = (output_dir / "build_compare_latest.md").read_text(encoding="utf-8")
    delta = payload["deltas"][0]

    assert status == 1
    assert delta["baseline_us_per_token"] == 9900.0
    assert delta["candidate_us_per_token"] == 10900.0
    assert delta["baseline_wall_us_per_token"] == 10400.0
    assert delta["candidate_wall_us_per_token"] == 11400.0
    assert delta["baseline_host_child_cpu_us"] == 510000.0
    assert delta["candidate_host_child_cpu_us"] == 590000.0
    assert delta["baseline_host_child_cpu_pct"] == 51.0
    assert delta["candidate_host_child_cpu_pct"] == 59.5
    assert delta["baseline_host_child_tok_per_cpu_s"] == 202.5
    assert delta["candidate_host_child_tok_per_cpu_s"] == 177.5
    assert delta["baseline_host_child_peak_rss_bytes"] == 1100000
    assert delta["candidate_host_child_peak_rss_bytes"] == 1400000
    assert {
        "us_per_token",
        "wall_us_per_token",
        "host_child_cpu_us",
        "host_child_cpu_pct",
        "host_child_tok_per_cpu_s",
        "host_child_peak_rss_bytes",
    }.issubset({row["metric"] for row in payload["regressions"]})
    assert csv_rows[0]["candidate_wall_us_per_token"] == "11400.0"
    assert csv_rows[0]["candidate_host_child_peak_rss_bytes"] == "1400000"
    assert "Host child CPU/RSS regressions: 4" in markdown
    assert "Candidate wall us/token" in markdown
    assert junit_root.attrib["failures"] == "6"
    assert "host_child_peak_rss_delta_pct" in "".join(
        failure.text or "" for failure in junit_root.iter("failure")
    )


def test_cli_can_gate_ok_run_coverage(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_report(baseline, "base", 100.0, 200000)
    write_report(candidate, "head", 100.0, 200000)

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--min-ok-runs-per-build",
            "2",
            "--fail-on-coverage",
        ]
    )

    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    coverage_rows = list(
        csv.DictReader((output_dir / "build_compare_coverage_violations_latest.csv").open(newline="", encoding="utf-8"))
    )
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()
    failures = junit_root.findall("./testcase/failure")

    assert status == 1
    assert payload["status"] == "fail"
    assert payload["min_ok_runs_per_build"] == 2
    assert len(payload["coverage_violations"]) == 2
    assert {row["build"] for row in coverage_rows} == {"base", "head"}
    assert {row["ok_runs"] for row in coverage_rows} == {"1"}
    assert junit_root.attrib["failures"] == "2"
    assert {failure.attrib["type"] for failure in failures} == {"build_compare_sample_coverage"}
    assert "Coverage violations: 2" in (output_dir / "build_compare_latest.md").read_text(encoding="utf-8")


def test_cli_can_gate_prompt_suite_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_report(baseline, "base", 100.0, 200000, prompt_suite_sha256="suite-a")
    write_report(candidate, "head", 100.0, 200000, prompt_suite_sha256="suite-b")

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--fail-on-prompt-suite-drift",
        ]
    )

    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    drift_rows = list(
        csv.DictReader((output_dir / "build_compare_prompt_suite_drift_latest.csv").open(newline="", encoding="utf-8"))
    )
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()
    failure = junit_root.find("./testcase/failure")

    assert status == 1
    assert payload["status"] == "fail"
    assert payload["deltas"][0]["baseline_prompt_suite_sha256"] == "suite-a"
    assert payload["deltas"][0]["candidate_prompt_suite_sha256"] == "suite-b"
    assert payload["prompt_suite_drift"][0]["candidate_build"] == "head"
    assert drift_rows[0]["baseline_prompt_suite_sha256"] == "suite-a"
    assert junit_root.attrib["failures"] == "1"
    assert failure is not None
    assert failure.attrib["type"] == "build_compare_prompt_suite_drift"
    assert "Prompt-suite drift: 1" in (output_dir / "build_compare_latest.md").read_text(encoding="utf-8")


def test_cli_can_fail_on_throughput_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_report(baseline, "base", 100.0, 200000)
    write_report(candidate, "head", 96.0, 210000)

    passing_status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--fail-on-regression",
        ]
    )
    failing_status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
            "--max-tok-regression-pct",
            "3",
            "--fail-on-regression",
        ]
    )

    assert passing_status == 0
    assert failing_status == 1


def test_missing_baseline_returns_error(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    write_report(report, "head", 100.0, 200000)

    status = build_compare.main(["--input", f"head={report}", "--baseline", "missing"])

    assert status == 2


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_compare_builds_computes_tok_per_s_and_elapsed_deltas(tmp_path)
        test_cli_writes_json_markdown_and_csv_reports(tmp_path)
        test_cli_can_gate_memory_growth(tmp_path)
        test_cli_can_gate_wall_clock_throughput_regression(tmp_path)
        test_cli_can_gate_p05_throughput_regression(tmp_path)
        test_cli_can_gate_p05_wall_clock_throughput_regression(tmp_path)
        test_cli_can_gate_ttft_growth(tmp_path)
        test_cli_can_gate_host_latency_cpu_and_rss_drift(tmp_path)
        test_cli_can_gate_ok_run_coverage(tmp_path)
        test_cli_can_gate_prompt_suite_drift(tmp_path)
        test_cli_can_fail_on_throughput_regression(tmp_path)
        test_missing_baseline_returns_error(tmp_path)
    print("build_compare_tests=ok")
