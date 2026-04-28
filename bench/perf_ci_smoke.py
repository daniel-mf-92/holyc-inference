#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for host-side perf regression dashboards."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "bench" / "fixtures" / "perf_regression" / "ci_pass.jsonl"
RESULTS = ROOT / "bench" / "results"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-perf-ci-") as tmp:
        tmp_path = Path(tmp)
        output_dir = Path(tmp) / "dashboards"
        command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(RESULTS),
            "--input",
            str(FIXTURE),
            "--output-dir",
            str(output_dir),
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report_path = output_dir / "perf_regression_latest.json"
        markdown_path = output_dir / "perf_regression_latest.md"
        junit_path = output_dir / "perf_regression_junit_latest.xml"
        sample_violations_path = output_dir / "perf_regression_sample_violations_latest.csv"
        variability_violations_path = output_dir / "perf_regression_variability_violations_latest.csv"
        commit_coverage_violations_path = (
            output_dir / "perf_regression_commit_coverage_violations_latest.csv"
        )
        comparison_coverage_violations_path = (
            output_dir / "perf_regression_comparison_coverage_violations_latest.csv"
        )
        prompt_suite_drift_path = output_dir / "perf_regression_prompt_suite_drift_latest.csv"
        telemetry_coverage_path = (
            output_dir / "perf_regression_telemetry_coverage_violations_latest.csv"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "pass":
            print(f"unexpected_status={report['status']}", file=sys.stderr)
            return 1
        if report["regressions"]:
            print("unexpected_regressions=true", file=sys.stderr)
            return 1
        if report["sample_violations"]:
            print("unexpected_sample_violations=true", file=sys.stderr)
            return 1
        if report["variability_violations"]:
            print("unexpected_variability_violations=true", file=sys.stderr)
            return 1
        if report["commit_coverage_violations"]:
            print("unexpected_commit_coverage_violations=true", file=sys.stderr)
            return 1
        if report["comparison_coverage_violations"]:
            print("unexpected_comparison_coverage_violations=true", file=sys.stderr)
            return 1
        if report["prompt_suite_drift_violations"]:
            print("unexpected_prompt_suite_drift=true", file=sys.stderr)
            return 1
        if report["telemetry_coverage_violations"]:
            print("unexpected_telemetry_coverage_violations=true", file=sys.stderr)
            return 1
        if "qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short" not in report["summaries"]:
            print("missing_ci_fixture_summary=true", file=sys.stderr)
            return 1
        fixture_summary = report["summaries"][
            "qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short"
        ]
        if fixture_summary.get("median_wall_tok_per_s") != 95.5:
            print("missing_wall_tok_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("median_ttft_us") != 49500.0:
            print("missing_ttft_summary=true", file=sys.stderr)
            return 1
        if "Perf Regression Dashboard" not in markdown_path.read_text(encoding="utf-8"):
            print("missing_markdown_dashboard=true", file=sys.stderr)
            return 1
        junit_root = ET.parse(junit_path).getroot()
        if junit_root.attrib.get("name") != "holyc_perf_regression":
            print("missing_junit_suite=true", file=sys.stderr)
            return 1
        if junit_root.attrib.get("failures") != "0":
            print("unexpected_junit_failures=true", file=sys.stderr)
            return 1
        if "key,commit,records,minimum_records" not in sample_violations_path.read_text(
            encoding="utf-8"
        ):
            print("missing_sample_violations_csv=true", file=sys.stderr)
            return 1
        if "key,commit,records,tok_per_s_cv_pct,threshold_pct" not in variability_violations_path.read_text(
            encoding="utf-8"
        ):
            print("missing_variability_violations_csv=true", file=sys.stderr)
            return 1
        if "key,commits,minimum_commits,latest_commit" not in commit_coverage_violations_path.read_text(
            encoding="utf-8"
        ):
            print("missing_commit_coverage_violations_csv=true", file=sys.stderr)
            return 1
        if (
            "key,baseline_commit,candidate_commit,missing_commits"
            not in comparison_coverage_violations_path.read_text(encoding="utf-8")
        ):
            print("missing_comparison_coverage_violations_csv=true", file=sys.stderr)
            return 1
        if "key,hashes,commits,sources" not in prompt_suite_drift_path.read_text(encoding="utf-8"):
            print("missing_prompt_suite_drift_csv=true", file=sys.stderr)
            return 1
        if "key,commit,metric,records,present_records" not in telemetry_coverage_path.read_text(
            encoding="utf-8"
        ):
            print("missing_telemetry_coverage_csv=true", file=sys.stderr)
            return 1

        audit_output = tmp_path / "airgap_audit.json"
        audit_markdown = tmp_path / "airgap_audit.md"
        audit_csv = tmp_path / "airgap_audit.csv"
        audit_junit = tmp_path / "airgap_audit.xml"
        audit_command = [
            sys.executable,
            str(ROOT / "bench" / "airgap_audit.py"),
            "--input",
            str(RESULTS),
            "--output",
            str(audit_output),
            "--markdown",
            str(audit_markdown),
            "--csv",
            str(audit_csv),
            "--junit",
            str(audit_junit),
        ]
        completed = subprocess.run(
            audit_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        audit_report = json.loads(audit_output.read_text(encoding="utf-8"))
        if audit_report["status"] != "pass":
            print(f"unexpected_airgap_status={audit_report['status']}", file=sys.stderr)
            return 1
        if audit_report["commands_checked"] < 1:
            print("missing_qemu_command_audit=true", file=sys.stderr)
            return 1
        if "Benchmark Air-Gap Audit" not in audit_markdown.read_text(encoding="utf-8"):
            print("missing_airgap_markdown=true", file=sys.stderr)
            return 1
        if "source,row,reason,command" not in audit_csv.read_text(encoding="utf-8"):
            print("missing_airgap_csv=true", file=sys.stderr)
            return 1
        audit_junit_root = ET.parse(audit_junit).getroot()
        if audit_junit_root.attrib.get("name") != "holyc_benchmark_airgap_audit":
            print("missing_airgap_junit_suite=true", file=sys.stderr)
            return 1
        if audit_junit_root.attrib.get("failures") != "0":
            print("unexpected_airgap_junit_failures=true", file=sys.stderr)
            return 1

        unsafe_fixture = tmp_path / "unsafe_qemu.json"
        unsafe_fixture.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        {
                            "benchmark": "qemu_prompt",
                            "command": [
                                "qemu-system-x86_64",
                                "-drive",
                                "file=/tmp/TempleOS.img,format=raw,if=ide",
                                "-device",
                                "e1000",
                            ],
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        unsafe_command = [
            sys.executable,
            str(ROOT / "bench" / "airgap_audit.py"),
            "--input",
            str(unsafe_fixture),
            "--output",
            str(tmp_path / "unsafe_airgap_audit.json"),
        ]
        completed = subprocess.run(
            unsafe_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("unsafe_qemu_command_not_rejected=true", file=sys.stderr)
            return 1

        bench_output_dir = tmp_path / "qemu_prompt_bench"
        bench_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_bench.py"),
            "--image",
            "/tmp/TempleOS.synthetic.img",
            "--prompts",
            str(ROOT / "bench" / "prompts" / "smoke.jsonl"),
            "--qemu-bin",
            str(ROOT / "bench" / "fixtures" / "qemu_synthetic_bench.py"),
            "--profile",
            "ci-airgap-smoke",
            "--model",
            "synthetic-smoke",
            "--quantization",
            "Q4_0",
            "--repeat",
            "3",
            "--max-prompt-cv-pct",
            "0.1",
            "--output-dir",
            str(bench_output_dir),
        ]
        completed = subprocess.run(
            bench_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        bench_report = json.loads(
            (bench_output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8")
        )
        suite_summary = bench_report["suite_summary"]
        if suite_summary.get("tok_per_s_stdev") is None:
            print("missing_suite_tok_stdev=true", file=sys.stderr)
            return 1
        if suite_summary.get("tok_per_s_cv_pct") is None:
            print("missing_suite_tok_cv=true", file=sys.stderr)
            return 1
        if suite_summary.get("ttft_us_median") is None:
            print("missing_suite_ttft_median=true", file=sys.stderr)
            return 1
        if suite_summary.get("ttft_us_p95") is None:
            print("missing_suite_ttft_p95=true", file=sys.stderr)
            return 1
        if suite_summary.get("host_overhead_us_median") is None:
            print("missing_suite_host_overhead=true", file=sys.stderr)
            return 1
        if not all("tok_per_s_cv_pct" in row for row in bench_report["summaries"]):
            print("missing_prompt_tok_cv=true", file=sys.stderr)
            return 1
        if not all(
            "host_overhead_us_median" in row and "host_overhead_pct_median" in row
            for row in bench_report["summaries"]
        ):
            print("missing_prompt_host_overhead=true", file=sys.stderr)
            return 1
        if not all(
            "ttft_us_median" in row and "ttft_us_p95" in row
            for row in bench_report["summaries"]
        ):
            print("missing_prompt_ttft=true", file=sys.stderr)
            return 1
        if bench_report.get("variability_findings"):
            print("unexpected_variability_findings=true", file=sys.stderr)
            return 1

        matrix_output_dir = tmp_path / "bench_matrix"
        matrix_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_matrix.py"),
            "--matrix",
            str(ROOT / "bench" / "fixtures" / "bench_matrix_smoke.json"),
            "--output-dir",
            str(matrix_output_dir),
            "--max-suite-cv-pct",
            "0.1",
        ]
        completed = subprocess.run(
            matrix_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        matrix_report = json.loads(
            (matrix_output_dir / "bench_matrix_latest.json").read_text(encoding="utf-8")
        )
        if matrix_report["status"] != "pass":
            print(f"unexpected_matrix_status={matrix_report['status']}", file=sys.stderr)
            return 1
        if matrix_report["variability_gates"].get("max_suite_cv_pct") != 0.1:
            print("missing_matrix_suite_cv_gate=true", file=sys.stderr)
            return 1
        if matrix_report["variability_gates"].get("max_prompt_cv_pct") != 0.1:
            print("missing_matrix_prompt_cv_gate=true", file=sys.stderr)
            return 1
        if not matrix_report["cells"] or any(
            "variability_findings" not in cell for cell in matrix_report["cells"]
        ):
            print("missing_matrix_variability_findings=true", file=sys.stderr)
            return 1
        if any(cell["variability_findings"] for cell in matrix_report["cells"]):
            print("unexpected_matrix_variability_findings=true", file=sys.stderr)
            return 1
        if "Variability gates" not in (matrix_output_dir / "bench_matrix_latest.md").read_text(
            encoding="utf-8"
        ):
            print("missing_matrix_variability_markdown=true", file=sys.stderr)
            return 1
        if "variability_findings" not in (
            matrix_output_dir / "bench_matrix_latest.csv"
        ).read_text(encoding="utf-8"):
            print("missing_matrix_variability_csv=true", file=sys.stderr)
            return 1

    print("perf_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
