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
        commit_points_path = output_dir / "perf_regression_commit_points_latest.csv"
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
        if fixture_summary.get("p05_tok_per_s") != 100.05:
            print("missing_p05_tok_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("median_ttft_us") != 49500.0:
            print("missing_ttft_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("p95_ttft_us") != 49950.0:
            print("missing_p95_ttft_summary=true", file=sys.stderr)
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
        if (
            "key,commit,latest_timestamp,records,tok_per_s_records,wall_tok_per_s_records,memory_records,ttft_us_records,p05_tok_per_s,median_tok_per_s,median_wall_tok_per_s,median_ttft_us,p95_ttft_us"
            not in commit_points_path.read_text(encoding="utf-8")
        ):
            print("missing_commit_points_csv=true", file=sys.stderr)
            return 1

        ttft_regression_fixture = tmp_path / "p95_ttft_regression.jsonl"
        ttft_regression_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "ttft-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-ttft",
                            "tok_per_s": 100.0,
                            "ttft_us": 100000,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:05:00Z",
                            "commit": "ttft-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-ttft",
                            "tok_per_s": 100.0,
                            "ttft_us": 125000,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        ttft_regression_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(ttft_regression_fixture),
            "--output-dir",
            str(tmp_path / "ttft_regression_dashboard"),
            "--p95-ttft-regression-pct",
            "10",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            ttft_regression_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("p95_ttft_regression_not_rejected=true", file=sys.stderr)
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

        source_audit_fixture = tmp_path / "qemu_source_audit.md"
        source_audit_fixture.write_text(
            "\n".join(
                [
                    "```bash",
                    "qemu-system-x86_64 \\",
                    "  -nic none \\",
                    "  -m 512M \\",
                    "  -drive file=/tmp/TempleOS.img,format=raw,if=ide",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        source_audit_output = tmp_path / "qemu_source_audit.json"
        source_audit_markdown = tmp_path / "qemu_source_audit.md.out"
        source_audit_csv = tmp_path / "qemu_source_audit.csv"
        source_audit_junit = tmp_path / "qemu_source_audit.xml"
        source_audit_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_source_audit.py"),
            "--input",
            str(source_audit_fixture),
            "--output",
            str(source_audit_output),
            "--markdown",
            str(source_audit_markdown),
            "--csv",
            str(source_audit_csv),
            "--junit",
            str(source_audit_junit),
        ]
        completed = subprocess.run(
            source_audit_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        source_audit_report = json.loads(source_audit_output.read_text(encoding="utf-8"))
        if source_audit_report["status"] != "pass":
            print(
                f"unexpected_qemu_source_audit_status={source_audit_report['status']}",
                file=sys.stderr,
            )
            return 1
        if source_audit_report["commands_checked"] != 1:
            print("missing_qemu_source_command_audit=true", file=sys.stderr)
            return 1
        if "QEMU Source Air-Gap Audit" not in source_audit_markdown.read_text(encoding="utf-8"):
            print("missing_qemu_source_audit_markdown=true", file=sys.stderr)
            return 1
        if "source,line,reason,command,text" not in source_audit_csv.read_text(encoding="utf-8"):
            print("missing_qemu_source_audit_csv=true", file=sys.stderr)
            return 1
        source_audit_junit_root = ET.parse(source_audit_junit).getroot()
        if source_audit_junit_root.attrib.get("name") != "holyc_qemu_source_airgap_audit":
            print("missing_qemu_source_audit_junit_suite=true", file=sys.stderr)
            return 1
        if source_audit_junit_root.attrib.get("failures") != "0":
            print("unexpected_qemu_source_audit_junit_failures=true", file=sys.stderr)
            return 1

        unsafe_source_fixture = tmp_path / "unsafe_qemu_source.md"
        unsafe_source_fixture.write_text(
            "qemu-system-x86_64 -m 512M -drive file=/tmp/TempleOS.img,format=raw,if=ide\n",
            encoding="utf-8",
        )
        unsafe_source_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_source_audit.py"),
            "--input",
            str(unsafe_source_fixture),
            "--output",
            str(tmp_path / "unsafe_qemu_source_audit.json"),
        ]
        completed = subprocess.run(
            unsafe_source_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("unsafe_qemu_source_command_not_rejected=true", file=sys.stderr)
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
            "--min-wall-tok-per-s",
            "1",
            "--max-memory-bytes",
            "100000000",
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
        telemetry_gates = bench_report["telemetry_gates"]
        if telemetry_gates.get("min_wall_tok_per_s") != 1.0:
            print("missing_min_wall_tok_gate=true", file=sys.stderr)
            return 1
        if telemetry_gates.get("max_memory_bytes") != 100000000:
            print("missing_max_memory_gate=true", file=sys.stderr)
            return 1
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
        if bench_report.get("telemetry_findings"):
            print("unexpected_bench_telemetry_findings=true", file=sys.stderr)
            return 1

        bench_gate_fail_dir = tmp_path / "qemu_prompt_bench_gate_fail"
        bench_gate_fail_command = [
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
            "--min-wall-tok-per-s",
            "1000000",
            "--max-memory-bytes",
            "1",
            "--output-dir",
            str(bench_gate_fail_dir),
        ]
        completed = subprocess.run(
            bench_gate_fail_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("bench_telemetry_gates_did_not_fail=true", file=sys.stderr)
            return 1
        bench_gate_fail_report = json.loads(
            (bench_gate_fail_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8")
        )
        gate_metrics = {finding.get("metric") for finding in bench_gate_fail_report["telemetry_findings"]}
        if not {"wall_tok_per_s", "memory_bytes"}.issubset(gate_metrics):
            print("missing_bench_gate_failure_metrics=true", file=sys.stderr)
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

        manifest_fixture_dir = tmp_path / "manifest_fixture"
        manifest_fixture_dir.mkdir()
        incomplete_report = manifest_fixture_dir / "qemu_prompt_bench_incomplete.json"
        incomplete_report.write_text(
            json.dumps(
                {
                    "generated_at": "2026-04-28T00:00:00Z",
                    "status": "pass",
                    "prompt_suite": {
                        "prompt_count": 1,
                        "suite_sha256": "manifest-smoke-suite",
                    },
                    "benchmarks": [
                        {
                            "benchmark": "qemu_prompt",
                            "profile": "manifest-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "commit": "manifest-smoke-commit",
                            "command": [
                                "qemu-system-x86_64",
                                "-nic",
                                "none",
                                "-drive",
                                "file=/tmp/TempleOS.img,format=raw,if=ide",
                            ],
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_output_dir = tmp_path / "manifest_output"
        manifest_airgap_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(manifest_fixture_dir),
            "--output-dir",
            str(manifest_output_dir),
            "--fail-on-airgap",
        ]
        completed = subprocess.run(
            manifest_airgap_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            print("manifest_airgap_gate_failed_on_non_airgap_issue=true", file=sys.stderr)
            return completed.returncode

        manifest_telemetry_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(manifest_fixture_dir),
            "--output-dir",
            str(manifest_output_dir),
            "--fail-on-telemetry",
        ]
        completed = subprocess.run(
            manifest_telemetry_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("manifest_telemetry_gate_did_not_fail=true", file=sys.stderr)
            return 1

        manifest_commit_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(manifest_fixture_dir),
            "--output-dir",
            str(manifest_output_dir),
            "--fail-on-commit-metadata",
        ]
        completed = subprocess.run(
            manifest_commit_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            print("manifest_commit_gate_failed_on_stale_only=true", file=sys.stderr)
            return completed.returncode

        complete_manifest_fixture_dir = tmp_path / "complete_manifest_fixture"
        complete_manifest_fixture_dir.mkdir()
        complete_report = complete_manifest_fixture_dir / "qemu_prompt_bench_complete.json"
        complete_payload = json.loads(incomplete_report.read_text(encoding="utf-8"))
        complete_payload["suite_summary"] = {
            "prompts": 1,
            "tok_per_s_median": 88.0,
            "memory_bytes_max": 123456,
        }
        complete_payload["benchmarks"][0]["tokens"] = 8
        complete_payload["benchmarks"][0]["elapsed_us"] = 90909
        complete_report.write_text(json.dumps(complete_payload) + "\n", encoding="utf-8")

        manifest_fresh_output_dir = tmp_path / "manifest_fresh_output"
        manifest_fresh_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(complete_manifest_fixture_dir),
            "--output-dir",
            str(manifest_fresh_output_dir),
            "--max-artifact-age-hours",
            "1000000",
            "--fail-on-stale-artifact",
        ]
        completed = subprocess.run(
            manifest_fresh_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            print("manifest_fresh_gate_failed=true", file=sys.stderr)
            return completed.returncode
        manifest_report = json.loads(
            (manifest_fresh_output_dir / "bench_artifact_manifest_latest.json").read_text(
                encoding="utf-8"
            )
        )
        latest_artifact = manifest_report["latest_artifacts"][0]
        if latest_artifact.get("freshness_status") != "pass":
            print("missing_manifest_freshness_pass=true", file=sys.stderr)
            return 1
        if latest_artifact.get("generated_age_seconds") is None:
            print("missing_manifest_artifact_age=true", file=sys.stderr)
            return 1
        if "Freshness" not in (
            manifest_fresh_output_dir / "bench_artifact_manifest_latest.md"
        ).read_text(encoding="utf-8"):
            print("missing_manifest_freshness_markdown=true", file=sys.stderr)
            return 1
        if "freshness_status" not in (
            manifest_fresh_output_dir / "bench_artifact_manifest_latest.csv"
        ).read_text(encoding="utf-8"):
            print("missing_manifest_freshness_csv=true", file=sys.stderr)
            return 1
        manifest_junit_root = ET.parse(
            manifest_fresh_output_dir / "bench_artifact_manifest_junit_latest.xml"
        ).getroot()
        if manifest_junit_root.attrib.get("name") != "holyc_bench_artifact_manifest":
            print("missing_manifest_junit_suite=true", file=sys.stderr)
            return 1
        if manifest_junit_root.attrib.get("failures") != "0":
            print("unexpected_manifest_junit_failures=true", file=sys.stderr)
            return 1

        stale_manifest_fixture_dir = tmp_path / "stale_manifest_fixture"
        stale_manifest_fixture_dir.mkdir()
        stale_report = stale_manifest_fixture_dir / "qemu_prompt_bench_stale.json"
        stale_payload = json.loads(complete_report.read_text(encoding="utf-8"))
        stale_payload["generated_at"] = "2000-01-01T00:00:00Z"
        stale_report.write_text(json.dumps(stale_payload) + "\n", encoding="utf-8")
        stale_manifest_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(stale_manifest_fixture_dir),
            "--output-dir",
            str(tmp_path / "stale_manifest_output"),
            "--max-artifact-age-hours",
            "1",
            "--fail-on-stale-artifact",
        ]
        completed = subprocess.run(
            stale_manifest_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("manifest_stale_gate_did_not_fail=true", file=sys.stderr)
            return 1

    print("perf_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
