#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for benchmark trend exports."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command_sha256(command: list[str]) -> str:
    encoded = json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_report(
    path: Path,
    *,
    commit: str,
    generated_at: str,
    tok_per_s: float,
    memory: int,
    command_extra: list[str] | None = None,
    launch_plan_sha256: str = "trend-smoke-launch-plan",
    qemu_version: str = "synthetic",
) -> None:
    command = [
        "qemu-system-x86_64",
        "-nic",
        "none",
        "-display",
        "none",
        "-drive",
        "file=/tmp/TempleOS.synthetic.img,format=raw,if=ide",
    ]
    if command_extra:
        command.extend(command_extra)
    command_hash = command_sha256(command)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "generated_at": generated_at,
                "command_sha256": command_hash,
                "launch_plan_sha256": launch_plan_sha256,
                "prompt_suite": {
                    "prompt_count": 1,
                    "suite_sha256": "trend-smoke-suite",
                },
                "environment": {
                    "platform": "ci-smoke",
                    "machine": "host",
                    "qemu_bin": command[0],
                    "qemu_version": qemu_version,
                },
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "command": command,
                        "command_sha256": command_hash,
                        "commit": commit,
                        "profile": "ci-airgap-smoke",
                        "model": "synthetic-smoke",
                        "quantization": "Q4_0",
                        "prompt": "trend-smoke",
                        "tokens": 32,
                        "elapsed_us": int(32 * 1_000_000 / tok_per_s),
                        "tok_per_s": tok_per_s,
                        "wall_tok_per_s": tok_per_s - 5.0,
                        "memory_bytes": memory,
                        "returncode": 0,
                        "timed_out": False,
                    }
                ],
                "summaries": [
                    {
                        "prompt": "trend-smoke",
                        "tok_per_s_median": tok_per_s,
                        "wall_tok_per_s_median": tok_per_s - 5.0,
                        "memory_bytes_max": memory,
                    }
                ],
                "suite_summary": {
                    "prompts": 1,
                    "total_tokens": 32,
                    "total_elapsed_us": int(32 * 1_000_000 / tok_per_s),
                    "tok_per_s_median": tok_per_s,
                    "wall_tok_per_s_median": tok_per_s - 5.0,
                    "memory_bytes_max": memory,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-trend-ci-") as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "results"
        output_dir = tmp_path / "dashboards"
        input_dir.mkdir()
        write_report(
            input_dir / "qemu_prompt_bench_base.json",
            commit="trend-base",
            generated_at="2026-04-27T10:00:00Z",
            tok_per_s=100.0,
            memory=64_000_000,
        )
        write_report(
            input_dir / "qemu_prompt_bench_head.json",
            commit="trend-head",
            generated_at="2026-04-27T10:05:00Z",
            tok_per_s=105.0,
            memory=65_000_000,
        )
        write_report(
            input_dir / "qemu_prompt_bench_newer.json",
            commit="trend-newer",
            generated_at="2026-04-27T10:10:00Z",
            tok_per_s=110.0,
            memory=66_000_000,
        )

        command = [
            sys.executable,
            str(ROOT / "bench" / "bench_trend_export.py"),
            "--input",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--fail-on-empty",
            "--fail-on-airgap",
            "--fail-on-telemetry",
            "--fail-on-command-drift",
            "--fail-on-launch-plan-drift",
            "--fail-on-environment-drift",
            "--min-points-per-key",
            "2",
            "--fail-on-tok-regression-pct",
            "10",
            "--fail-on-wall-tok-regression-pct",
            "10",
            "--fail-on-memory-growth-pct",
            "10",
            "--fail-on-window-tok-regression-pct",
            "10",
            "--fail-on-window-wall-tok-regression-pct",
            "10",
            "--fail-on-window-memory-growth-pct",
            "10",
            "--window-points",
            "2",
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

        report = json.loads((output_dir / "bench_trend_export_latest.json").read_text(encoding="utf-8"))
        if report["status"] != "pass":
            print(f"unexpected_status={report['status']}", file=sys.stderr)
            return 1
        if report["trend_keys"] != 1 or report["trend_points"] != 3:
            print("unexpected_trend_counts=true", file=sys.stderr)
            return 1
        latest = report["latest"][0]
        if latest["latest_commit"] != "trend-newer":
            print("latest_commit_not_head=true", file=sys.stderr)
            return 1
        if abs(latest["median_tok_per_s_delta_pct"] - (500.0 / 105.0)) > 0.000001:
            print("unexpected_tok_delta=true", file=sys.stderr)
            return 1
        if abs(latest["max_memory_delta_pct"] - (100.0 / 65.0)) > 0.000001:
            print("unexpected_memory_delta=true", file=sys.stderr)
            return 1
        windows = report["windows"]
        if len(windows) != 1:
            print("missing_window_report=true", file=sys.stderr)
            return 1
        window = windows[0]
        if window["window_points"] != 2 or window["window_start_commit"] != "trend-head":
            print("unexpected_window_span=true", file=sys.stderr)
            return 1
        if window["guest_tok_per_s_min"] != 105.0 or window["guest_tok_per_s_max"] != 110.0:
            print("unexpected_window_guest_stats=true", file=sys.stderr)
            return 1
        if window["guest_tok_per_s_median"] != 107.5:
            print("unexpected_window_guest_median=true", file=sys.stderr)
            return 1
        drift = report["drift"]
        if drift["command_sha256"] or drift["launch_plan_sha256"] or drift["environment_sha256"]:
            print("unexpected_drift=true", file=sys.stderr)
            return 1
        thresholds = report["thresholds"]
        if thresholds["fail_on_tok_regression_pct"] != 10.0:
            print("missing_regression_threshold=true", file=sys.stderr)
            return 1
        if thresholds["min_points_per_key"] != 2:
            print("missing_min_points_threshold=true", file=sys.stderr)
            return 1
        if thresholds["window_points"] != 2:
            print("missing_window_points_threshold=true", file=sys.stderr)
            return 1
        if thresholds["fail_on_window_tok_regression_pct"] != 10.0:
            print("missing_window_tok_threshold=true", file=sys.stderr)
            return 1
        if thresholds["fail_on_command_drift"] is not True:
            print("missing_command_drift_threshold=true", file=sys.stderr)
            return 1
        markdown = (output_dir / "bench_trend_export_latest.md").read_text(encoding="utf-8")
        if "Benchmark Trend Export" not in markdown:
            print("missing_markdown_title=true", file=sys.stderr)
            return 1
        if "min_points_per_key=2" not in markdown:
            print("missing_markdown_min_points=true", file=sys.stderr)
            return 1
        if "fail_on_tok_regression_pct=10.000" not in markdown:
            print("missing_markdown_threshold=true", file=sys.stderr)
            return 1
        latest_csv = (output_dir / "bench_trend_export_latest.csv").read_text(encoding="utf-8")
        if "median_tok_per_s_delta_pct" not in latest_csv:
            print("missing_latest_csv_delta=true", file=sys.stderr)
            return 1
        points_csv = (output_dir / "bench_trend_export_points_latest.csv").read_text(encoding="utf-8")
        if "trend-base" not in points_csv or "trend-head" not in points_csv or "trend-newer" not in points_csv:
            print("missing_points_csv_history=true", file=sys.stderr)
            return 1
        windows_csv = (output_dir / "bench_trend_export_windows_latest.csv").read_text(encoding="utf-8")
        if "guest_tok_per_s_median" not in windows_csv or "trend-head" not in windows_csv:
            print("missing_windows_csv_stats=true", file=sys.stderr)
            return 1
        drift_csv = (output_dir / "bench_trend_export_drift_latest.csv").read_text(encoding="utf-8")
        if "key,field,hashes,points,commits,sources" not in drift_csv:
            print("missing_drift_csv_header=true", file=sys.stderr)
            return 1
        junit_root = ET.parse(output_dir / "bench_trend_export_junit_latest.xml").getroot()
        if junit_root.attrib.get("name") != "holyc_bench_trend_export":
            print("missing_junit_suite=true", file=sys.stderr)
            return 1
        if junit_root.attrib.get("failures") != "0":
            print("unexpected_junit_failures=true", file=sys.stderr)
            return 1

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        empty_completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "bench_trend_export.py"),
                "--input",
                str(empty_dir),
                "--output-dir",
                str(tmp_path / "empty_dashboards"),
                "--fail-on-empty",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if empty_completed.returncode == 0:
            print("empty_trend_export_not_rejected=true", file=sys.stderr)
            return 1

        regression_dir = tmp_path / "regression_results"
        regression_dir.mkdir()
        write_report(
            regression_dir / "qemu_prompt_bench_base.json",
            commit="regression-base",
            generated_at="2026-04-27T11:00:00Z",
            tok_per_s=100.0,
            memory=64_000_000,
        )
        write_report(
            regression_dir / "qemu_prompt_bench_head.json",
            commit="regression-head",
            generated_at="2026-04-27T11:05:00Z",
            tok_per_s=90.0,
            memory=80_000_000,
        )
        regression_completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "bench_trend_export.py"),
                "--input",
                str(regression_dir),
                "--output-dir",
                str(tmp_path / "regression_dashboards"),
                "--fail-on-tok-regression-pct",
                "5",
                "--fail-on-wall-tok-regression-pct",
                "5",
                "--fail-on-memory-growth-pct",
                "10",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if regression_completed.returncode == 0:
            print("trend_regression_not_rejected=true", file=sys.stderr)
            return 1
        regression_report = json.loads(
            (tmp_path / "regression_dashboards" / "bench_trend_export_latest.json").read_text(
                encoding="utf-8"
            )
        )
        findings = "\n".join(regression_report["findings"])
        if "guest tok/s regressed" not in findings:
            print("missing_guest_tok_regression_finding=true", file=sys.stderr)
            return 1
        if "wall tok/s regressed" not in findings:
            print("missing_wall_tok_regression_finding=true", file=sys.stderr)
            return 1
        if "max memory grew" not in findings:
            print("missing_memory_growth_finding=true", file=sys.stderr)
            return 1

        sparse_dir = tmp_path / "sparse_results"
        sparse_dir.mkdir()
        write_report(
            sparse_dir / "qemu_prompt_bench_head.json",
            commit="sparse-head",
            generated_at="2026-04-27T12:05:00Z",
            tok_per_s=100.0,
            memory=64_000_000,
        )
        sparse_completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "bench_trend_export.py"),
                "--input",
                str(sparse_dir),
                "--output-dir",
                str(tmp_path / "sparse_dashboards"),
                "--min-points-per-key",
                "2",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if sparse_completed.returncode == 0:
            print("sparse_trend_history_not_rejected=true", file=sys.stderr)
            return 1
        sparse_report = json.loads(
            (tmp_path / "sparse_dashboards" / "bench_trend_export_latest.json").read_text(
                encoding="utf-8"
            )
        )
        sparse_findings = "\n".join(sparse_report["findings"])
        if "below minimum 2" not in sparse_findings:
            print("missing_sparse_history_finding=true", file=sys.stderr)
            return 1

        drift_dir = tmp_path / "drift_results"
        drift_dir.mkdir()
        write_report(
            drift_dir / "qemu_prompt_bench_base.json",
            commit="drift-base",
            generated_at="2026-04-27T12:30:00Z",
            tok_per_s=100.0,
            memory=64_000_000,
            launch_plan_sha256="launch-plan-a",
            qemu_version="synthetic-a",
        )
        write_report(
            drift_dir / "qemu_prompt_bench_head.json",
            commit="drift-head",
            generated_at="2026-04-27T12:35:00Z",
            tok_per_s=101.0,
            memory=64_000_000,
            command_extra=["-smp", "1"],
            launch_plan_sha256="launch-plan-b",
            qemu_version="synthetic-b",
        )
        drift_completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "bench_trend_export.py"),
                "--input",
                str(drift_dir),
                "--output-dir",
                str(tmp_path / "drift_dashboards"),
                "--fail-on-command-drift",
                "--fail-on-launch-plan-drift",
                "--fail-on-environment-drift",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if drift_completed.returncode == 0:
            print("trend_drift_not_rejected=true", file=sys.stderr)
            return 1
        drift_report = json.loads(
            (tmp_path / "drift_dashboards" / "bench_trend_export_latest.json").read_text(
                encoding="utf-8"
            )
        )
        drift_findings = "\n".join(drift_report["findings"])
        if "command_sha256 drift" not in drift_findings:
            print("missing_command_drift_finding=true", file=sys.stderr)
            return 1
        if "launch_plan_sha256 drift" not in drift_findings:
            print("missing_launch_plan_drift_finding=true", file=sys.stderr)
            return 1
        if "environment_sha256 drift" not in drift_findings:
            print("missing_environment_drift_finding=true", file=sys.stderr)
            return 1
        drift_outputs = (tmp_path / "drift_dashboards" / "bench_trend_export_drift_latest.csv").read_text(
            encoding="utf-8"
        )
        if "launch_plan_sha256" not in drift_outputs or "environment_sha256" not in drift_outputs:
            print("missing_drift_csv_rows=true", file=sys.stderr)
            return 1

        window_regression_dir = tmp_path / "window_regression_results"
        window_regression_dir.mkdir()
        write_report(
            window_regression_dir / "qemu_prompt_bench_base.json",
            commit="window-base",
            generated_at="2026-04-27T13:00:00Z",
            tok_per_s=100.0,
            memory=64_000_000,
        )
        write_report(
            window_regression_dir / "qemu_prompt_bench_mid.json",
            commit="window-mid",
            generated_at="2026-04-27T13:05:00Z",
            tok_per_s=99.0,
            memory=65_000_000,
        )
        write_report(
            window_regression_dir / "qemu_prompt_bench_head.json",
            commit="window-head",
            generated_at="2026-04-27T13:10:00Z",
            tok_per_s=94.0,
            memory=72_000_000,
        )
        window_regression_completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "bench_trend_export.py"),
                "--input",
                str(window_regression_dir),
                "--output-dir",
                str(tmp_path / "window_regression_dashboards"),
                "--window-points",
                "3",
                "--fail-on-window-tok-regression-pct",
                "5",
                "--fail-on-window-wall-tok-regression-pct",
                "5",
                "--fail-on-window-memory-growth-pct",
                "10",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if window_regression_completed.returncode == 0:
            print("window_trend_regression_not_rejected=true", file=sys.stderr)
            return 1
        window_regression_report = json.loads(
            (tmp_path / "window_regression_dashboards" / "bench_trend_export_latest.json").read_text(
                encoding="utf-8"
            )
        )
        window_findings = "\n".join(window_regression_report["findings"])
        if "recent-window guest tok/s regressed" not in window_findings:
            print("missing_window_guest_tok_regression_finding=true", file=sys.stderr)
            return 1
        if "recent-window wall tok/s regressed" not in window_findings:
            print("missing_window_wall_tok_regression_finding=true", file=sys.stderr)
            return 1
        if "recent-window max memory grew" not in window_findings:
            print("missing_window_memory_growth_finding=true", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
