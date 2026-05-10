#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for perf_slo_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import perf_slo_audit


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-perf-slo-smoke-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "qemu_prompt_bench_latest.json"
        output_dir = tmp_path / "dashboards"
        input_path.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        {
                            "prompt": "short",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic",
                            "quantization": "Q4_0",
                            "commit": "abc123",
                            "phase": "measured",
                            "exit_class": "ok",
                            "returncode": 0,
                            "timed_out": False,
                            "tok_per_s": 160.0,
                            "wall_tok_per_s": 140.0,
                            "us_per_token": 6250.0,
                            "wall_us_per_token": 7142.857,
                            "ttft_us": 10000,
                            "memory_bytes": 67174400,
                            "memory_bytes_per_token": 2099200.0,
                            "host_child_peak_rss_bytes": 512000,
                        }
                    ],
                    "warmups": [
                        {
                            "prompt": "warmup",
                            "phase": "warmup",
                            "exit_class": "timeout",
                            "tok_per_s": 1.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        pass_status = perf_slo_audit.main(
            [
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "perf_slo_audit_smoke_latest",
                "--min-rows",
                "1",
                "--require-success",
                "--max-failure-pct",
                "0",
                "--min-tok-per-s",
                "100",
                "--min-wall-tok-per-s",
                "100",
                "--max-us-per-token",
                "7000",
                "--max-wall-us-per-token",
                "8000",
                "--max-ttft-us",
                "20000",
                "--max-memory-bytes",
                "80000000",
                "--max-memory-bytes-per-token",
                "3000000",
                "--max-host-child-peak-rss-bytes",
                "1000000",
            ]
        )
        if rc := require(pass_status == 0, "unexpected_pass_status"):
            return rc

        report = json.loads((output_dir / "perf_slo_audit_smoke_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_report_status"):
            return rc
        if rc := require(report["summary"]["rows"] == 1, "warmup_row_not_filtered"):
            return rc
        if rc := require(report["summary"]["metrics"]["tok_per_s"]["min"] == 160.0, "missing_metric_summary"):
            return rc
        if rc := require("No findings." in (output_dir / "perf_slo_audit_smoke_latest.md").read_text(encoding="utf-8"), "missing_markdown_pass"):
            return rc
        if rc := require(
            "source,row,severity,kind,prompt,metric,value,threshold,detail"
            in (output_dir / "perf_slo_audit_smoke_latest.csv").read_text(encoding="utf-8"),
            "missing_csv_header",
        ):
            return rc
        junit_root = ET.parse(output_dir / "perf_slo_audit_smoke_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_perf_slo_audit", "missing_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        fail_status = perf_slo_audit.main(
            [
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "perf_slo_audit_fail_latest",
                "--require-success",
                "--min-tok-per-s",
                "200",
                "--max-memory-bytes",
                "1024",
            ]
        )
        if rc := require(fail_status == 1, "unexpected_fail_status"):
            return rc
        failed = json.loads((output_dir / "perf_slo_audit_fail_latest.json").read_text(encoding="utf-8"))
        if rc := require(failed["status"] == "fail", "missing_fail_report_status"):
            return rc
        if rc := require(
            {finding["metric"] for finding in failed["findings"]} == {"tok_per_s", "memory_bytes"},
            "missing_expected_findings",
        ):
            return rc

    print("perf_slo_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
