#!/usr/bin/env python3
"""Stdlib-only smoke gate for qemu_prompt_length_bucket_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"generated_at": "2026-05-01T00:00:00Z", "benchmarks": rows}) + "\n", encoding="utf-8")


def run_audit(artifact: Path, output_dir: Path, stem: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_length_bucket_audit.py"),
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            stem,
            "--require-buckets",
            "--min-successful-samples-per-bucket",
            "1",
            "--min-prompts-per-bucket",
            "1",
            "--max-failure-pct",
            "25",
            *extra,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-length-bucket-audit-") as tmp:
        tmp_path = Path(tmp)
        artifact = tmp_path / "qemu_prompt_bench_latest.json"
        output_dir = tmp_path / "out"
        rows = [
            {
                "profile": "ci",
                "model": "synthetic",
                "quantization": "Q4_0",
                "prompt": "short",
                "prompt_sha256": "s",
                "phase": "measured",
                "exit_class": "ok",
                "prompt_bytes": 32,
                "tokens": 16,
                "wall_tok_per_s": 120.0,
                "ttft_us": 1000,
            },
            {
                "profile": "ci",
                "model": "synthetic",
                "quantization": "Q4_0",
                "prompt": "medium",
                "prompt_sha256": "m",
                "phase": "measured",
                "exit_class": "ok",
                "prompt_bytes": 512,
                "tokens": 32,
                "wall_tok_per_s": 100.0,
                "ttft_us": 2000,
            },
            {
                "profile": "ci",
                "model": "synthetic",
                "quantization": "Q4_0",
                "prompt": "long",
                "prompt_sha256": "l",
                "phase": "measured",
                "exit_class": "ok",
                "prompt_bytes": 2048,
                "tokens": 64,
                "wall_tok_per_s": 80.0,
                "ttft_us": 4000,
            },
            {
                "profile": "ci",
                "model": "synthetic",
                "quantization": "Q4_0",
                "prompt": "warmup-ignored",
                "prompt_sha256": "w",
                "phase": "warmup",
                "exit_class": "timeout",
                "prompt_bytes": 2048,
            },
        ]
        write_artifact(artifact, rows)
        completed = run_audit(artifact, output_dir, "pass")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((output_dir / "pass.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "pass_report_failed"):
            return rc
        if rc := require(report["summary"]["samples"] == 3, "warmup_row_not_filtered"):
            return rc
        if rc := require(report["summary"]["nonempty_buckets"] == 3, "missing_bucket_coverage"):
            return rc
        junit = ET.parse(output_dir / "pass_junit.xml").getroot()
        if rc := require(junit.attrib["failures"] == "0", "unexpected_pass_junit_failure"):
            return rc

        failing_artifact = tmp_path / "qemu_prompt_bench_failing.json"
        write_artifact(failing_artifact, rows[:2] + [{"phase": "measured", "exit_class": "ok", "prompt": "bad"}])
        failed = run_audit(failing_artifact, output_dir, "fail")
        if rc := require(failed.returncode == 1, "failing_report_passed"):
            return rc
        failed_report = json.loads((output_dir / "fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require("empty_bucket" in kinds, "missing_empty_bucket_finding"):
            return rc
        if rc := require("missing_prompt_bytes" in kinds, "missing_prompt_bytes_finding"):
            return rc

        custom = run_audit(
            artifact,
            output_dir,
            "custom",
            "--bucket",
            "tiny:0:64",
            "--bucket",
            "rest:65:",
            "--min-successful-samples-per-bucket",
            "1",
        )
        if custom.returncode != 0:
            sys.stdout.write(custom.stdout)
            sys.stderr.write(custom.stderr)
            return custom.returncode
        custom_report = json.loads((output_dir / "custom.json").read_text(encoding="utf-8"))
        if rc := require(custom_report["summary"]["buckets"] == 2, "custom_bucket_count_mismatch"):
            return rc

    print("qemu_prompt_length_bucket_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
