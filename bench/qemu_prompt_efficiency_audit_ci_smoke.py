#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU prompt efficiency audit."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-efficiency-ci-") as tmp:
        tmp_path = Path(tmp)
        artifact = tmp_path / "qemu_prompt_bench_latest.json"
        output_dir = tmp_path / "out"
        artifact.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        {
                            "prompt": "short",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "phase": "measured",
                            "exit_class": "ok",
                            "timed_out": False,
                            "tokens": 32,
                            "prompt_bytes": 16,
                            "elapsed_us": 8000,
                            "wall_elapsed_us": 10000,
                            "tokens_per_prompt_byte": 2.0,
                            "prompt_bytes_per_s": 2000.0,
                            "wall_prompt_bytes_per_s": 1600.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_efficiency_audit.py"),
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_prompt_efficiency_audit_latest",
            "--min-rows",
            "1",
            "--min-tokens-per-prompt-byte",
            "1.0",
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        payload = json.loads((output_dir / "qemu_prompt_efficiency_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_efficiency_status"):
            return rc
        if rc := require(payload["summary"]["rows"] == 1, "unexpected_efficiency_rows"):
            return rc
        rows = list(csv.DictReader((output_dir / "qemu_prompt_efficiency_audit_latest.csv").open(encoding="utf-8")))
        if rc := require(rows[0]["tokens_per_prompt_byte"] == "2.0", "unexpected_efficiency_density"):
            return rc
        findings = list(csv.DictReader((output_dir / "qemu_prompt_efficiency_audit_latest_findings.csv").open(encoding="utf-8")))
        if rc := require(findings == [], "unexpected_efficiency_findings"):
            return rc
        junit = ET.parse(output_dir / "qemu_prompt_efficiency_audit_latest_junit.xml").getroot()
        if rc := require(junit.attrib["failures"] == "0", "unexpected_efficiency_junit_failure"):
            return rc

        bad_artifact = tmp_path / "qemu_prompt_bench_bad.json"
        bad_artifact.write_text(
            json.dumps({"benchmarks": [{"phase": "measured", "exit_class": "ok", "tokens": 4, "prompt_bytes": 8, "tokens_per_prompt_byte": 2.0}]}),
            encoding="utf-8",
        )
        bad_completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_prompt_efficiency_audit.py"),
                str(bad_artifact),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "bad",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if rc := require(bad_completed.returncode == 1, "expected_bad_efficiency_failure"):
            return rc
        bad_payload = json.loads((output_dir / "bad.json").read_text(encoding="utf-8"))
        bad_kinds = {finding["kind"] for finding in bad_payload["findings"]}
        if rc := require("tokens_per_prompt_byte_drift" in bad_kinds, "missing_bad_efficiency_drift"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
