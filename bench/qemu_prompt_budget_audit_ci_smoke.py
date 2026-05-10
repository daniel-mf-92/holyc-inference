#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU prompt budget audit."""

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
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-prompt-budget-ci-") as tmp:
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
                            "tokens": 32,
                            "expected_tokens": 32,
                            "expected_tokens_match": True,
                            "prompt_bytes": 16,
                            "guest_prompt_bytes": 16,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_budget_audit.py"),
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_prompt_budget_audit_latest",
            "--max-prompt-bytes",
            "32",
            "--max-tokens",
            "40",
            "--require-expected-tokens",
            "--require-guest-prompt-bytes",
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        payload = json.loads((output_dir / "qemu_prompt_budget_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_prompt_budget_status"):
            return rc
        if rc := require(payload["summary"]["max_prompt_bytes"] == 16, "unexpected_prompt_budget_bytes"):
            return rc
        rows = list(csv.DictReader((output_dir / "qemu_prompt_budget_audit_latest.csv").open(encoding="utf-8")))
        if rc := require(rows[0]["prompt"] == "short", "unexpected_prompt_budget_row"):
            return rc
        findings = list(csv.DictReader((output_dir / "qemu_prompt_budget_audit_latest_findings.csv").open(encoding="utf-8")))
        if rc := require(findings == [], "unexpected_prompt_budget_findings"):
            return rc
        junit = ET.parse(output_dir / "qemu_prompt_budget_audit_latest_junit.xml").getroot()
        if rc := require(junit.attrib["failures"] == "0", "unexpected_prompt_budget_junit_failure"):
            return rc

        bad_artifact = tmp_path / "qemu_prompt_bench_bad.json"
        bad_artifact.write_text(
            json.dumps({"benchmarks": [{"phase": "measured", "exit_class": "ok", "tokens": 41, "expected_tokens": 32, "prompt_bytes": 64, "guest_prompt_bytes": 16}]}),
            encoding="utf-8",
        )
        bad_completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_prompt_budget_audit.py"),
                str(bad_artifact),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "bad",
                "--max-prompt-bytes",
                "32",
                "--max-tokens",
                "40",
                "--require-expected-tokens",
                "--require-guest-prompt-bytes",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if rc := require(bad_completed.returncode == 1, "expected_bad_prompt_budget_failure"):
            return rc
        bad_payload = json.loads((output_dir / "bad.json").read_text(encoding="utf-8"))
        bad_kinds = {finding["kind"] for finding in bad_payload["findings"]}
        if rc := require("prompt_bytes_over_budget" in bad_kinds, "missing_bad_prompt_byte_budget"):
            return rc
        if rc := require("tokens_over_budget" in bad_kinds, "missing_bad_token_budget"):
            return rc
        if rc := require("expected_tokens_mismatch" in bad_kinds, "missing_bad_expected_token_mismatch"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
