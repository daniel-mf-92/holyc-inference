#!/usr/bin/env python3
"""Stdlib-only CI smoke checks for prompt audit drift gates."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_prompt_bench


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    prompt_path = ROOT / "bench" / "prompts" / "smoke.jsonl"
    expected_suite = qemu_prompt_bench.prompt_suite_hash(qemu_prompt_bench.load_prompt_cases(prompt_path))

    with tempfile.TemporaryDirectory(prefix="holyc-prompt-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        output_path = tmp_path / "prompt_audit.json"
        markdown_path = tmp_path / "prompt_audit.md"
        csv_path = tmp_path / "prompt_audit.csv"
        junit_path = tmp_path / "prompt_audit_junit.xml"
        pass_command = [
            sys.executable,
            str(ROOT / "bench" / "prompt_audit.py"),
            "--prompts",
            str(prompt_path),
            "--output",
            str(output_path),
            "--markdown",
            str(markdown_path),
            "--csv",
            str(csv_path),
            "--junit",
            str(junit_path),
            "--min-prompts",
            "2",
            "--max-prompt-bytes",
            "1024",
            "--expect-suite-sha256",
            expected_suite,
        ]
        completed = subprocess.run(
            pass_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads(output_path.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_prompt_audit_status"):
            return rc
        if rc := require(report["summary"]["suite_sha256"] == expected_suite, "suite_hash_mismatch"):
            return rc
        if rc := require(report["limits"]["expect_suite_sha256"] == expected_suite, "missing_suite_gate"):
            return rc
        if rc := require("Suite sha256" in markdown_path.read_text(encoding="utf-8"), "missing_markdown_summary"):
            return rc
        if rc := require("row_type,prompt_id,sha256" in csv_path.read_text(encoding="utf-8"), "missing_csv_header"):
            return rc
        junit_root = ET.parse(junit_path).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_prompt_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        fail_output_path = tmp_path / "prompt_audit_drift.json"
        fail_junit_path = tmp_path / "prompt_audit_drift_junit.xml"
        fail_command = [
            sys.executable,
            str(ROOT / "bench" / "prompt_audit.py"),
            "--prompts",
            str(prompt_path),
            "--output",
            str(fail_output_path),
            "--junit",
            str(fail_junit_path),
            "--expect-suite-sha256",
            "0" * 64,
        ]
        completed = subprocess.run(
            fail_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("prompt_suite_gate_did_not_fail=true", file=sys.stderr)
            return 1
        fail_report = json.loads(fail_output_path.read_text(encoding="utf-8"))
        issue_messages = [issue["message"] for issue in fail_report["issues"]]
        if rc := require(
            any("does not match expected" in message for message in issue_messages),
            "missing_suite_gate_issue",
        ):
            return rc
        fail_junit_root = ET.parse(fail_junit_path).getroot()
        if rc := require(fail_junit_root.attrib.get("failures") == "1", "missing_failed_junit"):
            return rc

    print("prompt_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
