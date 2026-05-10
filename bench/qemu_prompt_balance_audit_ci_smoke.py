#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_prompt_balance_audit.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "bench" / "results" / "qemu_prompt_bench_latest.json"


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-prompt-balance-") as tmp:
        output_dir = Path(tmp) / "results"
        command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_balance_audit.py"),
            str(ARTIFACT),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_prompt_balance_audit_smoke",
            "--min-prompts",
            "2",
            "--min-measured-runs",
            "4",
            "--min-successful-runs-per-prompt",
            "2",
            "--max-successful-run-delta",
            "0",
            "--fail-on-failed-runs",
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        json_path = output_dir / "qemu_prompt_balance_audit_smoke.json"
        csv_path = output_dir / "qemu_prompt_balance_audit_smoke.csv"
        findings_path = output_dir / "qemu_prompt_balance_audit_smoke_findings.csv"
        markdown_path = output_dir / "qemu_prompt_balance_audit_smoke.md"
        junit_path = output_dir / "qemu_prompt_balance_audit_smoke_junit.xml"

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_status"):
            return rc
        if rc := require(payload["summary"]["prompts"] == 2, "unexpected_prompt_count"):
            return rc
        if rc := require(payload["summary"]["successful_runs"] == 4, "unexpected_successful_runs"):
            return rc
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        if rc := require({row["prompt"] for row in rows} == {"smoke-code", "smoke-short"}, "unexpected_prompt_rows"):
            return rc
        if rc := require({row["successful_runs"] for row in rows} == {"2"}, "unbalanced_smoke_rows"):
            return rc
        findings = list(csv.DictReader(findings_path.open(encoding="utf-8")))
        if rc := require(findings == [], "unexpected_findings"):
            return rc
        if rc := require("No prompt balance findings." in markdown_path.read_text(encoding="utf-8"), "missing_markdown_pass"):
            return rc
        junit_root = ET.parse(junit_path).getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_junit_failure"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
