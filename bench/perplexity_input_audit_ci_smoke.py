#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for perplexity input artifact auditing."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOLYC = ROOT / "bench" / "eval" / "samples" / "holyc_smoke_logprobs.jsonl"
LLAMA = ROOT / "bench" / "eval" / "samples" / "llama_smoke_logprobs.jsonl"


def run_command(command: list[str], *, expected_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0 and not expected_failure:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-perplexity-input-audit-ci-") as tmp:
        output_dir = Path(tmp)
        stem = "perplexity_input_audit_smoke"
        completed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "perplexity_input_audit.py"),
                str(HOLYC),
                str(LLAMA),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                stem,
                "--min-records",
                "6",
                "--min-records-per-source",
                "3",
                "--min-tokens",
                "22",
                "--min-tokens-per-source",
                "11",
            ]
        )
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads((output_dir / f"{stem}.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_perplexity_input_status"):
            return rc
        if rc := require(report["summary"]["sources"] == 2, "unexpected_perplexity_input_source_count"):
            return rc
        if rc := require(report["summary"]["records"] == 6, "unexpected_perplexity_input_record_count"):
            return rc
        if rc := require(report["summary"]["tokens"] == 22, "unexpected_perplexity_input_token_count"):
            return rc
        if rc := require(report["findings"] == [], "unexpected_perplexity_input_findings"):
            return rc

        expected_files = [
            f"{stem}.json",
            f"{stem}.md",
            f"{stem}.csv",
            f"{stem}_sources.csv",
            f"{stem}_findings.csv",
            f"{stem}_junit.xml",
        ]
        if rc := require(all((output_dir / name).exists() for name in expected_files), "missing_perplexity_input_artifact"):
            return rc
        if rc := require("Perplexity Input Audit" in (output_dir / f"{stem}.md").read_text(encoding="utf-8"), "missing_perplexity_input_markdown"):
            return rc
        if rc := require(len(load_csv_rows(output_dir / f"{stem}.csv")) == 6, "unexpected_perplexity_input_csv_rows"):
            return rc
        if rc := require(len(load_csv_rows(output_dir / f"{stem}_sources.csv")) == 2, "unexpected_perplexity_input_source_rows"):
            return rc
        if rc := require(load_csv_rows(output_dir / f"{stem}_findings.csv") == [], "unexpected_perplexity_input_finding_rows"):
            return rc
        junit_root = ET.parse(output_dir / f"{stem}_junit.xml").getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_perplexity_input_audit", "missing_perplexity_input_junit"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_perplexity_input_junit_failures"):
            return rc

        bad = output_dir / "bad.jsonl"
        bad.write_text('{"id":"bad","token_count":1,"token_logprobs":[0.5]}\n', encoding="utf-8")
        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "perplexity_input_audit.py"),
                str(bad),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "perplexity_input_audit_fail",
                "--require-dataset",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "expected_perplexity_input_failure"):
            return rc
        failed_report = json.loads((output_dir / "perplexity_input_audit_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require({"positive_logprob", "missing_dataset"} <= kinds, "missing_expected_perplexity_input_findings"):
            return rc

    print("perplexity_input_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
