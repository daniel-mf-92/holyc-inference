#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval_choice_distribution_audit.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"
HOLYC = ROOT / "bench" / "eval" / "samples" / "holyc_smoke_scored_predictions.jsonl"
LLAMA = ROOT / "bench" / "eval" / "samples" / "llama_smoke_scored_predictions.jsonl"


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-choice-distribution-") as tmp:
        output_dir = Path(tmp) / "results"
        command = [
            sys.executable,
            str(ROOT / "bench" / "eval_choice_distribution_audit.py"),
            "--gold",
            str(GOLD),
            "--dataset",
            "smoke",
            "--split",
            "validation",
            "--predictions",
            f"holyc={HOLYC}",
            "--predictions",
            f"llama={LLAMA}",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "eval_choice_distribution_audit_smoke",
            "--choice-collapse-min-predictions",
            "4",
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        json_path = output_dir / "eval_choice_distribution_audit_smoke.json"
        csv_path = output_dir / "eval_choice_distribution_audit_smoke.csv"
        findings_path = output_dir / "eval_choice_distribution_audit_smoke_findings.csv"
        markdown_path = output_dir / "eval_choice_distribution_audit_smoke.md"
        junit_path = output_dir / "eval_choice_distribution_audit_smoke_junit.xml"

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_status"):
            return rc
        if rc := require(payload["summary"]["engines"] == ["holyc", "llama"], "unexpected_engines"):
            return rc
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        if rc := require(len(rows) == 24, "unexpected_distribution_row_count"):
            return rc
        if rc := require({row["engine"] for row in rows} == {"holyc", "llama"}, "missing_engine_rows"):
            return rc
        findings = list(csv.DictReader(findings_path.open(encoding="utf-8")))
        if rc := require(findings == [], "unexpected_findings"):
            return rc
        if rc := require("No choice distribution findings." in markdown_path.read_text(encoding="utf-8"), "missing_markdown_pass"):
            return rc
        junit_root = ET.parse(junit_path).getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_junit_failure"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
