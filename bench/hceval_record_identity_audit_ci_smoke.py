#!/usr/bin/env python3
"""Smoke gate for packed HCEval record identity auditing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "bench"))

import dataset_pack


def write_hceval(path: Path, records: list[dataset_pack.EvalRecord]) -> None:
    dataset_pack.write_outputs(records, path, path.with_suffix(path.suffix + ".manifest.json"), "identity-smoke", "validation")


def record(record_id: str, prompt: str, choices: list[str], answer_index: int) -> dataset_pack.EvalRecord:
    return dataset_pack.EvalRecord(
        record_id=record_id,
        dataset="identity-smoke",
        split="validation",
        prompt=prompt,
        choices=choices,
        answer_index=answer_index,
        provenance="synthetic",
    )


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "hceval_record_identity_audit.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_dir / "hceval_record_identity_audit_smoke.json"),
            "--markdown",
            str(output_dir / "hceval_record_identity_audit_smoke.md"),
            "--csv",
            str(output_dir / "hceval_record_identity_audit_smoke.csv"),
            "--artifacts-csv",
            str(output_dir / "hceval_record_identity_audit_smoke_artifacts.csv"),
            "--findings-csv",
            str(output_dir / "hceval_record_identity_audit_smoke_findings.csv"),
            "--junit",
            str(output_dir / "hceval_record_identity_audit_smoke_junit.xml"),
            "--fail-on-findings",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-hceval-record-identity-") as tmp:
        tmp_path = Path(tmp)

        clean = tmp_path / "clean.hceval"
        write_hceval(
            clean,
            [
                record("arc-1", "A frozen lake is mostly made of what?", ["water", "metal", "sand", "wood"], 0),
                record("arc-2", "Which tool measures temperature?", ["ruler", "thermometer", "scale", "timer"], 1),
            ],
        )
        clean_out = tmp_path / "clean_out"
        completed = run_audit(clean, clean_out)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        clean_report = json.loads((clean_out / "hceval_record_identity_audit_smoke.json").read_text(encoding="utf-8"))
        clean_junit = ET.parse(clean_out / "hceval_record_identity_audit_smoke_junit.xml").getroot()
        checks = [
            require(clean_report["status"] == "pass", "clean_identity_status_not_pass=true"),
            require(clean_report["summary"]["records"] == 2, "clean_identity_record_count_wrong=true"),
            require(clean_report["findings"] == [], "clean_identity_has_findings=true"),
            require(clean_junit.attrib.get("failures") == "0", "clean_identity_junit_failures=true"),
            require(
                "HCEval Record Identity Audit"
                in (clean_out / "hceval_record_identity_audit_smoke.md").read_text(encoding="utf-8"),
                "clean_identity_markdown_missing_title=true",
            ),
        ]
        if not all(checks):
            return 1

        dirty = tmp_path / "dirty.hceval"
        duplicate_choices = ["evaporation", "freezing", "melting", "condensation"]
        write_hceval(
            dirty,
            [
                record("dup-1", "Water turning into vapor is called what?", duplicate_choices, 0),
                record("dup-1", "Water turning into vapor is called what?", duplicate_choices, 0),
            ],
        )
        dirty_out = tmp_path / "dirty_out"
        completed = run_audit(dirty, dirty_out)
        if completed.returncode == 0:
            print("dirty_identity_not_rejected=true", file=sys.stderr)
            return 1
        dirty_report = json.loads((dirty_out / "hceval_record_identity_audit_smoke.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in dirty_report["findings"]}
        dirty_junit = ET.parse(dirty_out / "hceval_record_identity_audit_smoke_junit.xml").getroot()
        checks = [
            require(dirty_report["status"] == "fail", "dirty_identity_status_not_fail=true"),
            require("duplicate_record_id" in kinds, "dirty_identity_missing_duplicate_id=true"),
            require("duplicate_input_payload" in kinds, "dirty_identity_missing_duplicate_input=true"),
            require("duplicate_answer_payload" in kinds, "dirty_identity_missing_duplicate_answer=true"),
            require(int(dirty_junit.attrib.get("failures", "0")) == 1, "dirty_identity_junit_failure_count_wrong=true"),
        ]
        if not all(checks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
