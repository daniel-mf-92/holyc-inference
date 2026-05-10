#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval hash audits."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_eval_report(path: Path, *, gold_sha256: str, bad_hash: bool = False) -> None:
    payload = {
        "dataset": "smoke-eval",
        "split": "validation",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "status": "pass",
        "gold_sha256": gold_sha256,
        "holyc_predictions_sha256": "not-a-sha" if bad_hash else sha256_bytes(b"holyc-predictions"),
        "llama_predictions_sha256": sha256_bytes(b"llama-predictions"),
        "rows": [],
        "summary": {"record_count": 3, "holyc_accuracy": 1.0, "llama_accuracy": 1.0, "agreement": 1.0},
        "regressions": [],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-hash-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        gold = tmp_path / "gold.jsonl"
        gold.write_text('{"record_id":"a"}\n{"record_id":"b"}\n{"record_id":"c"}\n', encoding="utf-8")
        gold_hash = sha256_bytes(gold.read_bytes())
        passing = tmp_path / "eval_compare_pass.json"
        write_eval_report(passing, gold_sha256=gold_hash)
        output_dir = tmp_path / "out"

        completed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "eval_hash_audit.py"),
                str(passing),
                "--gold-path",
                str(gold),
                "--output",
                str(output_dir / "eval_hash_audit_smoke.json"),
                "--markdown",
                str(output_dir / "eval_hash_audit_smoke.md"),
                "--csv",
                str(output_dir / "eval_hash_audit_smoke.csv"),
                "--findings-csv",
                str(output_dir / "eval_hash_audit_smoke_findings.csv"),
                "--junit",
                str(output_dir / "eval_hash_audit_smoke_junit.xml"),
                "--fail-on-findings",
            ]
        )
        if completed.returncode != 0:
            return completed.returncode
        report = json.loads((output_dir / "eval_hash_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(report["summary"]["rows"] == 3, "unexpected_row_count"):
            return rc
        if rc := require("Eval Hash Audit" in (output_dir / "eval_hash_audit_smoke.md").read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        junit = ET.parse(output_dir / "eval_hash_audit_smoke_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_hash_audit", "missing_junit"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_junit_failure"):
            return rc

        failing = tmp_path / "eval_compare_fail.json"
        write_eval_report(failing, gold_sha256=sha256_bytes(b"wrong-gold"), bad_hash=True)
        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "eval_hash_audit.py"),
                str(failing),
                "--gold-path",
                str(gold),
                "--output",
                str(output_dir / "eval_hash_audit_fail.json"),
                "--markdown",
                str(output_dir / "eval_hash_audit_fail.md"),
                "--csv",
                str(output_dir / "eval_hash_audit_fail.csv"),
                "--findings-csv",
                str(output_dir / "eval_hash_audit_fail_findings.csv"),
                "--junit",
                str(output_dir / "eval_hash_audit_fail_junit.xml"),
                "--fail-on-findings",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "expected_failure"):
            return rc
        failed_report = json.loads((output_dir / "eval_hash_audit_fail.json").read_text(encoding="utf-8"))
        gates = {finding["gate"] for finding in failed_report["findings"]}
        if rc := require("invalid_hash_format" in gates, "missing_invalid_hash_format"):
            return rc
        if rc := require("gold_hash_mismatch" in gates, "missing_gold_hash_mismatch"):
            return rc

    print("eval_hash_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
