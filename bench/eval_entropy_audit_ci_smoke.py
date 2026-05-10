#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval entropy audits."""

from __future__ import annotations

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


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_audit(output_dir: Path, holyc: Path, llama: Path, stem: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "eval_entropy_audit.py"),
            "--holyc",
            str(holyc),
            "--llama",
            str(llama),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            stem,
            *extra_args,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-entropy-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(
            holyc,
            [
                {"id": "a", "scores": [4.0, 1.0, 0.0, -1.0]},
                {"id": "b", "scores": [0.5, 2.0, 0.0, -0.5]},
            ],
        )
        write_jsonl(
            llama,
            [
                {"id": "a", "scores": [4.1, 1.1, 0.0, -1.0]},
                {"id": "b", "scores": [0.4, 2.1, 0.0, -0.4]},
            ],
        )
        passed = run_audit(tmp_path, holyc, llama, "pass", "--min-records", "2", "--fail-on-findings")
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode
        payload = json.loads((tmp_path / "pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(payload["summaries"][0]["engine"] == "holyc", "missing_holyc_summary"):
            return rc
        if rc := require((tmp_path / "pass_records.csv").read_text(encoding="utf-8").startswith("source,"), "missing_records_csv"):
            return rc
        junit = ET.parse(tmp_path / "pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_entropy_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_pass_junit_failure"):
            return rc

        bad_holyc = tmp_path / "bad_holyc.jsonl"
        bad_llama = tmp_path / "bad_llama.jsonl"
        write_jsonl(
            bad_holyc,
            [
                {"id": "a", "scores": [50.0, 0.0, 0.0, 0.0]},
                {"id": "b", "scores": [0.0, 0.0, 0.0, 0.0]},
            ],
        )
        write_jsonl(
            bad_llama,
            [
                {"id": "a", "scores": [1.0, 0.9, 0.8, 0.7]},
                {"id": "c", "scores": [1.0, 0.0, 0.0, 0.0]},
            ],
        )
        failed = run_audit(
            tmp_path,
            bad_holyc,
            bad_llama,
            "fail",
            "--min-records",
            "2",
            "--min-normalized-entropy",
            "0.05",
            "--max-normalized-entropy",
            "0.98",
            "--max-record-entropy-delta",
            "0.2",
            "--fail-on-findings",
        )
        if rc := require(failed.returncode == 2, "expected_failure_status"):
            return rc
        failed_payload = json.loads((tmp_path / "fail.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in failed_payload["findings"]}
        if rc := require("normalized_entropy_low" in metrics, "missing_low_entropy_finding"):
            return rc
        if rc := require("normalized_entropy_high" in metrics, "missing_high_entropy_finding"):
            return rc
        if rc := require("missing_llama" in metrics and "missing_holyc" in metrics, "missing_pair_findings"):
            return rc
        if rc := require("record_entropy_delta" in metrics, "missing_delta_finding"):
            return rc
        failed_junit = ET.parse(tmp_path / "fail_junit.xml").getroot()
        if rc := require(failed_junit.attrib.get("failures") == "1", "unexpected_failed_junit"):
            return rc

    print("eval_entropy_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
