#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval score sparsity audits."""

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
            str(ROOT / "bench" / "eval_score_sparsity_audit.py"),
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
    with tempfile.TemporaryDirectory(prefix="holyc-eval-score-sparsity-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(
            holyc,
            [
                {"id": "a", "scores": [4.0, 1.0, 0.5, 0.25]},
                {"id": "b", "scores": [0.5, 2.0, -0.25, -0.5]},
            ],
        )
        write_jsonl(
            llama,
            [
                {"id": "a", "scores": [4.1, 1.1, 0.4, 0.2]},
                {"id": "b", "scores": [0.4, 2.1, -0.2, -0.4]},
            ],
        )
        passed = run_audit(
            tmp_path,
            holyc,
            llama,
            "pass",
            "--min-records",
            "2",
            "--min-nonzero-scores-per-record",
            "4",
            "--min-unique-scores-per-record",
            "4",
            "--max-zero-score-pct",
            "0",
            "--fail-on-findings",
        )
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode
        payload = json.loads((tmp_path / "pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require((tmp_path / "pass_records.csv").read_text(encoding="utf-8").startswith("source,"), "missing_records_csv"):
            return rc
        junit = ET.parse(tmp_path / "pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_score_sparsity_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_pass_junit_failure"):
            return rc

        sparse_holyc = tmp_path / "sparse_holyc.jsonl"
        sparse_llama = tmp_path / "sparse_llama.jsonl"
        write_jsonl(
            sparse_holyc,
            [
                {"id": "a", "scores": [1.0, 0.0, 0.0, 0.0]},
                {"id": "b", "scores": [0.0, 0.0, 0.0, 0.0]},
            ],
        )
        write_jsonl(
            sparse_llama,
            [
                {"id": "a", "scores": [4.0, 1.0, 0.5, 0.25]},
                {"id": "b", "scores": [3.0, 2.0, 1.0, -1.0]},
            ],
        )
        failed = run_audit(
            tmp_path,
            sparse_holyc,
            sparse_llama,
            "fail",
            "--min-records",
            "2",
            "--min-nonzero-scores-per-record",
            "2",
            "--min-unique-scores-per-record",
            "2",
            "--max-zero-score-pct",
            "50",
            "--fail-on-findings",
        )
        if rc := require(failed.returncode == 2, "expected_failure_status"):
            return rc
        failed_payload = json.loads((tmp_path / "fail.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in failed_payload["findings"]}
        if rc := require("zero_score_pct" in metrics, "missing_zero_pct_finding"):
            return rc
        if rc := require("nonzero_score_count" in metrics, "missing_nonzero_count_finding"):
            return rc
        if rc := require("unique_score_count" in metrics, "missing_unique_count_finding"):
            return rc
        failed_junit = ET.parse(tmp_path / "fail_junit.xml").getroot()
        if rc := require(failed_junit.attrib.get("failures") == "1", "unexpected_failed_junit"):
            return rc

    print("eval_score_sparsity_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
