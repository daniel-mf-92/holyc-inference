#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval score order audits."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def run_audit(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "bench" / "eval_score_order_audit.py"), *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-score-order-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing = tmp_path / "passing.jsonl"
        failing = tmp_path / "failing.jsonl"
        output_dir = tmp_path / "out"
        write_jsonl(
            passing,
            [
                {"id": "a", "prediction": 0, "scores": [3.0, 1.0, 0.0]},
                {"id": "b", "prediction": "B", "scores": [0.0, 4.0, 2.0]},
            ],
        )
        write_jsonl(
            failing,
            [
                {"id": "a", "prediction": 1, "scores": [3.0, 1.0, 0.0]},
                {"id": "b", "prediction": 0, "scores": [2.0, 2.0, 1.0]},
            ],
        )

        completed = run_audit(
            [
                "--predictions",
                f"holyc={passing}",
                "--predictions",
                f"llama={passing}",
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "score_order_pass",
                "--require-both",
                "--min-checked-records",
                "2",
            ]
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        payload = json.loads((output_dir / "score_order_pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(payload["summary"]["checked_count"] == 4, "unexpected_checked_count"):
            return rc
        rows = list(csv.DictReader((output_dir / "score_order_pass_records.csv").open(encoding="utf-8")))
        if rc := require({row["matches_top_score"] for row in rows} == {"True"}, "unexpected_record_match"):
            return rc
        if rc := require("No score order findings." in (output_dir / "score_order_pass.md").read_text(encoding="utf-8"), "missing_pass_markdown"):
            return rc
        junit = ET.parse(output_dir / "score_order_pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_score_order_audit", "missing_junit_name"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_junit_failure"):
            return rc

        failed = run_audit(
            [
                "--predictions",
                f"holyc={failing}",
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "score_order_fail",
                "--require-both",
            ]
        )
        if rc := require(failed.returncode == 1, "expected_failure_status"):
            sys.stdout.write(failed.stdout)
            sys.stderr.write(failed.stderr)
            return rc
        failed_payload = json.loads((output_dir / "score_order_fail.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in failed_payload["findings"]}
        if rc := require({"prediction_score_mismatch", "top_score_tie"} <= metrics, "missing_failure_metrics"):
            return rc
    print("eval_score_order_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
