#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval_score_parity_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_audit(output_dir: Path, stem: str, holyc: Path, llama: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "eval_score_parity_audit.py"),
            "--gold",
            str(GOLD),
            "--holyc",
            str(holyc),
            "--llama",
            str(llama),
            "--dataset",
            "smoke-eval",
            "--split",
            "validation",
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
    with tempfile.TemporaryDirectory(prefix="holyc-eval-score-parity-ci-") as tmp:
        tmp_path = Path(tmp)
        holyc_pass = tmp_path / "holyc_pass.jsonl"
        llama_pass = tmp_path / "llama_pass.jsonl"
        passing_rows = [
            {"id": "smoke-hellaswag-1", "scores": [5.0, 1.0, 0.5, 0.25]},
            {"id": "smoke-arc-1", "scores": [4.0, 1.0, 0.0, -1.0]},
            {"id": "smoke-truthfulqa-1", "scores": [3.0, 0.0, -1.0, -2.0]},
        ]
        write_jsonl(holyc_pass, passing_rows)
        write_jsonl(llama_pass, passing_rows)
        passed = run_audit(
            tmp_path,
            "score_parity_pass",
            holyc_pass,
            llama_pass,
            "--require-scores",
            "--min-score-parity-pct",
            "100",
            "--max-top-score-tie-pct",
            "0",
        )
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode
        payload = json.loads((tmp_path / "score_parity_pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(payload["summary"]["paired_scored_rows"] == 3, "unexpected_paired_scored_rows"):
            return rc
        if rc := require(payload["summary"]["holyc_top_score_tie_rows"] == 0, "unexpected_holyc_top_score_ties"):
            return rc
        if rc := require("No score parity findings." in (tmp_path / "score_parity_pass.md").read_text(encoding="utf-8"), "missing_pass_markdown"):
            return rc
        if rc := require((tmp_path / "score_parity_pass_pairs.csv").read_text(encoding="utf-8").startswith("record_id,"), "missing_pairs_csv"):
            return rc
        junit = ET.parse(tmp_path / "score_parity_pass_junit.xml").getroot()
        if rc := require(junit.attrib["name"] == "holyc_eval_score_parity_audit", "missing_junit_suite"):
            return rc

        holyc_fail = tmp_path / "holyc_fail.jsonl"
        llama_fail = tmp_path / "llama_fail.jsonl"
        write_jsonl(
            holyc_fail,
            [
                {"id": "smoke-hellaswag-1", "scores": [5.0, 5.0, 0.5, 0.25]},
                {"id": "smoke-arc-1", "prediction": 0},
                {"id": "unknown-extra", "scores": [1.0, 0.0, 0.0, 0.0]},
            ],
        )
        write_jsonl(
            llama_fail,
            [
                {"id": "smoke-hellaswag-1", "scores": [5.0, 1.0, 0.5, 0.25]},
                {"id": "smoke-arc-1", "scores": [4.0, 1.0, 0.0, -1.0]},
            ],
        )
        failed = run_audit(
            tmp_path,
            "score_parity_fail",
            holyc_fail,
            llama_fail,
            "--require-scores",
            "--min-paired-rows",
            "3",
            "--max-top-score-tie-pct",
            "0",
        )
        if rc := require(failed.returncode == 1, "expected_failure_status"):
            return rc
        failed_payload = json.loads((tmp_path / "score_parity_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_payload["findings"]}
        expected = {
            "missing_scores",
            "score_presence_mismatch",
            "missing_id",
            "extra_id",
            "min_paired_rows",
            "max_top_score_tie_pct",
        }
        if rc := require(expected <= kinds, "missing_expected_findings"):
            return rc

    print("eval_score_parity_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
