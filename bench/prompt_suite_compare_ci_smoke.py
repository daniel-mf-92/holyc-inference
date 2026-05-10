#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for prompt suite comparison."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_compare(baseline: Path, candidate: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bench" / "prompt_suite_compare.py"),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--output",
        str(output_dir / "prompt_suite_compare_latest.json"),
        "--markdown",
        str(output_dir / "prompt_suite_compare_latest.md"),
        "--csv",
        str(output_dir / "prompt_suite_compare_latest.csv"),
        "--junit",
        str(output_dir / "prompt_suite_compare_junit_latest.xml"),
        *extra_args,
    ]
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-prompt-suite-compare-ci-") as tmp:
        tmp_path = Path(tmp)
        baseline = tmp_path / "baseline.jsonl"
        same = tmp_path / "same.jsonl"
        changed = tmp_path / "changed.jsonl"
        rows = [
            {"prompt_id": "short", "prompt": "Say hello.", "expected_tokens": 8},
            {"prompt_id": "code", "prompt": "Write a tiny loop.", "expected_tokens": 16},
        ]
        write_jsonl(baseline, rows)
        write_jsonl(same, rows)
        write_jsonl(
            changed,
            [
                {"prompt_id": "code", "prompt": "Write a tiny loop.", "expected_tokens": 16},
                {"prompt_id": "short", "prompt": "Say hello!", "expected_tokens": 9},
            ],
        )

        pass_dir = tmp_path / "pass"
        completed = run_compare(baseline, same, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        pass_report = json.loads((pass_dir / "prompt_suite_compare_latest.json").read_text(encoding="utf-8"))
        pass_junit = ET.parse(pass_dir / "prompt_suite_compare_junit_latest.xml").getroot()
        checks = [
            require(pass_report["status"] == "pass", "prompt_suite_compare_pass_status=false"),
            require(pass_report["compared_prompt_count"] == 2, "prompt_suite_compare_pass_count=false"),
            require(pass_junit.attrib.get("failures") == "0", "prompt_suite_compare_pass_junit=false"),
            require(
                "Prompt suites match" in (pass_dir / "prompt_suite_compare_latest.md").read_text(encoding="utf-8"),
                "prompt_suite_compare_pass_markdown=false",
            ),
        ]
        if not all(checks):
            return 1

        fail_dir = tmp_path / "fail"
        completed = run_compare(baseline, changed, fail_dir)
        if completed.returncode == 0:
            print("prompt_suite_compare_changed_not_rejected=true", file=sys.stderr)
            return 1
        fail_report = json.loads((fail_dir / "prompt_suite_compare_latest.json").read_text(encoding="utf-8"))
        reasons = {finding["kind"] for finding in fail_report["findings"]}
        fail_junit = ET.parse(fail_dir / "prompt_suite_compare_junit_latest.xml").getroot()
        checks = [
            require(fail_report["status"] == "fail", "prompt_suite_compare_fail_status=false"),
            require("prompt_order_mismatch" in reasons, "prompt_suite_compare_missing_order=false"),
            require("prompt_text_mismatch" in reasons, "prompt_suite_compare_missing_text=false"),
            require("expected_tokens_mismatch" in reasons, "prompt_suite_compare_missing_expected_tokens=false"),
            require(fail_junit.attrib.get("failures") == "1", "prompt_suite_compare_fail_junit=false"),
        ]
        if not all(checks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
