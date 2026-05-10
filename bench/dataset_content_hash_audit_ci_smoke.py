#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for dataset content hash audits."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def choices_hash(choices: list[str]) -> str:
    return hashlib.sha256(json.dumps(choices, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def input_hash(prompt_sha256: str, choices_sha256: str) -> str:
    payload = json.dumps(
        {"choices_sha256": choices_sha256, "prompt_sha256": prompt_sha256},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_audit(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "bench" / "dataset_content_hash_audit.py"), *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-content-hash-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "out"
        prompt = "Which tool measures temperature?"
        choices = ["thermometer", "ruler", "scale", "compass"]
        p_hash = prompt_hash(prompt)
        c_hash = choices_hash(choices)
        i_hash = input_hash(p_hash, c_hash)
        passing = tmp_path / "passing.jsonl"
        failing = tmp_path / "failing.jsonl"
        write_jsonl(
            passing,
            [
                {
                    "id": "row-1",
                    "dataset": "smoke",
                    "split": "validation",
                    "prompt": prompt,
                    "choices": choices,
                    "answer_index": 0,
                    "prompt_sha256": p_hash,
                    "choices_sha256": c_hash,
                    "input_sha256": i_hash,
                }
            ],
        )
        write_jsonl(
            failing,
            [
                {
                    "id": "row-1",
                    "dataset": "smoke",
                    "split": "validation",
                    "prompt": prompt,
                    "choices": choices,
                    "answer_index": 0,
                    "metadata": {"prompt_hash": "0" * 64},
                    "choices_sha256": c_hash,
                }
            ],
        )

        passed = run_audit(
            [
                "--input",
                str(passing),
                "--require-all-hashes",
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "content_hash_pass",
            ]
        )
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode
        payload = json.loads((output_dir / "content_hash_pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(payload["summary"]["prompt_hash"]["match"] == 1, "missing_prompt_hash_match"):
            return rc
        rows = list(csv.DictReader((output_dir / "content_hash_pass.csv").open(encoding="utf-8")))
        if rc := require(rows[0]["input_hash_status"] == "match", "missing_input_hash_match"):
            return rc
        junit = ET.parse(output_dir / "content_hash_pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "dataset_content_hash_audit", "missing_junit_name"):
            return rc

        failed = run_audit(
            [
                "--input",
                str(failing),
                "--require-all-hashes",
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "content_hash_fail",
            ]
        )
        if rc := require(failed.returncode == 1, "expected_failure_status"):
            sys.stdout.write(failed.stdout)
            sys.stderr.write(failed.stderr)
            return rc
        failed_payload = json.loads((output_dir / "content_hash_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_payload["findings"]}
        if rc := require({"prompt_hash_mismatch", "missing_input_hash"} <= kinds, "missing_failure_kinds"):
            return rc
    print("dataset_content_hash_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
