#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval prompt hash audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_compare
import eval_prompt_hash_audit


GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def hashed_rows() -> list[dict[str, object]]:
    gold = eval_compare.load_gold(GOLD, "smoke-eval", "validation")
    rows: list[dict[str, object]] = []
    for record_id, case in gold.items():
        hashes = eval_prompt_hash_audit.expected_hashes(case)
        rows.append(
            {
                "id": record_id,
                "prediction": 0,
                "prompt_sha256": hashes.prompt_sha256,
                "choices_sha256": hashes.choices_sha256,
                "input_sha256": hashes.input_sha256,
            }
        )
    return rows


def run_audit(output_dir: Path, holyc: Path, llama: Path, stem: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "eval_prompt_hash_audit.py"),
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
    with tempfile.TemporaryDirectory(prefix="holyc-eval-prompt-hash-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(holyc, hashed_rows())
        write_jsonl(llama, hashed_rows())

        passed = run_audit(
            tmp_path,
            holyc,
            llama,
            "pass",
            "--require-hashes",
            "--min-hashed-rows",
            "6",
            "--fail-on-findings",
        )
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode
        payload = json.loads((tmp_path / "pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(payload["summary"]["matched_hash_fields"] == 18, "unexpected_match_count"):
            return rc
        junit = ET.parse(tmp_path / "pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_prompt_hash_audit", "missing_junit_name"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_pass_junit_failure"):
            return rc

        bad_holyc_rows = hashed_rows()
        bad_holyc_rows[0]["input_sha256"] = "f" * 64
        bad_holyc = tmp_path / "bad_holyc.jsonl"
        bad_llama = tmp_path / "bad_llama.jsonl"
        write_jsonl(bad_holyc, bad_holyc_rows)
        write_jsonl(bad_llama, [{"id": "smoke-hellaswag-1", "prediction": 0}, {"id": "extra", "prediction": 0}])
        failed = run_audit(tmp_path, bad_holyc, bad_llama, "fail", "--require-hashes", "--fail-on-findings")
        if rc := require(failed.returncode == 2, "expected_failure_status"):
            sys.stdout.write(failed.stdout)
            sys.stderr.write(failed.stderr)
            return rc
        failed_payload = json.loads((tmp_path / "fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_payload["findings"]}
        expected = {"input_sha256_mismatch", "missing_prompt_sha256", "missing_prediction", "extra_prediction"}
        if rc := require(expected <= kinds, "missing_failure_kinds"):
            return rc

    print("eval_prompt_hash_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
