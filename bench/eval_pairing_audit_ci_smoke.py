#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for HolyC-vs-llama eval pairing audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_audit(holyc: Path, llama: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bench" / "eval_pairing_audit.py"),
        "--holyc",
        str(holyc),
        "--llama",
        str(llama),
        "--output",
        str(output_dir / "eval_pairing_audit_latest.json"),
        "--markdown",
        str(output_dir / "eval_pairing_audit_latest.md"),
        "--csv",
        str(output_dir / "eval_pairing_audit_latest.csv"),
        "--findings-csv",
        str(output_dir / "eval_pairing_audit_findings_latest.csv"),
        "--junit",
        str(output_dir / "eval_pairing_audit_junit_latest.xml"),
        "--min-records",
        "3",
        "--require-same-order",
        "--require-predictions",
        "--fail-on-findings",
    ]
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-pairing-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing_holyc = tmp_path / "passing_holyc.jsonl"
        passing_llama = tmp_path / "passing_llama.jsonl"
        failing_holyc = tmp_path / "failing_holyc.jsonl"
        failing_llama = tmp_path / "failing_llama.jsonl"
        base_rows = [
            {"id": "arc-1", "dataset": "arc", "split": "validation", "model": "tiny", "quantization": "Q4_0", "prompt_sha256": "aaa", "prediction": "A", "metadata": {"tokenizer_sha256": "tok-a"}},
            {"id": "arc-2", "dataset": "arc", "split": "validation", "model": "tiny", "quantization": "Q4_0", "prompt_sha256": "bbb", "scores": [0.1, 0.9], "metadata": {"tokenizer_sha256": "tok-a"}},
            {"id": "truth-1", "dataset": "truthfulqa", "split": "validation", "model": "tiny", "quantization": "Q4_0", "prompt_sha256": "ccc", "prediction": 0, "metadata": {"tokenizer_sha256": "tok-a"}},
        ]
        write_jsonl(passing_holyc, base_rows)
        write_jsonl(passing_llama, base_rows)
        write_jsonl(failing_holyc, base_rows)
        write_jsonl(
            failing_llama,
            [
                {**base_rows[1], "quantization": "Q8_0"},
                {**base_rows[0], "prompt_sha256": "changed", "metadata": {"tokenizer_sha256": "tok-b"}},
                {"id": "extra", "dataset": "arc", "split": "validation", "model": "tiny", "quantization": "Q4_0", "prediction": "B"},
            ],
        )

        pass_dir = tmp_path / "pass"
        completed = run_audit(passing_holyc, passing_llama, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        pass_payload = json.loads((pass_dir / "eval_pairing_audit_latest.json").read_text(encoding="utf-8"))
        pass_junit = ET.parse(pass_dir / "eval_pairing_audit_junit_latest.xml").getroot()
        checks = [
            require(pass_payload["status"] == "pass", "eval_pairing_pass_status=false"),
            require(pass_payload["summary"]["paired_records"] == 3, "eval_pairing_pass_pair_count=false"),
            require(pass_junit.attrib.get("failures") == "0", "eval_pairing_pass_junit=false"),
            require("No eval pairing findings" in (pass_dir / "eval_pairing_audit_latest.md").read_text(encoding="utf-8"), "eval_pairing_pass_markdown=false"),
        ]
        if not all(checks):
            return 1

        fail_dir = tmp_path / "fail"
        completed = run_audit(failing_holyc, failing_llama, fail_dir)
        if completed.returncode == 0:
            print("eval_pairing_failing_report_not_rejected=true", file=sys.stderr)
            return 1
        fail_payload = json.loads((fail_dir / "eval_pairing_audit_latest.json").read_text(encoding="utf-8"))
        fail_junit = ET.parse(fail_dir / "eval_pairing_audit_junit_latest.xml").getroot()
        kinds = {finding["kind"] for finding in fail_payload["findings"]}
        checks = [
            require(fail_payload["status"] == "fail", "eval_pairing_fail_status=false"),
            require("metadata_mismatch" in kinds, "eval_pairing_missing_metadata_gate=false"),
            require(any(finding["field"] == "tokenizer_sha256" for finding in fail_payload["findings"]), "eval_pairing_missing_nested_identity_gate=false"),
            require("order_mismatch" in kinds, "eval_pairing_missing_order_gate=false"),
            require("missing_holyc_record" in kinds, "eval_pairing_missing_counterpart_gate=false"),
            require(fail_junit.attrib.get("failures") == "1", "eval_pairing_fail_junit=false"),
        ]
        if not all(checks):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
