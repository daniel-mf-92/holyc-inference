#!/usr/bin/env python3
"""Smoke gate for eval_record_order_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_record_order_audit


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def ordered_rows() -> list[dict[str, object]]:
    return [
        {"id": "smoke-hellaswag-1", "prediction": 0},
        {"id": "smoke-arc-1", "prediction": 0},
        {"id": "smoke-truthfulqa-1", "prediction": 0},
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_audit(output_dir: Path, stem: str, holyc: Path, llama: Path) -> int:
    return eval_record_order_audit.main(
        [
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
            "--fail-on-findings",
        ]
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-record-order-") as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(holyc, ordered_rows())
        write_jsonl(llama, ordered_rows())

        output_dir = ROOT / "bench" / "results"
        status = run_audit(output_dir, "eval_record_order_audit_smoke_latest", holyc, llama)
        require(status == 0, "passing order smoke should pass")
        payload = json.loads((output_dir / "eval_record_order_audit_smoke_latest.json").read_text(encoding="utf-8"))
        require(payload["status"] == "pass", "passing payload should pass")
        require(payload["summary"]["paired_records"] == 3, "paired record count should be 3")
        junit = ET.parse(output_dir / "eval_record_order_audit_smoke_latest_junit.xml").getroot()
        require(junit.attrib["failures"] == "0", "passing junit should have no failures")

        write_jsonl(llama, [ordered_rows()[1], ordered_rows()[0], ordered_rows()[2]])
        failing = run_audit(tmp_path, "failing", holyc, llama)
        require(failing == 1, "reordered llama rows should fail")
        failed_payload = json.loads((tmp_path / "failing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_payload["findings"]}
        require({"order_mismatch", "engine_order_mismatch"} <= kinds, "expected order mismatch findings")

    print("eval_record_order_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
