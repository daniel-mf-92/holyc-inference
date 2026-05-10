#!/usr/bin/env python3
"""Smoke gate for eval_error_overlap_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_error_overlap_audit


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_audit(output_dir: Path, stem: str, holyc: Path, llama: Path, *extra: str) -> int:
    return eval_error_overlap_audit.main(
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
            *extra,
        ]
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-error-overlap-") as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(
            holyc,
            [
                {"id": "smoke-hellaswag-1", "prediction": 0},
                {"id": "smoke-arc-1", "prediction": 1},
                {"id": "smoke-truthfulqa-1", "prediction": 1},
            ],
        )
        write_jsonl(
            llama,
            [
                {"id": "smoke-hellaswag-1", "prediction": 0},
                {"id": "smoke-arc-1", "prediction": 2},
                {"id": "smoke-truthfulqa-1", "prediction": 0},
            ],
        )

        output_dir = ROOT / "bench" / "results"
        status = run_audit(
            output_dir,
            "eval_error_overlap_audit_smoke_latest",
            holyc,
            llama,
            "--min-paired-records",
            "3",
            "--min-error-jaccard",
            "0.25",
            "--max-holyc-unique-error-excess",
            "1",
        )
        require(status == 0, "overlap smoke should pass")
        payload = json.loads((output_dir / "eval_error_overlap_audit_smoke_latest.json").read_text(encoding="utf-8"))
        require(payload["status"] == "pass", "passing payload should pass")
        require(payload["summary"]["paired_records"] == 3, "paired count should be 3")
        require(payload["summary"]["shared_errors"] == 1, "shared error count should be 1")
        require(payload["summary"]["holyc_unique_errors"] == 1, "HolyC unique error count should be 1")
        require(payload["summary"]["llama_unique_errors"] == 0, "llama unique error count should be 0")
        require(abs(payload["summary"]["error_jaccard"] - 0.5) < 0.000001, "unexpected Jaccard")
        junit = ET.parse(output_dir / "eval_error_overlap_audit_smoke_latest_junit.xml").getroot()
        require(junit.attrib["failures"] == "0", "passing junit should have no failures")

        failing = run_audit(tmp_path, "failing", holyc, llama, "--min-error-jaccard", "0.75")
        require(failing == 1, "high Jaccard gate should fail")
        failed_payload = json.loads((tmp_path / "failing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_payload["findings"]}
        require("min_error_jaccard" in kinds, "expected min_error_jaccard finding")

    print("eval_error_overlap_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
