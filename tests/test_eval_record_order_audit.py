#!/usr/bin/env python3
"""Tests for host-side eval record-order audit."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import eval_record_order_audit


GOLD = BENCH / "datasets" / "samples" / "smoke_eval.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def ordered_rows() -> list[dict[str, object]]:
    return [
        {"id": "smoke-hellaswag-1", "prediction": 0},
        {"id": "smoke-arc-1", "prediction": 0},
        {"id": "smoke-truthfulqa-1", "prediction": 0},
    ]


def args(tmp_path: Path, holyc: Path, llama: Path, stem: str) -> list[str]:
    return [
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
        str(tmp_path),
        "--output-stem",
        stem,
        "--fail-on-findings",
    ]


def test_order_audit_passes_matching_gold_order(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, ordered_rows())
    write_jsonl(llama, ordered_rows())

    assert eval_record_order_audit.main(args(tmp_path, holyc, llama, "pass")) == 0
    payload = json.loads((tmp_path / "pass.json").read_text(encoding="utf-8"))

    assert payload["status"] == "pass"
    assert payload["summary"]["gold_records"] == 3
    assert payload["summary"]["paired_records"] == 3
    assert all(row["engines_match"] for row in payload["rows"])
    assert "No eval record order findings." in (tmp_path / "pass.md").read_text(encoding="utf-8")
    junit = ET.parse(tmp_path / "pass_junit.xml").getroot()
    assert junit.attrib["name"] == "holyc_eval_record_order_audit"
    assert junit.attrib["failures"] == "0"


def test_order_audit_fails_reordered_engine_rows(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, ordered_rows())
    write_jsonl(llama, [ordered_rows()[1], ordered_rows()[0], ordered_rows()[2]])

    assert eval_record_order_audit.main(args(tmp_path, holyc, llama, "fail")) == 1
    payload = json.loads((tmp_path / "fail.json").read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in payload["findings"]}

    assert "order_mismatch" in kinds
    assert "engine_order_mismatch" in kinds


def test_order_audit_reports_missing_extra_and_duplicate_ids(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, ordered_rows() + [ordered_rows()[0], {"id": "extra", "prediction": 0}])
    write_jsonl(llama, ordered_rows()[:2])

    assert eval_record_order_audit.main(args(tmp_path, holyc, llama, "ids")) == 1
    payload = json.loads((tmp_path / "ids.json").read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in payload["findings"]}

    assert {"duplicate_id", "extra_id", "missing_id"} <= kinds


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_order_audit_passes_matching_gold_order(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_order_audit_fails_reordered_engine_rows(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_order_audit_reports_missing_extra_and_duplicate_ids(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
