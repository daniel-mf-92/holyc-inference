#!/usr/bin/env python3
"""Tests for HolyC-vs-llama score delta audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_score_delta_audit


def write_gold(path: Path) -> None:
    rows = [
        {
            "id": "case-1",
            "dataset": "smoke",
            "split": "validation",
            "question": "Pick A",
            "choices": [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
            "answerKey": "A",
        },
        {
            "id": "case-2",
            "dataset": "smoke",
            "split": "validation",
            "question": "Pick B",
            "choices": [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
            "answerKey": "B",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def base_predictions() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    holyc = [
        {"id": "case-1", "scores": [3.0, 1.0]},
        {"id": "case-2", "scores": [1.0, 4.0]},
    ]
    llama = [
        {"id": "case-1", "scores": [3.1, 0.95]},
        {"id": "case-2", "scores": [1.05, 4.1]},
    ]
    return holyc, llama


def run_audit(tmp_path: Path, extra: list[str] | None = None) -> int:
    gold = tmp_path / "gold.jsonl"
    holyc_path = tmp_path / "holyc.jsonl"
    llama_path = tmp_path / "llama.jsonl"
    holyc, llama = base_predictions()
    write_gold(gold)
    write_predictions(holyc_path, holyc)
    write_predictions(llama_path, llama)
    args = [
        "--gold",
        str(gold),
        "--holyc",
        str(holyc_path),
        "--llama",
        str(llama_path),
        "--dataset",
        "smoke",
        "--split",
        "validation",
        "--output-dir",
        str(tmp_path / "out"),
        "--output-stem",
        "delta",
    ]
    if extra:
        args.extend(extra)
    return eval_score_delta_audit.main(args)


def test_score_delta_audit_passes_with_small_deltas(tmp_path: Path) -> None:
    status = run_audit(
        tmp_path,
        ["--max-abs-delta", "0.2", "--max-mean-abs-delta", "0.1", "--min-top-index-match-pct", "100"],
    )

    assert status == 0
    payload = json.loads((tmp_path / "out/delta.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["paired_scored_records"] == 2
    assert payload["summary"]["top_index_match_pct"] == 100.0


def test_score_delta_audit_flags_threshold_and_top_index_failures(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    holyc_path = tmp_path / "holyc.jsonl"
    llama_path = tmp_path / "llama.jsonl"
    holyc, llama = base_predictions()
    llama[1] = {"id": "case-2", "scores": [5.0, 0.0]}
    write_gold(gold)
    write_predictions(holyc_path, holyc)
    write_predictions(llama_path, llama)

    status = eval_score_delta_audit.main(
        [
            "--gold",
            str(gold),
            "--holyc",
            str(holyc_path),
            "--llama",
            str(llama_path),
            "--dataset",
            "smoke",
            "--split",
            "validation",
            "--min-top-index-match-pct",
            "100",
            "--max-abs-delta",
            "0.2",
            "--max-mean-abs-delta",
            "0.1",
            "--output-dir",
            str(tmp_path / "out"),
            "--output-stem",
            "delta",
        ]
    )

    assert status == 1
    payload = json.loads((tmp_path / "out/delta.json").read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in payload["findings"]}
    assert {"top_index_match", "max_abs_delta", "mean_abs_delta"} <= kinds


def test_score_delta_audit_writes_sidecars(tmp_path: Path) -> None:
    status = run_audit(tmp_path, ["--max-abs-delta", "0.2", "--max-mean-abs-delta", "0.1"])

    assert status == 0
    rows = list(csv.DictReader((tmp_path / "out/delta.csv").open(encoding="utf-8")))
    assert rows[0]["record_id"] == "case-1"
    findings = list(csv.DictReader((tmp_path / "out/delta_findings.csv").open(encoding="utf-8")))
    assert findings == []
    markdown = (tmp_path / "out/delta.md").read_text(encoding="utf-8")
    assert "Eval Score Delta Audit" in markdown
    junit_root = ET.parse(tmp_path / "out/delta_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_eval_score_delta_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_score_delta_audit_passes_with_small_deltas(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_score_delta_audit_flags_threshold_and_top_index_failures(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_score_delta_audit_writes_sidecars(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
