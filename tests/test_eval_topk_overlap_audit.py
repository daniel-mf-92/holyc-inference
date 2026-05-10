#!/usr/bin/env python3
"""Host-side tests for eval top-k overlap audit."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_compare
import eval_topk_overlap_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def gold_rows() -> list[dict[str, object]]:
    return [
        {"id": "a", "prompt": "A?", "choices": ["yes", "no", "maybe"], "answer_index": 0},
        {"id": "b", "prompt": "B?", "choices": ["red", "blue", "green"], "answer_index": 1},
    ]


def test_build_pairs_computes_topk_overlap(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    holyc_path = tmp_path / "holyc.jsonl"
    llama_path = tmp_path / "llama.jsonl"
    write_jsonl(gold_path, gold_rows())
    write_jsonl(holyc_path, [{"id": "a", "scores": [3, 2, 1]}, {"id": "b", "scores": [0, 4, 3]}])
    write_jsonl(llama_path, [{"id": "a", "scores": [3, 1, 2]}, {"id": "b", "scores": [0, 4, 3]}])

    findings: list[eval_topk_overlap_audit.Finding] = []
    gold = eval_compare.load_gold(gold_path, "unit", "validation")
    holyc = eval_topk_overlap_audit.load_predictions(holyc_path, "holyc", gold, findings)
    llama = eval_topk_overlap_audit.load_predictions(llama_path, "llama.cpp", gold, findings)
    pairs = eval_topk_overlap_audit.build_pairs(gold, holyc, llama, 2, findings)

    assert findings == []
    assert len(pairs) == 2
    assert pairs[0].overlap_count == 1
    assert pairs[0].jaccard == 1 / 3
    assert pairs[1].topk_exact_match is True


def test_cli_writes_sidecars_and_gates(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    holyc_path = tmp_path / "holyc.jsonl"
    llama_path = tmp_path / "llama.jsonl"
    out = tmp_path / "out"
    write_jsonl(gold_path, gold_rows())
    write_jsonl(holyc_path, [{"id": "a", "scores": [3, 2, 1]}, {"id": "b", "scores": [0, 4, 3]}])
    write_jsonl(llama_path, [{"id": "a", "scores": [3, 2, 1]}, {"id": "b", "scores": [0, 4, 3]}])

    status = eval_topk_overlap_audit.main(
        [
            "--gold",
            str(gold_path),
            "--holyc",
            str(holyc_path),
            "--llama",
            str(llama_path),
            "--dataset",
            "unit",
            "--split",
            "validation",
            "--top-k",
            "2",
            "--min-topk-exact-match-pct",
            "100",
            "--min-avg-jaccard",
            "1",
            "--max-top1-disagree-pct",
            "0",
            "--output-dir",
            str(out),
            "--output-stem",
            "topk",
        ]
    )

    assert status == 0
    payload = json.loads((out / "topk.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert "Eval Top-k Overlap Audit" in (out / "topk.md").read_text(encoding="utf-8")
    assert "record_id" in (out / "topk.csv").read_text(encoding="utf-8")
    assert "kind" in (out / "topk_findings.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(out / "topk_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_eval_topk_overlap_audit"


def test_cli_fails_on_missing_scores(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    holyc_path = tmp_path / "holyc.jsonl"
    llama_path = tmp_path / "llama.jsonl"
    out = tmp_path / "out"
    write_jsonl(gold_path, gold_rows())
    write_jsonl(holyc_path, [{"id": "a", "prediction": 0}, {"id": "b", "prediction": 1}])
    write_jsonl(llama_path, [{"id": "a", "scores": [3, 2, 1]}, {"id": "b", "scores": [0, 4, 3]}])

    status = eval_topk_overlap_audit.main(
        [
            "--gold",
            str(gold_path),
            "--holyc",
            str(holyc_path),
            "--llama",
            str(llama_path),
            "--output-dir",
            str(out),
            "--output-stem",
            "topk",
        ]
    )

    assert status == 1
    payload = json.loads((out / "topk.json").read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in payload["findings"]}
    assert "missing_scores" in kinds


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-eval-topk-test-") as tmp:
        base = Path(tmp)
        test_build_pairs_computes_topk_overlap(base / "pairs")
        test_cli_writes_sidecars_and_gates(base / "cli")
        test_cli_fails_on_missing_scores(base / "missing")
    print("test_eval_topk_overlap_audit=ok")
