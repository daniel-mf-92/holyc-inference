#!/usr/bin/env python3
"""Host-side checks for the offline HolyC vs llama.cpp eval comparator."""

from __future__ import annotations

import importlib.util
import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

COMPARE_PATH = BENCH_PATH / "eval_compare.py"
spec = importlib.util.spec_from_file_location("eval_compare", COMPARE_PATH)
eval_compare = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_compare"] = eval_compare
spec.loader.exec_module(eval_compare)


def test_smoke_predictions_compare_cleanly() -> None:
    gold = BENCH_PATH / "datasets" / "samples" / "smoke_eval.jsonl"
    holyc = BENCH_PATH / "eval" / "samples" / "holyc_smoke_predictions.jsonl"
    llama = BENCH_PATH / "eval" / "samples" / "llama_smoke_predictions.jsonl"

    gold_rows = eval_compare.load_gold(gold, "smoke-eval", "validation")
    holyc_predictions = eval_compare.load_predictions(holyc, gold_rows)
    llama_predictions = eval_compare.load_predictions(llama, gold_rows)
    rows, summary = eval_compare.compare(gold_rows, holyc_predictions, llama_predictions)

    assert len(rows) == 3
    assert summary["holyc_accuracy"] == 1.0
    assert summary["llama_accuracy"] == 1.0
    assert summary["agreement"] == 1.0


def test_cli_writes_json_and_markdown_report() -> None:
    gold = BENCH_PATH / "datasets" / "samples" / "smoke_eval.jsonl"
    holyc = BENCH_PATH / "eval" / "samples" / "holyc_smoke_predictions.jsonl"
    llama = BENCH_PATH / "eval" / "samples" / "llama_smoke_predictions.jsonl"

    with tempfile.TemporaryDirectory() as tmp:
        assert (
            eval_compare.main(
                [
                    "--gold",
                    str(gold),
                    "--holyc",
                    str(holyc),
                    "--llama",
                    str(llama),
                    "--dataset",
                    "smoke-eval",
                    "--split",
                    "validation",
                    "--model",
                    "synthetic-smoke",
                    "--quantization",
                    "Q4_0",
                    "--output-dir",
                    tmp,
                    "--output-stem",
                    "smoke",
                ]
            )
            == 0
        )
        payload = json.loads((Path(tmp) / "smoke.json").read_text(encoding="utf-8"))
        csv_rows = list(csv.DictReader((Path(tmp) / "smoke.csv").open(newline="", encoding="utf-8")))
        assert payload["summary"]["record_count"] == 3
        assert (Path(tmp) / "smoke.md").exists()
        assert len(csv_rows) == 3
        assert csv_rows[0]["record_id"] == "smoke-arc-1"
        assert csv_rows[0]["holyc_correct"] == "True"
        assert csv_rows[0]["engines_agree"] == "True"


def test_missing_prediction_fails_fast() -> None:
    gold = BENCH_PATH / "datasets" / "samples" / "smoke_eval.jsonl"
    llama = BENCH_PATH / "eval" / "samples" / "llama_smoke_predictions.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        holyc = Path(tmp) / "missing.jsonl"
        holyc.write_text('{"id":"smoke-hellaswag-1","prediction":0}\n', encoding="utf-8")
        assert (
            eval_compare.main(
                [
                    "--gold",
                    str(gold),
                    "--holyc",
                    str(holyc),
                    "--llama",
                    str(llama),
                    "--dataset",
                    "smoke-eval",
                    "--split",
                    "validation",
                    "--output-dir",
                    tmp,
                ]
            )
            == 2
        )


if __name__ == "__main__":
    test_smoke_predictions_compare_cleanly()
    test_cli_writes_json_and_markdown_report()
    test_missing_prediction_fails_fast()
    print("eval_compare_tests=ok")
