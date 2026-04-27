#!/usr/bin/env python3
"""Host-side checks for the offline HolyC vs llama.cpp eval comparator."""

from __future__ import annotations

import importlib.util
import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
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
    assert summary["holyc_macro_f1"] == 1.0
    assert summary["llama_macro_f1"] == 1.0
    assert summary["agreement"] == 1.0
    assert summary["holyc_confusion_matrix"]["matrix"][0][0] == 3


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
        junit_root = ET.parse(Path(tmp) / "smoke_junit.xml").getroot()
        assert payload["summary"]["record_count"] == 3
        assert payload["status"] == "pass"
        assert payload["regressions"] == []
        assert payload["summary"]["class_count"] == 4
        holyc_interval = payload["summary"]["confidence_intervals"]["holyc_accuracy"]
        assert holyc_interval["method"] == "wilson"
        assert holyc_interval["successes"] == 3
        assert holyc_interval["total"] == 3
        assert round(holyc_interval["lower"], 4) == 0.4385
        assert holyc_interval["upper"] == 1.0
        assert payload["summary"]["holyc_per_answer_index"][0]["support"] == 3
        assert (Path(tmp) / "smoke.md").exists()
        markdown = (Path(tmp) / "smoke.md").read_text(encoding="utf-8")
        assert "## Confidence Intervals" in markdown
        assert "No quality gate regressions." in markdown
        assert len(csv_rows) == 3
        assert csv_rows[0]["record_id"] == "smoke-arc-1"
        assert csv_rows[0]["holyc_correct"] == "True"
        assert csv_rows[0]["engines_agree"] == "True"
        assert junit_root.attrib["name"] == "holyc_eval_compare"
        assert junit_root.attrib["failures"] == "0"


def test_compare_reports_macro_f1_and_confusion_matrix() -> None:
    gold = {
        "a": eval_compare.GoldCase("a", "unit", "validation", 0, ["A", "B"]),
        "b": eval_compare.GoldCase("b", "unit", "validation", 0, ["A", "B"]),
        "c": eval_compare.GoldCase("c", "unit", "validation", 1, ["A", "B"]),
    }
    holyc = {
        "a": eval_compare.Prediction("a", 0, 0, None),
        "b": eval_compare.Prediction("b", 1, 1, None),
        "c": eval_compare.Prediction("c", 1, 1, None),
    }
    llama = {
        "a": eval_compare.Prediction("a", 0, 0, None),
        "b": eval_compare.Prediction("b", 0, 0, None),
        "c": eval_compare.Prediction("c", 0, 0, None),
    }

    _, summary = eval_compare.compare(gold, holyc, llama)

    assert round(summary["holyc_macro_f1"], 4) == 0.6667
    assert round(summary["llama_macro_f1"], 4) == 0.4
    assert summary["holyc_confusion_matrix"]["matrix"] == [[1, 1], [0, 1]]
    assert summary["llama_confusion_matrix"]["matrix"] == [[2, 0], [1, 0]]


def test_confidence_level_can_be_configured() -> None:
    interval_90 = eval_compare.wilson_interval(8, 10, 0.90)
    interval_99 = eval_compare.wilson_interval(8, 10, 0.99)

    assert interval_90["point"] == 0.8
    assert interval_99["lower"] < interval_90["lower"]
    assert interval_99["upper"] > interval_90["upper"]


def test_invalid_confidence_level_fails_fast() -> None:
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
                    "--output-dir",
                    tmp,
                    "--confidence-level",
                    "0.97",
                ]
            )
            == 2
        )


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


def test_cli_can_fail_on_quality_gate_regression() -> None:
    gold = BENCH_PATH / "datasets" / "samples" / "smoke_eval.jsonl"
    llama = BENCH_PATH / "eval" / "samples" / "llama_smoke_predictions.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        holyc = Path(tmp) / "wrong.jsonl"
        holyc.write_text(
            "\n".join(
                [
                    '{"id":"smoke-arc-1","prediction":1}',
                    '{"id":"smoke-hellaswag-1","prediction":0}',
                    '{"id":"smoke-truthfulqa-1","prediction":0}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        status = eval_compare.main(
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
                "--output-stem",
                "gated",
                "--min-holyc-accuracy",
                "0.9",
                "--min-agreement",
                "0.9",
                "--max-accuracy-drop",
                "0.1",
                "--fail-on-regression",
            ]
        )
        payload = json.loads((Path(tmp) / "gated.json").read_text(encoding="utf-8"))
        junit_root = ET.parse(Path(tmp) / "gated_junit.xml").getroot()

        assert status == 1
        assert payload["status"] == "fail"
        assert {row["metric"] for row in payload["regressions"]} == {
            "accuracy_delta_holyc_minus_llama",
            "agreement",
            "holyc_accuracy",
        }
        assert junit_root.attrib["failures"] == "3"
        assert junit_root.find("./testcase/failure") is not None


if __name__ == "__main__":
    test_smoke_predictions_compare_cleanly()
    test_cli_writes_json_and_markdown_report()
    test_compare_reports_macro_f1_and_confusion_matrix()
    test_confidence_level_can_be_configured()
    test_invalid_confidence_level_fails_fast()
    test_missing_prediction_fails_fast()
    test_cli_can_fail_on_quality_gate_regression()
    print("eval_compare_tests=ok")
