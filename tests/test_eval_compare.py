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
    assert summary["holyc_calibration"]["scored_count"] == 1
    assert summary["llama_calibration"]["scored_count"] == 1
    assert summary["holyc_rank_metrics"]["scored_count"] == 1
    assert summary["holyc_rank_metrics"]["top_1_accuracy"] == 1.0
    assert summary["holyc_rank_metrics"]["mean_reciprocal_rank"] == 1.0
    assert summary["holyc_margin_metrics"]["scored_count"] == 1
    assert summary["holyc_margin_metrics"]["score_coverage"] == 1 / 3
    assert summary["holyc_tie_metrics"]["scored_count"] == 1
    assert summary["holyc_tie_metrics"]["tie_rate"] == 0.0
    assert [item["dataset"] for item in summary["dataset_breakdown"]] == [
        "arc-smoke",
        "hellaswag-smoke",
        "truthfulqa-smoke",
    ]
    assert all(item["holyc_accuracy"] == 1.0 for item in summary["dataset_breakdown"])
    assert rows[0].holyc_confidence is not None
    assert rows[0].holyc_margin is not None


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
        breakdown_rows = list(csv.DictReader((Path(tmp) / "smoke_breakdown.csv").open(newline="", encoding="utf-8")))
        calibration_rows = list(
            csv.DictReader((Path(tmp) / "smoke_calibration_bins.csv").open(newline="", encoding="utf-8"))
        )
        margin_rows = list(csv.DictReader((Path(tmp) / "smoke_margins.csv").open(newline="", encoding="utf-8")))
        tie_rows = list(csv.DictReader((Path(tmp) / "smoke_score_ties.csv").open(newline="", encoding="utf-8")))
        disagreement_rows = list(
            csv.DictReader((Path(tmp) / "smoke_disagreements.csv").open(newline="", encoding="utf-8"))
        )
        junit_root = ET.parse(Path(tmp) / "smoke_junit.xml").getroot()
        assert payload["summary"]["record_count"] == 3
        assert payload["status"] == "pass"
        assert payload["regressions"] == []
        assert payload["summary"]["class_count"] == 4
        assert payload["summary"]["holyc_calibration"]["scored_count"] == 1
        assert payload["summary"]["llama_calibration"]["score_coverage"] == 1 / 3
        assert payload["summary"]["holyc_tie_metrics"]["tie_rate"] == 0.0
        holyc_interval = payload["summary"]["confidence_intervals"]["holyc_accuracy"]
        assert holyc_interval["method"] == "wilson"
        assert holyc_interval["successes"] == 3
        assert holyc_interval["total"] == 3
        assert round(holyc_interval["lower"], 4) == 0.4385
        assert holyc_interval["upper"] == 1.0
        assert payload["summary"]["holyc_per_answer_index"][0]["support"] == 3
        assert payload["summary"]["dataset_breakdown"][0]["dataset"] == "arc-smoke"
        assert payload["summary"]["dataset_breakdown"][0]["record_count"] == 1
        assert payload["summary"]["dataset_breakdown"][0]["mcnemar_exact"]["p_value"] == 1.0
        assert payload["max_mcnemar_loss_p"] is None
        assert (Path(tmp) / "smoke.md").exists()
        markdown = (Path(tmp) / "smoke.md").read_text(encoding="utf-8")
        assert "## Dataset Breakdown" in markdown
        assert "## Confidence Intervals" in markdown
        assert "## Score Calibration" in markdown
        assert "## Score Ranking" in markdown
        assert "## Score Margins" in markdown
        assert "## Score Ties" in markdown
        assert "No quality gate regressions." in markdown
        assert len(csv_rows) == 3
        assert len(breakdown_rows) == 3
        assert len(calibration_rows) == 20
        assert len(margin_rows) == 8
        assert len(tie_rows) == 8
        assert {row["engine"] for row in calibration_rows} == {"holyc", "llama"}
        assert margin_rows[0]["scope"] == "overall"
        assert margin_rows[0]["engine"] == "holyc"
        assert margin_rows[0]["scored_count"] == "1"
        assert tie_rows[0]["scope"] == "overall"
        assert tie_rows[0]["engine"] == "holyc"
        assert tie_rows[0]["tie_rate"] == "0.0"
        assert calibration_rows[0]["bin_index"] == "0"
        assert calibration_rows[-1]["bin_index"] == "9"
        assert breakdown_rows[0]["dataset"] == "arc-smoke"
        assert breakdown_rows[0]["record_count"] == "1"
        assert disagreement_rows == []
        assert csv_rows[0]["record_id"] == "smoke-arc-1"
        assert csv_rows[0]["holyc_correct"] == "True"
        assert csv_rows[0]["engines_agree"] == "True"
        assert csv_rows[0]["holyc_confidence"] != ""
        assert csv_rows[0]["holyc_gold_rank"] == "1"
        assert csv_rows[0]["holyc_top_score_tie_count"] == "1"
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
    assert summary["holyc_calibration"]["scored_count"] == 0
    assert summary["holyc_tie_metrics"]["scored_count"] == 0


def test_compare_reports_score_vector_calibration() -> None:
    gold = {
        "a": eval_compare.GoldCase("a", "unit", "validation", 0, ["A", "B"]),
        "b": eval_compare.GoldCase("b", "unit", "validation", 1, ["A", "B"]),
    }
    holyc = {
        "a": eval_compare.Prediction("a", 0, 0, [3.0, 1.0]),
        "b": eval_compare.Prediction("b", 0, 0, [2.0, 0.0]),
    }
    llama = {
        "a": eval_compare.Prediction("a", 0, 0, [4.0, 0.0]),
        "b": eval_compare.Prediction("b", 1, 1, [0.0, 4.0]),
    }

    rows, summary = eval_compare.compare(gold, holyc, llama)

    assert round(rows[0].holyc_confidence or 0.0, 4) == 0.8808
    assert round(rows[0].holyc_margin or 0.0, 4) == 0.7616
    assert summary["holyc_calibration"]["scored_count"] == 2
    assert summary["holyc_calibration"]["total_count"] == 2
    assert summary["holyc_calibration"]["score_coverage"] == 1.0
    assert round(summary["holyc_calibration"]["accuracy_when_scored"], 4) == 0.5
    assert summary["holyc_calibration"]["brier_score"] > summary["llama_calibration"]["brier_score"]
    assert summary["holyc_calibration"]["ece"] > 0.0
    assert rows[0].holyc_gold_rank == 1
    assert rows[1].holyc_gold_rank == 2
    assert summary["holyc_rank_metrics"]["top_1_accuracy"] == 0.5
    assert summary["holyc_rank_metrics"]["top_2_accuracy"] == 1.0
    assert summary["holyc_rank_metrics"]["mean_gold_rank"] == 1.5
    assert summary["holyc_rank_metrics"]["mean_reciprocal_rank"] == 0.75
    assert summary["llama_rank_metrics"]["mean_reciprocal_rank"] == 1.0
    assert summary["holyc_margin_metrics"]["scored_count"] == 2
    assert summary["holyc_margin_metrics"]["low_margin_threshold"] == 0.10
    assert round(summary["holyc_margin_metrics"]["mean_margin"], 4) == 0.7616
    assert summary["llama_margin_metrics"]["mean_margin"] > summary["holyc_margin_metrics"]["mean_margin"]
    assert summary["holyc_tie_metrics"]["tied_count"] == 0
    assert summary["holyc_tie_metrics"]["max_top_score_tie_count"] == 1


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


def test_score_vector_must_match_choice_count() -> None:
    gold = eval_compare.GoldCase("score-short", "unit", "validation", 0, ["A", "B", "C"])

    try:
        eval_compare.normalize_prediction(
            {"id": "score-short", "scores": [1.0, 0.5]},
            gold,
            Path("predictions.jsonl"),
            0,
        )
    except ValueError as exc:
        assert "scores length 2 does not match choice count 3" in str(exc)
    else:
        raise AssertionError("short score vector should fail")


def test_score_vector_rejects_non_finite_values() -> None:
    gold = eval_compare.GoldCase("score-nan", "unit", "validation", 0, ["A", "B"])

    try:
        eval_compare.normalize_prediction(
            {"id": "score-nan", "scores": [1.0, float("nan")]},
            gold,
            Path("predictions.jsonl"),
            0,
        )
    except ValueError as exc:
        assert "scores must contain only finite numbers" in str(exc)
    else:
        raise AssertionError("non-finite score vector should fail")


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


def test_cli_writes_disagreement_csv_for_engine_mismatches() -> None:
    gold = BENCH_PATH / "datasets" / "samples" / "smoke_eval.jsonl"
    llama = BENCH_PATH / "eval" / "samples" / "llama_smoke_predictions.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "wrong.jsonl"
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
                "mismatched",
            ]
        )
        rows = list(csv.DictReader((tmp_path / "mismatched_disagreements.csv").open(newline="", encoding="utf-8")))

        assert status == 0
        assert len(rows) == 1
        assert rows[0]["record_id"] == "smoke-arc-1"
        assert rows[0]["answer_index"] == "0"
        assert rows[0]["holyc_prediction"] == "1"
        assert rows[0]["llama_prediction"] == "0"
        assert rows[0]["holyc_correct"] == "False"
        assert rows[0]["llama_correct"] == "True"


def test_cli_can_gate_dataset_breakdown_regressions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gold = tmp_path / "gold.jsonl"
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        gold.write_text(
            "\n".join(
                [
                    '{"id":"a1","dataset":"alpha","split":"validation","prompt":"A1?","choices":["A","B"],"answer_index":0}',
                    '{"id":"a2","dataset":"alpha","split":"validation","prompt":"A2?","choices":["A","B"],"answer_index":0}',
                    '{"id":"a3","dataset":"alpha","split":"validation","prompt":"A3?","choices":["A","B"],"answer_index":0}',
                    '{"id":"b1","dataset":"beta","split":"validation","prompt":"B1?","choices":["A","B"],"answer_index":0}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        holyc.write_text(
            "\n".join(
                [
                    '{"id":"a1","prediction":0}',
                    '{"id":"a2","prediction":0}',
                    '{"id":"a3","prediction":0}',
                    '{"id":"b1","prediction":1}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        llama.write_text(
            "\n".join(
                [
                    '{"id":"a1","prediction":0}',
                    '{"id":"a2","prediction":0}',
                    '{"id":"a3","prediction":0}',
                    '{"id":"b1","prediction":0}',
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
                "mixed",
                "--split",
                "validation",
                "--output-dir",
                tmp,
                "--output-stem",
                "breakdown_gated",
                "--min-holyc-accuracy",
                "0.7",
                "--gate-dataset-breakdowns",
                "--fail-on-regression",
            ]
        )
        payload = json.loads((tmp_path / "breakdown_gated.json").read_text(encoding="utf-8"))
        junit_root = ET.parse(tmp_path / "breakdown_gated_junit.xml").getroot()

        assert status == 1
        assert payload["summary"]["holyc_accuracy"] == 0.75
        assert payload["status"] == "fail"
        assert len(payload["regressions"]) == 1
        assert payload["regressions"][0]["scope"] == "dataset_split"
        assert payload["regressions"][0]["dataset"] == "beta"
        assert junit_root.attrib["failures"] == "1"


def test_cli_can_gate_significant_paired_mcnemar_loss() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gold = tmp_path / "gold.jsonl"
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        gold.write_text(
            "\n".join(
                f'{{"id":"r{index}","dataset":"unit","split":"validation",'
                f'"prompt":"Q{index}?","choices":["A","B"],"answer_index":0}}'
                for index in range(5)
            )
            + "\n",
            encoding="utf-8",
        )
        holyc.write_text(
            "\n".join(f'{{"id":"r{index}","prediction":1}}' for index in range(5)) + "\n",
            encoding="utf-8",
        )
        llama.write_text(
            "\n".join(f'{{"id":"r{index}","prediction":0}}' for index in range(5)) + "\n",
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
                "unit",
                "--split",
                "validation",
                "--output-dir",
                tmp,
                "--output-stem",
                "mcnemar_gated",
                "--max-mcnemar-loss-p",
                "0.1",
                "--fail-on-regression",
            ]
        )
        payload = json.loads((tmp_path / "mcnemar_gated.json").read_text(encoding="utf-8"))
        junit_root = ET.parse(tmp_path / "mcnemar_gated_junit.xml").getroot()

        assert status == 1
        assert payload["max_mcnemar_loss_p"] == 0.1
        assert payload["summary"]["mcnemar_exact"]["p_value"] == 0.0625
        assert payload["regressions"][0]["metric"] == "mcnemar_exact_p_value"
        assert payload["regressions"][0]["value"] == 0.0625
        assert junit_root.attrib["failures"] == "1"


def test_cli_can_gate_holyc_margin_telemetry() -> None:
    gold = BENCH_PATH / "datasets" / "samples" / "smoke_eval.jsonl"
    holyc = BENCH_PATH / "eval" / "samples" / "holyc_smoke_predictions.jsonl"
    llama = BENCH_PATH / "eval" / "samples" / "llama_smoke_predictions.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
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
                "margin_gated",
                "--min-holyc-margin-coverage",
                "0.5",
                "--min-holyc-mean-margin",
                "0.9999",
                "--fail-on-regression",
            ]
        )
        payload = json.loads((Path(tmp) / "margin_gated.json").read_text(encoding="utf-8"))
        junit_root = ET.parse(Path(tmp) / "margin_gated_junit.xml").getroot()

        assert status == 1
        assert payload["min_holyc_margin_coverage"] == 0.5
        assert payload["min_holyc_mean_margin"] == 0.9999
        assert {row["metric"] for row in payload["regressions"]} == {
            "holyc_margin_score_coverage",
            "holyc_mean_margin",
        }
        assert junit_root.attrib["failures"] == "2"


def test_cli_can_gate_holyc_top_score_ties() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gold = tmp_path / "gold.jsonl"
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        gold.write_text(
            "\n".join(
                [
                    '{"id":"tie-a","dataset":"unit","split":"validation","prompt":"A?","choices":["A","B"],"answer_index":0}',
                    '{"id":"tie-b","dataset":"unit","split":"validation","prompt":"B?","choices":["A","B"],"answer_index":1}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        holyc.write_text(
            "\n".join(
                [
                    '{"id":"tie-a","scores":[1.0,1.0]}',
                    '{"id":"tie-b","scores":[0.0,2.0]}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        llama.write_text(
            "\n".join(
                [
                    '{"id":"tie-a","scores":[2.0,0.0]}',
                    '{"id":"tie-b","scores":[0.0,2.0]}',
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
                "unit",
                "--split",
                "validation",
                "--output-dir",
                tmp,
                "--output-stem",
                "tie_gated",
                "--max-holyc-score-tie-rate",
                "0.1",
                "--fail-on-regression",
            ]
        )
        payload = json.loads((tmp_path / "tie_gated.json").read_text(encoding="utf-8"))
        tie_rows = list(csv.DictReader((tmp_path / "tie_gated_score_ties.csv").open(newline="", encoding="utf-8")))
        junit_root = ET.parse(tmp_path / "tie_gated_junit.xml").getroot()

        assert status == 1
        assert payload["max_holyc_score_tie_rate"] == 0.1
        assert payload["summary"]["holyc_tie_metrics"]["tied_count"] == 1
        assert payload["summary"]["holyc_tie_metrics"]["tie_rate"] == 0.5
        assert payload["regressions"][0]["metric"] == "holyc_score_tie_rate"
        assert tie_rows[0]["tie_rate"] == "0.5"
        assert junit_root.attrib["failures"] == "1"


def test_mcnemar_gate_does_not_fail_when_holyc_wins() -> None:
    summary = {
        "accuracy_delta_holyc_minus_llama": 1.0,
        "agreement": 0.0,
        "holyc_accuracy": 1.0,
        "mcnemar_exact": eval_compare.exact_mcnemar_test(5, 0),
        "paired_correctness": {
            "both_correct": 0,
            "both_wrong": 0,
            "holyc_only_correct": 5,
            "llama_only_correct": 0,
        },
    }

    assert eval_compare.find_regressions(summary, max_mcnemar_loss_p=0.1) == []


if __name__ == "__main__":
    test_smoke_predictions_compare_cleanly()
    test_cli_writes_json_and_markdown_report()
    test_compare_reports_macro_f1_and_confusion_matrix()
    test_compare_reports_score_vector_calibration()
    test_confidence_level_can_be_configured()
    test_invalid_confidence_level_fails_fast()
    test_score_vector_must_match_choice_count()
    test_score_vector_rejects_non_finite_values()
    test_missing_prediction_fails_fast()
    test_cli_can_fail_on_quality_gate_regression()
    test_cli_writes_disagreement_csv_for_engine_mismatches()
    test_cli_can_gate_dataset_breakdown_regressions()
    test_cli_can_gate_significant_paired_mcnemar_loss()
    test_cli_can_gate_holyc_margin_telemetry()
    test_cli_can_gate_holyc_top_score_ties()
    test_mcnemar_gate_does_not_fail_when_holyc_wins()
    print("eval_compare_tests=ok")
