from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_baseline
import eval_compare


def test_build_rows_reports_majority_and_random_baselines() -> None:
    gold = eval_compare.load_gold(ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl", "smoke-eval", "validation")
    rows = eval_baseline.build_rows(gold)

    assert rows[0].scope == "overall"
    assert rows[0].records == 3
    assert rows[0].majority_answer_index == 0
    assert rows[0].majority_accuracy == 1.0
    assert rows[0].random_expected_accuracy == 0.25
    assert {row.dataset for row in rows[1:]} == {"arc-smoke", "hellaswag-smoke", "truthfulqa-smoke"}


def test_eval_baseline_cli_writes_reports_and_gates() -> None:
    gold = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"
    with tempfile.TemporaryDirectory(prefix="holyc-eval-baseline-test-") as tmp:
        output_dir = Path(tmp) / "out"
        status = eval_baseline.main(
            [
                "--gold",
                str(gold),
                "--dataset",
                "smoke-eval",
                "--split",
                "validation",
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "baseline",
                "--min-records",
                "3",
            ]
        )
        assert status == 0
        payload = json.loads((output_dir / "baseline.json").read_text(encoding="utf-8"))
        csv_rows = list(csv.DictReader((output_dir / "baseline.csv").open(newline="", encoding="utf-8")))
        junit = ET.parse(output_dir / "baseline_junit.xml").getroot()
        assert payload["status"] == "pass"
        assert payload["summary"]["records"] == 3
        assert len(csv_rows) == 4
        assert "No baseline gate findings." in (output_dir / "baseline.md").read_text(encoding="utf-8")
        assert junit.attrib["name"] == "holyc_eval_baseline"

        failed = eval_baseline.main(
            [
                "--gold",
                str(gold),
                "--dataset",
                "smoke-eval",
                "--split",
                "validation",
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "baseline_fail",
                "--max-majority-accuracy",
                "0.5",
            ]
        )
        assert failed == 1
        fail_payload = json.loads((output_dir / "baseline_fail.json").read_text(encoding="utf-8"))
        assert fail_payload["findings"][0]["gate"] == "max_majority_accuracy"


if __name__ == "__main__":
    test_build_rows_reports_majority_and_random_baselines()
    test_eval_baseline_cli_writes_reports_and_gates()
