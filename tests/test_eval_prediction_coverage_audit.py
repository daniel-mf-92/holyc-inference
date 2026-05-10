from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_prediction_coverage_audit


GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def rows() -> list[dict[str, object]]:
    return [
        {"id": "smoke-hellaswag-1", "prediction": 0},
        {"id": "smoke-arc-1", "prediction": 0},
        {"id": "smoke-truthfulqa-1", "prediction": 0},
    ]


def parse_args(extra: list[str], holyc: Path, llama: Path) -> object:
    return eval_prediction_coverage_audit.build_parser().parse_args(
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
            *extra,
        ]
    )


def test_build_report_accepts_full_prediction_coverage(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, rows())
    write_jsonl(llama, rows())

    report = eval_prediction_coverage_audit.build_report(
        parse_args(["--min-gold-records", "3", "--min-coverage-pct", "100"], holyc, llama)
    )

    assert report["status"] == "pass"
    assert report["summary"]["gold_records"] == 3
    assert report["summary"]["paired_coverage_pct"] == 100.0
    assert report["findings"] == []


def test_build_report_flags_missing_extra_and_duplicate_predictions(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, [rows()[0], rows()[0], {"id": "extra", "prediction": 1}])
    write_jsonl(llama, rows()[:2])

    report = eval_prediction_coverage_audit.build_report(
        parse_args(["--min-coverage-pct", "100", "--min-slice-coverage-pct", "100"], holyc, llama)
    )
    kinds = {finding["kind"] for finding in report["findings"]}

    assert report["status"] == "fail"
    assert {"duplicate_id", "extra_id", "missing_id", "min_coverage_pct", "min_slice_coverage_pct"} <= kinds
    assert report["summary"]["paired_records"] == 1


def test_cli_writes_report_artifacts(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(holyc, rows())
    write_jsonl(llama, rows())

    status = eval_prediction_coverage_audit.main(
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
            "coverage",
            "--fail-on-findings",
        ]
    )

    payload = json.loads((output_dir / "coverage.json").read_text(encoding="utf-8"))
    rows_csv = list(csv.DictReader((output_dir / "coverage.csv").open(encoding="utf-8")))
    findings_csv = list(csv.DictReader((output_dir / "coverage_findings.csv").open(encoding="utf-8")))
    junit_root = ET.parse(output_dir / "coverage_junit.xml").getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert len(rows_csv) == 3
    assert findings_csv == []
    assert "Eval Prediction Coverage Audit" in (output_dir / "coverage.md").read_text(encoding="utf-8")
    assert junit_root.attrib["name"] == "holyc_eval_prediction_coverage_audit"
    assert junit_root.attrib["failures"] == "0"
