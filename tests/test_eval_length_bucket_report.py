from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_compare
import eval_length_bucket_report


def write_gold(path: Path) -> None:
    rows = [
        {
            "id": "short",
            "ctx": "Short prompt",
            "endings": ["right", "wrong", "bad", "no"],
            "label": "0",
        },
        {
            "id": "long",
            "question": "Which answer is correct? " + ("context " * 20),
            "choices": [
                {"label": "A", "text": "wrong"},
                {"label": "B", "text": "right"},
                {"label": "C", "text": "bad"},
                {"label": "D", "text": "no"},
            ],
            "answerKey": "B",
        },
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_bucket_rows_capture_accuracy_by_prompt_size(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    holyc_path = tmp_path / "holyc.jsonl"
    llama_path = tmp_path / "llama.jsonl"
    write_gold(gold_path)
    write_predictions(holyc_path, [{"id": "short", "prediction": 0}, {"id": "long", "prediction": 1}])
    write_predictions(llama_path, [{"id": "short", "prediction": 1}, {"id": "long", "prediction": 1}])

    gold = eval_compare.load_gold(gold_path, "smoke", "validation")
    holyc = eval_compare.load_predictions(holyc_path, gold)
    llama = eval_compare.load_predictions(llama_path, gold)
    rows = eval_length_bucket_report.build_bucket_rows(gold, holyc, llama, [64])

    assert [row.bucket for row in rows] == ["0-64", "65+"]
    assert rows[0].holyc_accuracy == 1.0
    assert rows[0].llama_accuracy == 0.0
    assert rows[1].holyc_accuracy_delta_vs_llama == 0.0


def test_audit_flags_low_count_accuracy_and_llama_loss() -> None:
    row = eval_length_bucket_report.BucketRow(
        bucket="0-128",
        min_prompt_bytes=0,
        max_prompt_bytes=128,
        record_count=1,
        holyc_correct=0,
        llama_correct=1,
        both_correct=0,
        holyc_only_correct=0,
        llama_only_correct=1,
        both_wrong=0,
        agreement_count=0,
        holyc_accuracy=0.0,
        llama_accuracy=1.0,
        agreement=0.0,
        holyc_accuracy_delta_vs_llama=-1.0,
    )

    findings = eval_length_bucket_report.audit_rows(
        [row],
        min_records_per_bucket=2,
        min_holyc_accuracy=0.5,
        max_holyc_accuracy_loss=0.25,
    )

    assert {finding.metric for finding in findings} == {
        "record_count",
        "holyc_accuracy",
        "holyc_accuracy_loss",
    }


def test_cli_writes_reports_and_junit(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    holyc_path = tmp_path / "holyc.jsonl"
    llama_path = tmp_path / "llama.jsonl"
    output_dir = tmp_path / "out"
    write_gold(gold_path)
    write_predictions(holyc_path, [{"id": "short", "prediction": 0}, {"id": "long", "prediction": 1}])
    write_predictions(llama_path, [{"id": "short", "prediction": 0}, {"id": "long", "prediction": 1}])

    status = eval_length_bucket_report.main(
        [
            "--gold",
            str(gold_path),
            "--holyc",
            str(holyc_path),
            "--llama",
            str(llama_path),
            "--dataset",
            "smoke",
            "--split",
            "validation",
            "--bucket-edges",
            "64",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "length",
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "length.json").read_text(encoding="utf-8"))
    assert report["summary"]["paired_records"] == 2
    assert "Eval Length Bucket Report" in (output_dir / "length.md").read_text(encoding="utf-8")
    assert "holyc_accuracy" in (output_dir / "length.csv").read_text(encoding="utf-8")
    assert "severity" in (output_dir / "length_findings.csv").read_text(encoding="utf-8")
    junit = ET.parse(output_dir / "length_junit.xml").getroot()
    assert junit.attrib["failures"] == "0"
