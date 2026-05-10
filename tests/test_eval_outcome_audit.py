#!/usr/bin/env python3
"""Host-side checks for eval outcome bucket audits."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

AUDIT_PATH = BENCH_PATH / "eval_outcome_audit.py"
spec = importlib.util.spec_from_file_location("eval_outcome_audit", AUDIT_PATH)
eval_outcome_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_outcome_audit"] = eval_outcome_audit
spec.loader.exec_module(eval_outcome_audit)


def row(record_id: str, dataset: str, holyc: int, llama: int, answer: int) -> dict[str, object]:
    return {
        "record_id": record_id,
        "dataset": dataset,
        "split": "validation",
        "answer_index": answer,
        "holyc_prediction": holyc,
        "llama_prediction": llama,
        "holyc_correct": holyc == answer,
        "llama_correct": llama == answer,
        "engines_agree": holyc == llama,
    }


def write_eval_report(path: Path) -> None:
    payload = {
        "dataset": "smoke-eval",
        "split": "validation",
        "rows": [
            row("a", "arc-smoke", 0, 0, 0),
            row("b", "arc-smoke", 0, 1, 0),
            row("c", "truthfulqa-smoke", 1, 0, 0),
            row("d", "truthfulqa-smoke", 2, 3, 0),
        ],
        "summary": {"record_count": 4},
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_audit_reports_outcome_buckets(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    write_eval_report(report_path)

    report = eval_outcome_audit.audit_reports(
        [report_path],
        min_records=4,
        max_llama_only_correct_pct=25,
        max_dataset_split_llama_only_correct_pct=50,
        max_both_wrong_pct=50,
    )

    overall = report["scopes"][0]
    assert report["status"] == "pass"
    assert report["scope_count"] == 3
    assert overall["both_correct"] == 1
    assert overall["holyc_only_correct"] == 1
    assert overall["llama_only_correct"] == 1
    assert overall["both_wrong_disagree"] == 1
    assert overall["llama_only_correct_pct"] == 25.0


def test_audit_gates_llama_only_and_both_wrong_rates(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    write_eval_report(report_path)

    report = eval_outcome_audit.audit_reports(
        [report_path],
        min_records=5,
        max_llama_only_correct_pct=10,
        max_dataset_split_llama_only_correct_pct=10,
        max_both_wrong_pct=10,
    )

    kinds = [finding["kind"] for finding in report["findings"]]
    assert report["status"] == "fail"
    assert "insufficient_records" in kinds
    assert kinds.count("llama_only_correct_pct_exceeded") == 2
    assert "both_wrong_pct_exceeded" in kinds


def test_cli_writes_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    csv_path = tmp_path / "audit.csv"
    findings_csv = tmp_path / "findings.csv"
    junit = tmp_path / "audit.xml"
    write_eval_report(report_path)

    status = eval_outcome_audit.main(
        [
            str(report_path),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--findings-csv",
            str(findings_csv),
            "--junit",
            str(junit),
            "--min-records",
            "4",
            "--max-llama-only-correct-pct",
            "25",
            "--max-dataset-split-llama-only-correct-pct",
            "50",
            "--max-both-wrong-pct",
            "50",
            "--fail-on-findings",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    finding_rows = list(csv.DictReader(findings_csv.open(newline="", encoding="utf-8")))
    junit_root = ET.parse(junit).getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert len(rows) == 3
    assert rows[0]["scope"] == "overall"
    assert finding_rows == []
    assert "Eval Outcome Audit" in markdown.read_text(encoding="utf-8")
    assert junit_root.attrib["name"] == "holyc_eval_outcome_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-eval-outcome-audit-tests-") as tmp:
        tmp_path = Path(tmp)
        test_audit_reports_outcome_buckets(tmp_path)
        test_audit_gates_llama_only_and_both_wrong_rates(tmp_path)
        test_cli_writes_outputs(tmp_path)
    print("eval_outcome_audit_tests=ok")
