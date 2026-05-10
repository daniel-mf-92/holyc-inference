from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

AUDIT_PATH = BENCH_PATH / "eval_confusion_audit.py"
spec = importlib.util.spec_from_file_location("eval_confusion_audit", AUDIT_PATH)
eval_confusion_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_confusion_audit"] = eval_confusion_audit
spec.loader.exec_module(eval_confusion_audit)


def report_payload(*, holyc_macro: float, llama_macro: float, holyc_recall: float = 0.9) -> dict[str, object]:
    def rows(recall: float, f1: float) -> list[dict[str, object]]:
        return [
            {
                "answer_index": 0,
                "label": "A",
                "support": 4,
                "true_positive": 3,
                "false_positive": 1,
                "false_negative": 1,
                "precision": 0.75,
                "recall": recall,
                "f1": f1,
            },
            {
                "answer_index": 1,
                "label": "B",
                "support": 4,
                "true_positive": 4,
                "false_positive": 0,
                "false_negative": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": f1,
            },
        ]

    return {
        "dataset": "unit-eval",
        "split": "validation",
        "summary": {
            "record_count": 8,
            "holyc_accuracy": holyc_macro,
            "holyc_macro_f1": holyc_macro,
            "holyc_per_answer_index": rows(holyc_recall, holyc_macro),
            "llama_accuracy": llama_macro,
            "llama_macro_f1": llama_macro,
            "llama_per_answer_index": rows(1.0, llama_macro),
        },
    }


def args(**overrides: object) -> Namespace:
    defaults = {
        "min_macro_f1": None,
        "min_accuracy": None,
        "min_class_support": None,
        "min_class_precision": None,
        "min_class_recall": None,
        "min_class_f1": None,
        "max_holyc_macro_f1_loss": None,
        "include_dataset_breakdown": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_audit_reports_passing_scope_and_class_summaries(tmp_path: Path) -> None:
    report = tmp_path / "eval.json"
    report.write_text(json.dumps(report_payload(holyc_macro=0.9, llama_macro=0.91)) + "\n", encoding="utf-8")

    payload = eval_confusion_audit.audit_reports(
        [report],
        args(min_macro_f1=0.85, min_class_support=4, min_class_recall=0.8, max_holyc_macro_f1_loss=0.05),
    )

    assert payload["status"] == "pass"
    assert payload["scope_summary_count"] == 2
    assert payload["class_summary_count"] == 4
    assert payload["findings"] == []
    assert payload["scope_summaries"][0]["engine"] == "holyc"


def test_audit_flags_macro_and_class_regressions(tmp_path: Path) -> None:
    report = tmp_path / "eval.json"
    report.write_text(json.dumps(report_payload(holyc_macro=0.5, llama_macro=0.9, holyc_recall=0.25)) + "\n", encoding="utf-8")

    payload = eval_confusion_audit.audit_reports(
        [report],
        args(min_macro_f1=0.8, min_class_recall=0.8, max_holyc_macro_f1_loss=0.1),
    )
    metrics = {finding["metric"] for finding in payload["findings"]}

    assert payload["status"] == "fail"
    assert "macro_f1" in metrics
    assert "class_recall" in metrics
    assert "macro_f1_loss_vs_llama" in metrics


def test_cli_writes_reports_and_junit(tmp_path: Path) -> None:
    report = tmp_path / "eval.json"
    output_dir = tmp_path / "out"
    report.write_text(json.dumps(report_payload(holyc_macro=0.9, llama_macro=0.91)) + "\n", encoding="utf-8")

    status = eval_confusion_audit.main(
        [
            str(report),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "confusion",
            "--min-macro-f1",
            "0.85",
            "--fail-on-findings",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "confusion.json").read_text(encoding="utf-8"))
    junit = ET.parse(output_dir / "confusion_junit.xml").getroot()
    assert payload["status"] == "pass"
    assert (output_dir / "confusion.csv").exists()
    assert (output_dir / "confusion_scopes.csv").exists()
    assert (output_dir / "confusion_classes.csv").exists()
    assert "Eval Confusion Audit" in (output_dir / "confusion.md").read_text(encoding="utf-8")
    assert junit.attrib["name"] == "holyc_eval_confusion_audit"
    assert junit.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-eval-confusion-test-") as tmp:
        tmp_path = Path(tmp) / "pass"
        tmp_path.mkdir()
        test_audit_reports_passing_scope_and_class_summaries(tmp_path)
    with tempfile.TemporaryDirectory(prefix="holyc-eval-confusion-test-") as tmp:
        tmp_path = Path(tmp) / "fail"
        tmp_path.mkdir()
        test_audit_flags_macro_and_class_regressions(tmp_path)
    with tempfile.TemporaryDirectory(prefix="holyc-eval-confusion-test-") as tmp:
        tmp_path = Path(tmp) / "cli"
        tmp_path.mkdir()
        test_cli_writes_reports_and_junit(tmp_path)
    print("eval_confusion_audit_tests=ok")
