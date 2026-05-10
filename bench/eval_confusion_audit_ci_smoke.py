#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval confusion audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def class_row(answer_index: int, support: int, precision: float, recall: float, f1: float) -> dict[str, object]:
    return {
        "answer_index": answer_index,
        "label": chr(ord("A") + answer_index),
        "support": support,
        "true_positive": int(round(support * recall)),
        "false_positive": 0 if precision == 1.0 else 1,
        "false_negative": support - int(round(support * recall)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_report(path: Path, *, holyc_macro: float, llama_macro: float, holyc_recall: float = 0.9) -> None:
    holyc_classes = [
        class_row(0, 5, 0.9, holyc_recall, holyc_macro),
        class_row(1, 5, 0.9, holyc_recall, holyc_macro),
    ]
    llama_classes = [
        class_row(0, 5, 0.95, 0.95, llama_macro),
        class_row(1, 5, 0.95, 0.95, llama_macro),
    ]
    path.write_text(
        json.dumps(
            {
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "summary": {
                    "record_count": 10,
                    "holyc_accuracy": holyc_macro,
                    "holyc_macro_f1": holyc_macro,
                    "holyc_per_answer_index": holyc_classes,
                    "llama_accuracy": llama_macro,
                    "llama_macro_f1": llama_macro,
                    "llama_per_answer_index": llama_classes,
                    "dataset_breakdown": [
                        {
                            "dataset": "arc-smoke",
                            "split": "validation",
                            "record_count": 10,
                            "holyc_accuracy": holyc_macro,
                            "holyc_macro_f1": holyc_macro,
                            "holyc_per_answer_index": holyc_classes,
                            "llama_accuracy": llama_macro,
                            "llama_macro_f1": llama_macro,
                            "llama_per_answer_index": llama_classes,
                        }
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_audit(output_dir: Path, report: Path, stem: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "eval_confusion_audit.py"),
            str(report),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            stem,
            *extra_args,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-confusion-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing_report = tmp_path / "eval_compare_pass.json"
        write_report(passing_report, holyc_macro=0.9, llama_macro=0.92)
        passed = run_audit(
            tmp_path,
            passing_report,
            "confusion_pass",
            "--min-macro-f1",
            "0.85",
            "--min-class-support",
            "5",
            "--min-class-recall",
            "0.85",
            "--max-holyc-macro-f1-loss",
            "0.05",
            "--fail-on-findings",
        )
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode

        payload = json.loads((tmp_path / "confusion_pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(payload["scope_summary_count"] == 4, "unexpected_scope_summary_count"):
            return rc
        if rc := require(payload["class_summary_count"] == 8, "unexpected_class_summary_count"):
            return rc
        if rc := require((tmp_path / "confusion_pass.csv").read_text(encoding="utf-8").startswith("severity,"), "missing_findings_csv_header"):
            return rc
        if rc := require((tmp_path / "confusion_pass_classes.csv").read_text(encoding="utf-8").startswith("source,"), "missing_classes_csv_header"):
            return rc
        junit = ET.parse(tmp_path / "confusion_pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_confusion_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_pass_junit_failure"):
            return rc

        failing_report = tmp_path / "eval_compare_fail.json"
        write_report(failing_report, holyc_macro=0.55, llama_macro=0.9, holyc_recall=0.4)
        failed = run_audit(
            tmp_path,
            failing_report,
            "confusion_fail",
            "--min-macro-f1",
            "0.8",
            "--min-class-recall",
            "0.8",
            "--max-holyc-macro-f1-loss",
            "0.1",
            "--include-dataset-breakdown",
            "--fail-on-findings",
        )
        if rc := require(failed.returncode == 2, "expected_failure_status"):
            return rc
        failed_payload = json.loads((tmp_path / "confusion_fail.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in failed_payload["findings"]}
        if rc := require("macro_f1" in metrics, "missing_macro_f1_finding"):
            return rc
        if rc := require("class_recall" in metrics, "missing_class_recall_finding"):
            return rc
        if rc := require("macro_f1_loss_vs_llama" in metrics, "missing_macro_loss_finding"):
            return rc
        failed_junit = ET.parse(tmp_path / "confusion_fail_junit.xml").getroot()
        if rc := require(failed_junit.attrib.get("failures") == "1", "unexpected_failed_junit"):
            return rc

    print("eval_confusion_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
