#!/usr/bin/env python3
"""Host-side checks for the offline eval input audit."""

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

AUDIT_PATH = BENCH_PATH / "eval_input_audit.py"
spec = importlib.util.spec_from_file_location("eval_input_audit", AUDIT_PATH)
eval_input_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_input_audit"] = eval_input_audit
spec.loader.exec_module(eval_input_audit)


def smoke_args(output_dir: Path, output_stem: str = "audit") -> list[str]:
    return [
        "--gold",
        str(BENCH_PATH / "datasets" / "samples" / "smoke_eval.jsonl"),
        "--holyc",
        str(BENCH_PATH / "eval" / "samples" / "holyc_smoke_predictions.jsonl"),
        "--llama",
        str(BENCH_PATH / "eval" / "samples" / "llama_smoke_predictions.jsonl"),
        "--dataset",
        "smoke-eval",
        "--split",
        "validation",
        "--model",
        "synthetic-smoke",
        "--quantization",
        "Q4_0",
        "--output-dir",
        str(output_dir),
        "--output-stem",
        output_stem,
    ]


def test_smoke_eval_inputs_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        assert eval_input_audit.main(smoke_args(tmp_path)) == 0
        report = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))

        assert report["status"] == "pass"
        assert report["gold_record_count"] == 3
        assert report["summary"]["holyc_coverage"] == 3
        assert report["summary"]["llama_coverage"] == 3
        assert report["choice_count_histogram"] == {"4": 3}
        assert (tmp_path / "audit.md").exists()
        assert list(csv.DictReader((tmp_path / "audit.csv").open(newline="", encoding="utf-8"))) == []
        junit_root = ET.parse(tmp_path / "audit_junit.xml").getroot()
        assert junit_root.attrib["name"] == "holyc_eval_input_audit"
        assert junit_root.attrib["failures"] == "0"


def test_missing_prediction_fails_with_structured_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc_missing.jsonl"
        holyc.write_text('{"id":"smoke-hellaswag-1","prediction":0}\n', encoding="utf-8")

        args = smoke_args(tmp_path, "missing")
        args[args.index(str(BENCH_PATH / "eval" / "samples" / "holyc_smoke_predictions.jsonl"))] = str(holyc)
        assert eval_input_audit.main(args) == 2

        report = json.loads((tmp_path / "missing.json").read_text(encoding="utf-8"))
        csv_rows = list(csv.DictReader((tmp_path / "missing.csv").open(newline="", encoding="utf-8")))
        junit_root = ET.parse(tmp_path / "missing_junit.xml").getroot()
        messages = [issue["message"] for issue in report["issues"]]
        assert report["status"] == "fail"
        assert len(csv_rows) == report["summary"]["errors"]
        assert junit_root.attrib["failures"] == "1"
        assert any("missing prediction id 'smoke-arc-1'" in message for message in messages)
        assert any("missing prediction id 'smoke-truthfulqa-1'" in message for message in messages)


def test_prediction_metadata_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc_meta.jsonl"
        holyc.write_text(
            "\n".join(
                [
                    '{"id":"smoke-hellaswag-1","prediction":0,"model":"wrong","quantization":"Q8_0"}',
                    '{"id":"smoke-arc-1","prediction":"A","model":"wrong","quantization":"Q8_0"}',
                    '{"id":"smoke-truthfulqa-1","prediction":"A","model":"wrong","quantization":"Q8_0"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        args = smoke_args(tmp_path, "metadata")
        args[args.index(str(BENCH_PATH / "eval" / "samples" / "holyc_smoke_predictions.jsonl"))] = str(holyc)
        assert eval_input_audit.main(args) == 2

        report = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        messages = [issue["message"] for issue in report["issues"]]
        assert any("model metadata ['wrong'] does not match expected 'synthetic-smoke'" in message for message in messages)
        assert any("quantization metadata ['Q8_0'] does not match expected 'Q4_0'" in message for message in messages)


if __name__ == "__main__":
    test_smoke_eval_inputs_pass()
    test_missing_prediction_fails_with_structured_report()
    test_prediction_metadata_mismatch_fails()
    print("eval_input_audit_tests=ok")
