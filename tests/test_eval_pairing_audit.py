#!/usr/bin/env python3
"""Host-side checks for HolyC-vs-llama eval pairing audits."""

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

AUDIT_PATH = BENCH_PATH / "eval_pairing_audit.py"
spec = importlib.util.spec_from_file_location("eval_pairing_audit", AUDIT_PATH)
eval_pairing_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_pairing_audit"] = eval_pairing_audit
spec.loader.exec_module(eval_pairing_audit)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def rows() -> list[dict[str, object]]:
    return [
        {"id": "a", "dataset": "arc", "split": "validation", "model": "tiny", "quantization": "Q4_0", "prompt_sha256": "111", "prediction": "A"},
        {"id": "b", "dataset": "arc", "split": "validation", "model": "tiny", "quantization": "Q4_0", "prompt_sha256": "222", "scores": [0.1, 0.9]},
        {"id": "c", "dataset": "truthfulqa", "split": "validation", "model": "tiny", "quantization": "Q4_0", "prompt_sha256": "333", "prediction": 0},
    ]


def test_audit_accepts_matching_prediction_streams(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, rows())
    write_jsonl(llama, rows())

    report = eval_pairing_audit.audit_pairing(holyc, llama, min_records=3, require_same_order=True, require_predictions=True)

    assert report["status"] == "pass"
    assert report["summary"]["paired_records"] == 3
    assert report["findings"] == []
    assert report["pairs"][1]["holyc_prediction"] == "1"


def test_audit_flags_order_metadata_and_counterpart_mismatches(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, rows())
    llama_rows = [
        {**rows()[1], "quantization": "Q8_0"},
        {**rows()[0], "prompt_sha256": "changed"},
        {"id": "extra", "dataset": "arc", "split": "validation", "model": "tiny", "quantization": "Q4_0", "prediction": "B"},
    ]
    write_jsonl(llama, llama_rows)

    report = eval_pairing_audit.audit_pairing(holyc, llama, min_records=3, require_same_order=True, require_predictions=True)

    kinds = {finding["kind"] for finding in report["findings"]}
    assert report["status"] == "fail"
    assert {"order_mismatch", "metadata_mismatch", "missing_llama_record", "missing_holyc_record", "insufficient_paired_records"} <= kinds


def test_audit_compares_nested_identity_metadata(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    base_rows = [
        {
            "id": "a",
            "prediction": "A",
            "metadata": {
                "dataset": "arc",
                "split": "validation",
                "model": "tiny",
                "model_sha256": "model-a",
                "tokenizer_sha256": "tok-a",
                "quantization": "Q4_0",
                "prompt_template_sha256": "template-a",
                "input_sha256": "input-a",
            },
        },
        {"id": "b", "prediction": "B", "metadata": {"model_sha256": "model-a", "tokenizer_sha256": "tok-a", "quantization": "Q4_0"}},
        {"id": "c", "prediction": "C", "metadata": {"model_sha256": "model-a", "tokenizer_sha256": "tok-a", "quantization": "Q4_0"}},
    ]
    write_jsonl(holyc, base_rows)
    write_jsonl(llama, [{**base_rows[0], "metadata": {**base_rows[0]["metadata"], "tokenizer_sha256": "tok-b"}}, *base_rows[1:]])

    report = eval_pairing_audit.audit_pairing(holyc, llama, min_records=3, require_same_order=True, require_predictions=True)

    nested_findings = [finding for finding in report["findings"] if finding["field"] == "tokenizer_sha256"]
    assert report["status"] == "fail"
    assert len(nested_findings) == 1
    assert nested_findings[0]["kind"] == "metadata_mismatch"


def test_cli_writes_outputs(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    csv_path = tmp_path / "audit.csv"
    findings_csv = tmp_path / "findings.csv"
    junit = tmp_path / "audit.xml"
    write_jsonl(holyc, rows())
    write_jsonl(llama, rows())

    status = eval_pairing_audit.main(
        [
            "--holyc",
            str(holyc),
            "--llama",
            str(llama),
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
            "3",
            "--require-same-order",
            "--require-predictions",
            "--fail-on-findings",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    finding_rows = list(csv.DictReader(findings_csv.open(newline="", encoding="utf-8")))
    junit_root = ET.parse(junit).getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert len(csv_rows) == 3
    assert finding_rows == []
    assert "Eval Pairing Audit" in markdown.read_text(encoding="utf-8")
    assert junit_root.attrib["name"] == "holyc_eval_pairing_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-eval-pairing-audit-tests-") as tmp:
        tmp_path = Path(tmp)
        test_audit_accepts_matching_prediction_streams(tmp_path)
        test_audit_flags_order_metadata_and_counterpart_mismatches(tmp_path)
        test_audit_compares_nested_identity_metadata(tmp_path)
        test_cli_writes_outputs(tmp_path)
    print("eval_pairing_audit_tests=ok")
