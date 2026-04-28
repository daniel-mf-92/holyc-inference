#!/usr/bin/env python3
"""Host-side checks for .hceval dataset inspection."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

INSPECT_PATH = BENCH_PATH / "hceval_inspect.py"
spec = importlib.util.spec_from_file_location("hceval_inspect", INSPECT_PATH)
hceval_inspect = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["hceval_inspect"] = hceval_inspect
spec.loader.exec_module(hceval_inspect)


def test_sample_hceval_parses_and_validates() -> None:
    dataset = hceval_inspect.parse_hceval(ROOT / "bench" / "results" / "datasets" / "smoke_eval.hceval")
    findings = hceval_inspect.validate_dataset(
        dataset,
        ROOT / "bench" / "results" / "datasets" / "smoke_eval.manifest.json",
    )

    assert findings == []
    assert dataset.metadata["format"] == "hceval-mc"
    assert [record.record_id for record in dataset.records] == [
        "smoke-hellaswag-1",
        "smoke-arc-1",
        "smoke-truthfulqa-1",
    ]
    assert dataset.records[0].offset > hceval_inspect.dataset_pack.HEADER.size
    assert all(record.length > hceval_inspect.dataset_pack.RECORD_HEADER.size for record in dataset.records)


def test_cli_writes_json_and_markdown() -> None:
    input_path = ROOT / "bench" / "results" / "datasets" / "smoke_eval.hceval"
    manifest_path = ROOT / "bench" / "results" / "datasets" / "smoke_eval.manifest.json"
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "inspect.json"
        markdown = Path(tmp) / "inspect.md"
        status = hceval_inspect.main(
            [
                "--input",
                str(input_path),
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--markdown",
                str(markdown),
            ]
        )

        assert status == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "pass"
        assert report["record_count"] == 3
        assert report["record_spans"][0]["record_id"] == "smoke-hellaswag-1"
        assert report["record_spans"][-1]["offset"] + report["record_spans"][-1]["length"] == input_path.stat().st_size
        assert report["byte_stats"]["max_prompt_bytes"] > 0
        assert "HCEval Dataset Inspection" in markdown.read_text(encoding="utf-8")


def test_manifest_record_spans_are_verified() -> None:
    input_path = ROOT / "bench" / "results" / "datasets" / "smoke_eval.hceval"
    manifest_path = ROOT / "bench" / "results" / "datasets" / "smoke_eval.manifest.json"
    with tempfile.TemporaryDirectory() as tmp:
        bad_manifest = Path(tmp) / "bad.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["record_spans"] = [
            {
                "record_id": "smoke-hellaswag-1",
                "offset": 1,
                "length": 1,
                "payload_bytes": 1,
            }
        ]
        bad_manifest.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

        dataset = hceval_inspect.parse_hceval(input_path)
        findings = hceval_inspect.validate_dataset(dataset, bad_manifest)

        assert "manifest record_spans does not match parsed binary" in findings


def test_cli_size_gates_report_findings() -> None:
    input_path = ROOT / "bench" / "results" / "datasets" / "smoke_eval.hceval"
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "inspect.json"
        status = hceval_inspect.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output),
                "--max-choice-bytes",
                "4",
            ]
        )

        assert status == 1
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "fail"
        assert any("choice is" in finding for finding in report["findings"])


def test_truncated_payload_fails_fast() -> None:
    source = ROOT / "bench" / "results" / "datasets" / "smoke_eval.hceval"
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.hceval"
        bad.write_bytes(source.read_bytes()[:-5])
        try:
            hceval_inspect.parse_hceval(bad)
        except ValueError as exc:
            assert "truncated" in str(exc) or "trailing" in str(exc)
        else:
            raise AssertionError("truncated hceval payload should fail")


if __name__ == "__main__":
    test_sample_hceval_parses_and_validates()
    test_cli_writes_json_and_markdown()
    test_manifest_record_spans_are_verified()
    test_cli_size_gates_report_findings()
    test_truncated_payload_fails_fast()
    print("hceval_inspect_tests=ok")
