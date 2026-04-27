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
        assert "HCEval Dataset Inspection" in markdown.read_text(encoding="utf-8")


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
    test_truncated_payload_fails_fast()
    print("hceval_inspect_tests=ok")
