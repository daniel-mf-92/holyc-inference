#!/usr/bin/env python3
"""Host-side checks for prompt suite parity comparison."""

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

COMPARE_PATH = BENCH_PATH / "prompt_suite_compare.py"
spec = importlib.util.spec_from_file_location("prompt_suite_compare", COMPARE_PATH)
prompt_suite_compare = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["prompt_suite_compare"] = prompt_suite_compare
spec.loader.exec_module(prompt_suite_compare)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_identical_prompt_suites_pass(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    rows = [
        {"prompt_id": "one", "prompt": "A", "expected_tokens": 4},
        {"prompt_id": "two", "prompt": "B", "expected_tokens": 8},
    ]
    write_jsonl(baseline, rows)
    write_jsonl(candidate, rows)

    report = prompt_suite_compare.compare_suites(baseline, candidate)

    assert report["status"] == "pass"
    assert report["baseline_prompt_count"] == 2
    assert report["candidate_prompt_count"] == 2
    assert report["compared_prompt_count"] == 2
    assert report["baseline_suite_sha256"] == report["candidate_suite_sha256"]
    assert report["findings"] == []


def test_prompt_text_order_and_expected_tokens_mismatches_fail(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(
        baseline,
        [
            {"prompt_id": "one", "prompt": "A", "expected_tokens": 4},
            {"prompt_id": "two", "prompt": "B", "expected_tokens": 8},
        ],
    )
    write_jsonl(
        candidate,
        [
            {"prompt_id": "two", "prompt": "B", "expected_tokens": 8},
            {"prompt_id": "one", "prompt": "changed", "expected_tokens": 5},
        ],
    )

    report = prompt_suite_compare.compare_suites(baseline, candidate)

    kinds = [finding["kind"] for finding in report["findings"]]
    assert report["status"] == "fail"
    assert "prompt_order_mismatch" in kinds
    assert "prompt_text_mismatch" in kinds
    assert "prompt_byte_mismatch" in kinds
    assert "expected_tokens_mismatch" in kinds
    assert "prompt_index_mismatch" in kinds


def test_ignore_order_allows_reordered_equal_prompts(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(
        baseline,
        [
            {"prompt_id": "one", "prompt": "A"},
            {"prompt_id": "two", "prompt": "B"},
        ],
    )
    write_jsonl(
        candidate,
        [
            {"prompt_id": "two", "prompt": "B"},
            {"prompt_id": "one", "prompt": "A"},
        ],
    )

    report = prompt_suite_compare.compare_suites(baseline, candidate, ignore_order=True)

    assert report["status"] == "pass"
    assert report["findings"] == []


def test_cli_writes_json_csv_markdown_and_junit(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "compare.json"
    markdown = tmp_path / "compare.md"
    csv_path = tmp_path / "compare.csv"
    junit = tmp_path / "compare.xml"
    write_jsonl(baseline, [{"prompt_id": "one", "prompt": "A", "expected_tokens": 4}])
    write_jsonl(candidate, [{"prompt_id": "one", "prompt": "A", "expected_tokens": 4}])

    status = prompt_suite_compare.main(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--junit",
            str(junit),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    junit_root = ET.parse(junit).getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert "Prompt Suite Compare" in markdown.read_text(encoding="utf-8")
    assert rows[0]["row_type"] == "prompt"
    assert rows[0]["status"] == "pass"
    assert junit_root.attrib["name"] == "holyc_prompt_suite_compare"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-prompt-suite-compare-tests-") as tmp:
        tmp_path = Path(tmp)
        test_identical_prompt_suites_pass(tmp_path)
        test_prompt_text_order_and_expected_tokens_mismatches_fail(tmp_path)
        test_ignore_order_allows_reordered_equal_prompts(tmp_path)
        test_cli_writes_json_csv_markdown_and_junit(tmp_path)
    print("prompt_suite_compare_tests=ok")
