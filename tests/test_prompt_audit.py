#!/usr/bin/env python3
"""Host-side checks for benchmark prompt audit tooling."""

from __future__ import annotations

import importlib.util
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

AUDIT_PATH = BENCH_PATH / "prompt_audit.py"
spec = importlib.util.spec_from_file_location("prompt_audit", AUDIT_PATH)
prompt_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["prompt_audit"] = prompt_audit
spec.loader.exec_module(prompt_audit)


def test_smoke_prompt_audit_passes() -> None:
    prompts = BENCH_PATH / "prompts" / "smoke.jsonl"

    report = prompt_audit.build_report(
        prompts,
        min_prompts=2,
        max_prompt_bytes=1024,
        fail_on_duplicate_text=False,
    )

    assert report["status"] == "pass"
    assert report["summary"]["prompt_count"] == 2
    assert len(report["summary"]["suite_sha256"]) == 64
    assert report["summary"]["bytes_max"] <= 1024
    assert report["summary"]["expected_token_prompts"] == 2
    assert report["summary"]["expected_tokens_total"] == 80
    assert {row["prompt_id"] for row in report["prompts"]} == {"smoke-short", "smoke-code"}


def test_duplicate_prompt_ids_fail(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt_id":"same","prompt":"first"}\n{"prompt_id":"same","prompt":"second"}\n',
        encoding="utf-8",
    )

    report = prompt_audit.build_report(
        prompts,
        min_prompts=1,
        max_prompt_bytes=None,
        fail_on_duplicate_text=False,
    )

    assert report["status"] == "fail"
    assert report["error_count"] == 1
    assert "duplicate prompt id" in report["issues"][0]["message"]


def test_duplicate_prompt_text_can_be_warning_or_error(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt_id":"a","prompt":"same text"}\n{"prompt_id":"b","prompt":"same text"}\n',
        encoding="utf-8",
    )

    warning_report = prompt_audit.build_report(
        prompts,
        min_prompts=1,
        max_prompt_bytes=None,
        fail_on_duplicate_text=False,
    )
    error_report = prompt_audit.build_report(
        prompts,
        min_prompts=1,
        max_prompt_bytes=None,
        fail_on_duplicate_text=True,
    )

    assert warning_report["status"] == "pass"
    assert warning_report["warning_count"] == 1
    assert error_report["status"] == "fail"
    assert error_report["error_count"] == 1


def test_expected_token_gates_fail_missing_and_out_of_range(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt_id":"missing","prompt":"No expected token budget"}\n'
        '{"prompt_id":"too-small","prompt":"Short","expected_tokens":1}\n'
        '{"prompt_id":"too-large","prompt":"Long","expected_tokens":128}\n',
        encoding="utf-8",
    )

    report = prompt_audit.build_report(
        prompts,
        min_prompts=1,
        max_prompt_bytes=None,
        fail_on_duplicate_text=False,
        require_expected_tokens=True,
        min_expected_token_prompts=3,
        min_expected_tokens=8,
        max_expected_tokens=64,
    )

    messages = [issue["message"] for issue in report["issues"]]
    assert report["status"] == "fail"
    assert report["summary"]["expected_token_prompts"] == 2
    assert report["summary"]["expected_tokens_min"] == 1
    assert report["summary"]["expected_tokens_max"] == 128
    assert any("missing expected_tokens" in message for message in messages)
    assert any("below min 8" in message for message in messages)
    assert any("above max 64" in message for message in messages)
    assert any("below required minimum 3" in message for message in messages)


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    prompts = BENCH_PATH / "prompts" / "smoke.jsonl"
    output = tmp_path / "prompt_audit.json"
    markdown = tmp_path / "prompt_audit.md"
    csv_path = tmp_path / "prompt_audit.csv"
    junit = tmp_path / "prompt_audit_junit.xml"

    status = prompt_audit.main(
        [
            "--prompts",
            str(prompts),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--junit",
            str(junit),
            "--min-prompts",
            "2",
            "--max-prompt-bytes",
            "1024",
            "--require-expected-tokens",
            "--min-expected-token-prompts",
            "2",
            "--min-expected-tokens",
            "16",
            "--max-expected-tokens",
            "64",
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    md = markdown.read_text(encoding="utf-8")
    csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    junit_root = ET.parse(junit).getroot()
    assert payload["status"] == "pass"
    assert payload["limits"]["require_expected_tokens"] is True
    assert "Prompt Audit" in md
    assert "Expected-token prompts" in md
    assert "smoke-short" in md
    prompt_rows = [row for row in csv_rows if row["row_type"] == "prompt"]
    assert {row["prompt_id"] for row in prompt_rows} == {
        "smoke-short",
        "smoke-code",
    }
    assert {row["expected_tokens"] for row in prompt_rows} == {"32", "48"}
    assert junit_root.attrib["failures"] == "0"
    assert junit_root.find("testcase") is not None


def test_junit_marks_errors_as_failures(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt_id":"one","prompt":"too short"}\n', encoding="utf-8")
    output = tmp_path / "prompt_audit.json"
    junit = tmp_path / "prompt_audit_junit.xml"

    status = prompt_audit.main(
        [
            "--prompts",
            str(prompts),
            "--output",
            str(output),
            "--junit",
            str(junit),
            "--min-prompts",
            "2",
        ]
    )

    junit_root = ET.parse(junit).getroot()
    assert status == 1
    assert junit_root.attrib["failures"] == "1"
    failure = junit_root.find("testcase/failure")
    assert failure is not None
    assert "prompt count 1 is below required minimum 2" in (failure.text or "")


if __name__ == "__main__":
    test_smoke_prompt_audit_passes()
    print("prompt_audit_tests=ok")
