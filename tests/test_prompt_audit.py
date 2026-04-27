#!/usr/bin/env python3
"""Host-side checks for benchmark prompt audit tooling."""

from __future__ import annotations

import importlib.util
import json
import sys
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


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    prompts = BENCH_PATH / "prompts" / "smoke.jsonl"
    output = tmp_path / "prompt_audit.json"
    markdown = tmp_path / "prompt_audit.md"

    status = prompt_audit.main(
        [
            "--prompts",
            str(prompts),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--min-prompts",
            "2",
            "--max-prompt-bytes",
            "1024",
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    md = markdown.read_text(encoding="utf-8")
    assert payload["status"] == "pass"
    assert "Prompt Audit" in md
    assert "smoke-short" in md


if __name__ == "__main__":
    test_smoke_prompt_audit_passes()
    print("prompt_audit_tests=ok")
