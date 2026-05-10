#!/usr/bin/env python3
"""Tests for prompt length coverage audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import prompt_length_audit


def write_suite(path: Path, prompts: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"id": prompt_id, "prompt": prompt} for prompt_id, prompt in prompts]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_audit_accepts_required_length_buckets(tmp_path: Path) -> None:
    suite = tmp_path / "prompts.jsonl"
    write_suite(suite, [("short", "s" * 12), ("medium", "m" * 160), ("long", "l" * 700)])
    args = prompt_length_audit.build_parser().parse_args(
        [
            str(suite),
            "--min-bucket-prompts",
            "short=1",
            "--min-bucket-prompts",
            "medium=1",
            "--min-bucket-prompts",
            "long=1",
        ]
    )
    args.buckets = args.buckets or list(prompt_length_audit.DEFAULT_BUCKETS)

    audit, rows, findings = prompt_length_audit.audit_source(suite, args.buckets, args)

    assert audit.status == "pass"
    assert audit.prompts == 3
    assert audit.bucket_counts["short"] == 1
    assert audit.bucket_counts["medium"] == 1
    assert audit.bucket_counts["long"] == 1
    assert [row.bucket for row in rows] == ["short", "medium", "long"]
    assert findings == []


def test_audit_flags_missing_bucket_and_duplicates(tmp_path: Path) -> None:
    suite = tmp_path / "prompts.jsonl"
    write_suite(suite, [("one", "duplicate"), ("two", "duplicate")])
    args = prompt_length_audit.build_parser().parse_args(
        [
            str(suite),
            "--min-bucket-prompts",
            "long=1",
            "--fail-on-duplicate-prompts",
        ]
    )
    args.buckets = args.buckets or list(prompt_length_audit.DEFAULT_BUCKETS)

    audit, _rows, findings = prompt_length_audit.audit_source(suite, args.buckets, args)

    assert audit.status == "fail"
    assert audit.duplicate_prompt_sha256 == 1
    assert {finding.kind for finding in findings} == {"duplicate_prompts", "min_bucket_prompts"}


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    suite = tmp_path / "prompts.jsonl"
    out = tmp_path / "out"
    write_suite(suite, [("short", "s" * 12), ("medium", "m" * 160), ("long", "l" * 700)])

    status = prompt_length_audit.main(
        [
            str(suite),
            "--output-dir",
            str(out),
            "--output-stem",
            "length",
            "--min-total-prompts",
            "3",
            "--min-bucket-prompts",
            "short=1",
            "--min-bucket-prompts",
            "medium=1",
            "--min-bucket-prompts",
            "long=1",
        ]
    )

    assert status == 0
    payload = json.loads((out / "length.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["bucket_counts"]["long"] == 1
    assert "Prompt Length Audit" in (out / "length.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((out / "length_prompts.csv").open(encoding="utf-8")))
    assert [row["bucket"] for row in rows] == ["short", "medium", "long"]
    junit_root = ET.parse(out / "length_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_prompt_length_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_sources_match(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    out = tmp_path / "out"
    empty.mkdir(parents=True)

    status = prompt_length_audit.main([str(empty), "--output-dir", str(out), "--output-stem", "length"])

    assert status == 1
    payload = json.loads((out / "length.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_sources"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_required_length_buckets(Path(tmp) / "accept")
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_bucket_and_duplicates(Path(tmp) / "fail")
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp) / "cli")
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_sources_match(Path(tmp) / "empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
