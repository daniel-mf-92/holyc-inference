#!/usr/bin/env python3
"""Tests for dataset content hash audits."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_content_hash_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def choices_hash(choices: list[str]) -> str:
    return hashlib.sha256(json.dumps(choices, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def combined_hash(prompt_sha256: str, choices_sha256: str) -> str:
    payload = json.dumps(
        {"choices_sha256": choices_sha256, "prompt_sha256": prompt_sha256},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args(extra: list[str]) -> object:
    return dataset_content_hash_audit.parse_args(extra)


def test_content_hash_audit_passes_matching_hashes(tmp_path: Path) -> None:
    prompt = "Pick the coldest object."
    choices = ["ice", "steam"]
    p_hash = prompt_hash(prompt)
    c_hash = choices_hash(choices)
    input_hash = combined_hash(p_hash, c_hash)
    rows = tmp_path / "rows.jsonl"
    write_jsonl(
        rows,
        [
            {
                "id": "a",
                "prompt": prompt,
                "choices": choices,
                "answer_index": 0,
                "prompt_sha256": p_hash,
                "choices_sha256": c_hash,
                "input_sha256": input_hash,
            }
        ],
    )

    report = dataset_content_hash_audit.build_report(parse_args(["--input", str(rows), "--require-all-hashes"]))

    assert report["status"] == "pass"
    assert report["summary"]["record_count"] == 1
    assert report["summary"]["prompt_hash"]["match"] == 1
    assert report["records"][0]["input_hash_status"] == "match"


def test_content_hash_audit_flags_mismatch_missing_and_schema_errors(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    write_jsonl(
        rows,
        [
            {"id": "bad-hash", "prompt": "Q?", "choices": ["A", "B"], "answer_index": 0, "prompt_sha256": "f" * 64},
            {"id": "missing", "prompt": "Q2?", "choices": ["A", "B"], "answer_index": 1},
            {"id": "schema", "prompt": "", "choices": ["A", "B"], "answer_index": 0},
        ],
    )

    report = dataset_content_hash_audit.build_report(parse_args(["--input", str(rows), "--require-all-hashes"]))

    kinds = {finding["kind"] for finding in report["findings"]}
    assert {"prompt_hash_mismatch", "missing_prompt_hash", "missing_choices_hash", "missing_input_hash", "schema_error"} <= kinds
    assert report["status"] == "fail"


def test_content_hash_cli_writes_sidecars(tmp_path: Path) -> None:
    prompt = "Pick the largest object."
    choices = ["grain", "planet"]
    p_hash = prompt_hash(prompt)
    c_hash = choices_hash(choices)
    rows = tmp_path / "rows.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(
        rows,
        [
            {
                "id": "a",
                "prompt": prompt,
                "choices": choices,
                "answer_index": 1,
                "metadata": {
                    "prompt_hash": p_hash,
                    "choices_hash": c_hash,
                    "prompt_choices_sha256": combined_hash(p_hash, c_hash),
                },
            }
        ],
    )

    status = dataset_content_hash_audit.main(
        [
            "--input",
            str(rows),
            "--require-all-hashes",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "content_hash",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "content_hash.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert "Dataset Content Hash Audit" in (output_dir / "content_hash.md").read_text(encoding="utf-8")
    record_rows = list(csv.DictReader((output_dir / "content_hash.csv").open(encoding="utf-8")))
    assert record_rows[0]["record_id"] == "a"
    finding_rows = list(csv.DictReader((output_dir / "content_hash_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "content_hash_junit.xml").getroot()
    assert junit_root.attrib["name"] == "dataset_content_hash_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_content_hash_audit_passes_matching_hashes(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_content_hash_audit_flags_mismatch_missing_and_schema_errors(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_content_hash_cli_writes_sidecars(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
