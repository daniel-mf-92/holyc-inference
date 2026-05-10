#!/usr/bin/env python3
"""Tests for QEMU prompt source audits."""

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

import qemu_prompt_source_audit


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(extra: list[str]) -> object:
    return qemu_prompt_source_audit.build_parser().parse_args(extra)


def write_prompts(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt_id": "short", "prompt": "Summarize integer-only inference.", "expected_tokens": 8}),
                json.dumps({"prompt_id": "code", "prompt": "Write a tiny loop.", "expected_tokens": 12}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_artifact(path: Path, prompt_source: Path, row: dict[str, object]) -> None:
    path.write_text(json.dumps({"prompt_suite": {"source": str(prompt_source)}, "benchmarks": [row]}), encoding="utf-8")


def good_row(**overrides: object) -> dict[str, object]:
    prompt = "Summarize integer-only inference."
    row: dict[str, object] = {
        "prompt": "short",
        "phase": "measured",
        "prompt_sha256": sha(prompt),
        "prompt_bytes": len(prompt.encode("utf-8")),
        "expected_tokens": 8,
    }
    row.update(overrides)
    return row


def test_audit_accepts_rows_matching_prompt_source(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_prompts(prompts)
    write_artifact(artifact, prompts, good_row())
    args = parse_args([str(artifact), "--require-expected-tokens"])

    rows, findings = qemu_prompt_source_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].checks == 3
    assert rows[0].expected_prompt_bytes == rows[0].prompt_bytes


def test_audit_rejects_prompt_hash_and_byte_drift(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_prompts(prompts)
    write_artifact(artifact, prompts, good_row(prompt_sha256="bad", prompt_bytes=1))
    args = parse_args([str(artifact)])

    rows, findings = qemu_prompt_source_audit.audit([artifact], args)

    assert len(rows) == 1
    assert {"prompt_sha256_mismatch", "prompt_bytes_mismatch"} <= {finding.kind for finding in findings}


def test_audit_rejects_unknown_prompt_id(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_prompts(prompts)
    write_artifact(artifact, prompts, good_row(prompt="missing"))
    args = parse_args([str(artifact)])

    rows, findings = qemu_prompt_source_audit.audit([artifact], args)

    assert len(rows) == 1
    assert {finding.kind for finding in findings} == {"unknown_prompt_id"}


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_prompts(prompts)
    write_artifact(artifact, prompts, good_row())

    status = qemu_prompt_source_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "prompt_source",
            "--require-expected-tokens",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "prompt_source.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert "No prompt source findings." in (output_dir / "prompt_source.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "prompt_source.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "short"
    finding_rows = list(csv.DictReader((output_dir / "prompt_source_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "prompt_source_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_prompt_source_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_rows_matching_prompt_source(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_rejects_prompt_hash_and_byte_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_rejects_unknown_prompt_id(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
