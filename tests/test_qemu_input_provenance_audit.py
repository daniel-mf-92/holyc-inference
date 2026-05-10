#!/usr/bin/env python3
"""Tests for QEMU input provenance audits."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_input_provenance_audit
import qemu_prompt_bench


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_artifact(path: Path, prompts: Path, image: Path, qemu_args: Path, *, drift: bool = False) -> None:
    cases = qemu_prompt_bench.load_prompt_cases(prompts)
    prompt_suite = qemu_prompt_bench.prompt_suite_metadata(prompts, cases)
    qemu_args_metadata = qemu_prompt_bench.input_file_metadata(qemu_args, include_sha256=True)
    if drift:
        prompt_suite = dict(prompt_suite, prompt_count=999)
        qemu_args_metadata = dict(qemu_args_metadata, sha256="0" * 64)
    payload = {
        "status": "pass",
        "prompt_suite": prompt_suite,
        "image": qemu_prompt_bench.input_file_metadata(image, include_sha256=True),
        "qemu_args_files": [qemu_args_metadata],
        "benchmarks": [],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_fixture(tmp_path: Path, *, drift: bool = False) -> Path:
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "TempleOS.img"
    qemu_args = tmp_path / "qemu.args"
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_jsonl(
        prompts,
        [
            {"id": "alpha", "prompt": "Alpha prompt", "expected_tokens": 3},
            {"id": "beta", "prompt": "Beta prompt", "expected_tokens": 4},
        ],
    )
    image.write_bytes(b"synthetic image")
    qemu_args.write_text("-display none -m 512M\n", encoding="utf-8")
    write_artifact(artifact, prompts, image, qemu_args, drift=drift)
    return artifact


def test_audit_accepts_matching_live_input_metadata(tmp_path: Path) -> None:
    artifact = make_fixture(tmp_path)
    args = qemu_input_provenance_audit.parse_args(
        [str(artifact), "--require-live-inputs", "--require-file-sha256", "--require-image-metadata"]
    )

    records, findings = qemu_input_provenance_audit.audit([artifact], args)

    assert findings == []
    assert len(records) == 1
    assert records[0].prompt_live_checked is True
    assert records[0].image_live_checked is True
    assert records[0].qemu_args_live_checked == 1


def test_audit_flags_prompt_and_qemu_args_metadata_drift(tmp_path: Path) -> None:
    artifact = make_fixture(tmp_path, drift=True)
    args = qemu_input_provenance_audit.parse_args(
        [str(artifact), "--require-live-inputs", "--require-file-sha256", "--require-image-metadata"]
    )

    records, findings = qemu_input_provenance_audit.audit([artifact], args)

    assert records[0].findings == 2
    kinds = {finding.kind for finding in findings}
    assert {"prompt_suite_drift", "sha256_drift"} <= kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = make_fixture(tmp_path)
    output_dir = tmp_path / "out"

    status = qemu_input_provenance_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "input_provenance",
            "--require-live-inputs",
            "--require-file-sha256",
            "--require-image-metadata",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "input_provenance.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["prompt_live_checked"] == 1
    assert "QEMU Input Provenance Audit" in (output_dir / "input_provenance.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "input_provenance.csv").open(encoding="utf-8")))
    assert rows[0]["qemu_args_live_checked"] == "1"
    finding_rows = list(csv.DictReader((output_dir / "input_provenance_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit = ET.parse(output_dir / "input_provenance_junit.xml").getroot()
    assert junit.attrib["name"] == "qemu_input_provenance_audit"
    assert junit.attrib["failures"] == "0"

