from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_serial_payload_audit


def args(**overrides) -> Namespace:
    defaults = {
        "pattern": [],
        "min_rows": 1,
        "require_ok_payload": True,
        "allow_multiple_payloads": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def artifact(row: dict) -> dict:
    base = {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "benchmarks": [row],
    }
    return base


def passing_row() -> dict:
    payload = {
        "tokens": 8,
        "elapsed_us": 1000,
        "time_to_first_token_us": 100,
        "memory_bytes": 4096,
        "prompt_bytes": 12,
        "prompt_sha256": "abc123",
    }
    return {
        "prompt": "smoke",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 8,
        "elapsed_us": 1000,
        "ttft_us": 100,
        "memory_bytes": 4096,
        "prompt_bytes": 12,
        "prompt_sha256": "abc123",
        "stdout_tail": "boot\nBENCH_RESULT: " + json.dumps(payload) + "\n",
        "stderr_tail": "",
    }


def test_audit_accepts_matching_serial_payload(tmp_path: Path) -> None:
    path = tmp_path / "qemu_prompt_bench_latest.json"
    path.write_text(json.dumps(artifact(passing_row())) + "\n", encoding="utf-8")

    rows, findings = qemu_serial_payload_audit.audit([path], args())

    assert findings == []
    assert len(rows) == 1
    assert rows[0].payload_count == 1
    assert rows[0].checks == 6


def test_audit_rejects_missing_invalid_duplicate_and_mismatched_payloads(tmp_path: Path) -> None:
    missing = passing_row()
    missing["stdout_tail"] = "booted without result\n"
    invalid = passing_row()
    invalid["stdout_tail"] = "BENCH_RESULT: {not-json}\n"
    duplicate = passing_row()
    duplicate["stdout_tail"] = duplicate["stdout_tail"] + duplicate["stdout_tail"]
    mismatch = passing_row()
    mismatch_payload = {
        "tokens": 7,
        "elapsed_us": 1000,
        "time_to_first_token_us": 100,
        "memory_bytes": 4096,
        "prompt_bytes": 12,
        "prompt_sha256": "abc123",
    }
    mismatch["stdout_tail"] = "BENCH_RESULT: " + json.dumps(mismatch_payload) + "\n"

    path = tmp_path / "qemu_prompt_bench_latest.json"
    path.write_text(
        json.dumps({"benchmarks": [missing, invalid, duplicate, mismatch]}) + "\n",
        encoding="utf-8",
    )

    rows, findings = qemu_serial_payload_audit.audit([path], args())
    kinds = {finding.kind for finding in findings}

    assert len(rows) == 4
    assert "missing_payload" in kinds
    assert "invalid_payload" in kinds
    assert "multiple_payloads" in kinds
    assert "payload_mismatch" in kinds


def test_cli_writes_reports_and_junit(tmp_path: Path) -> None:
    input_path = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    input_path.write_text(json.dumps(artifact(passing_row())) + "\n", encoding="utf-8")

    status = qemu_serial_payload_audit.main(
        [str(input_path), "--output-dir", str(output_dir), "--output-stem", "serial_payload"]
    )

    assert status == 0
    report = json.loads((output_dir / "serial_payload.json").read_text(encoding="utf-8"))
    junit = ET.parse(output_dir / "serial_payload_junit.xml").getroot()
    assert report["status"] == "pass"
    assert report["summary"]["rows_with_payload"] == 1
    assert junit.attrib["failures"] == "0"
    assert "QEMU Serial Payload Audit" in (output_dir / "serial_payload.md").read_text(encoding="utf-8")
