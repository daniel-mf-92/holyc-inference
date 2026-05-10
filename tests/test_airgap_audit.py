#!/usr/bin/env python3
"""Tests for saved benchmark QEMU air-gap audits."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import airgap_audit


def test_load_records_flattens_benchmark_commands(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "profile": "ci",
                "command": ["qemu-system-x86_64", "-nic", "none"],
                "benchmarks": [{"prompt": "p0", "phase": "measured"}],
            }
        ),
        encoding="utf-8",
    )

    records = airgap_audit.load_records([report])

    assert len(records) == 1
    assert records[0].prompt == "p0"
    assert records[0].command == ["qemu-system-x86_64", "-nic", "none"]


def test_evaluate_flags_missing_nic_and_recorded_drift(tmp_path: Path) -> None:
    report = tmp_path / "bench.jsonl"
    report.write_text(
        json.dumps(
            {
                "prompt": "bad",
                "phase": "measured",
                "command": ["qemu-system-x86_64", "-net", "none", "-device", "e1000"],
                "command_airgap_ok": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = airgap_audit.load_records([report])
    findings = airgap_audit.evaluate(records, min_commands=1)

    assert {finding.kind for finding in findings} == {"airgap_violation", "recorded_airgap_drift"}
    assert any("missing explicit `-nic none`" in finding.detail for finding in findings)
    assert any("network device" in finding.detail for finding in findings)


def test_evaluate_flags_tls_options_in_saved_commands(tmp_path: Path) -> None:
    report = tmp_path / "bench.jsonl"
    report.write_text(
        json.dumps(
            {
                "prompt": "bad",
                "phase": "measured",
                "command": [
                    "qemu-system-x86_64",
                    "-nic",
                    "none",
                    "-object",
                    "tls-creds-x509,id=tls0,endpoint=server,dir=/tmp/tls",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = airgap_audit.load_records([report])
    findings = airgap_audit.evaluate(records, min_commands=1)

    assert any("tls option" in finding.detail for finding in findings)


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.csv"
    output_dir = tmp_path / "results"
    report.write_text(
        "\n".join(
            [
                "prompt,phase,command,command_airgap_ok",
                "p0,measured,\"qemu-system-x86_64 -nic none -serial stdio\",true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    status = airgap_audit.main([str(report), "--output-dir", str(output_dir), "--min-commands", "1"])

    assert status == 0
    payload = json.loads((output_dir / "airgap_audit_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["commands_with_explicit_nic_none"] == 1
    assert "No air-gap findings." in (output_dir / "airgap_audit_latest.md").read_text(encoding="utf-8")
    assert "severity" in (output_dir / "airgap_audit_latest.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "airgap_audit_latest_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_airgap_audit"
    assert junit_root.attrib["failures"] == "0"
