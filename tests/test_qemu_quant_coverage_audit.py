#!/usr/bin/env python3
"""Tests for QEMU quantization coverage audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_quant_coverage_audit


def row(prompt: str, quantization: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": quantization,
        "phase": "measured",
        "exit_class": "ok",
        "returncode": 0,
        "timed_out": False,
        "command_airgap_ok": True,
        "command_has_explicit_nic_none": True,
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_quant_coverage_audit.build_parser().parse_args(extra)


def test_audit_accepts_required_quantizations(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row("a", "Q4_0"), row("b", "Q4_0"), row("a", "Q8_0"), row("b", "Q8_0")])
    args = parse_args([str(artifact), "--min-rows-per-quant", "2", "--min-prompts-per-quant", "2"])

    rows, findings = qemu_quant_coverage_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 2
    assert {item.quantization for item in rows} == {"Q4_0", "Q8_0"}
    assert {item.airgap_ok_rows for item in rows} == {2}


def test_audit_flags_missing_quantization_and_low_success_count(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row("a", "Q4_0"),
            row("b", "Q4_0", exit_class="timeout", returncode=124, timed_out=True),
        ],
    )
    args = parse_args([str(artifact), "--min-ok-rows-per-quant", "2"])

    rows, findings = qemu_quant_coverage_audit.audit([artifact], args)

    assert rows[0].ok_rows == 1
    assert {"missing_required_quantization", "min_ok_rows_per_quant"} <= {finding.kind for finding in findings}


def test_audit_can_require_airgapped_command_telemetry(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row("a", "Q4_0", command_airgap_ok=False),
            row("b", "Q4_0", command_has_explicit_nic_none=False),
            row("a", "Q8_0"),
        ],
    )
    args = parse_args([str(artifact), "--require-airgap-command", "--require-quant", "Q4_0"])

    rows, findings = qemu_quant_coverage_audit.audit([artifact], args)

    assert len(rows) == 2
    assert {"missing_airgap_command", "missing_explicit_nic_none"} <= {finding.kind for finding in findings}


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row("a", "Q4_0"), row("a", "Q8_0")])
    output_dir = tmp_path / "out"

    status = qemu_quant_coverage_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "quant_coverage"]
    )

    assert status == 0
    payload = json.loads((output_dir / "quant_coverage.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["groups"] == 2
    assert payload["summary"]["airgap_ok_rows"] == 2
    assert "No quantization coverage findings." in (output_dir / "quant_coverage.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "quant_coverage.csv").open(encoding="utf-8")))
    assert {row["quantization"] for row in rows} == {"Q4_0", "Q8_0"}
    finding_rows = list(csv.DictReader((output_dir / "quant_coverage_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "quant_coverage_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_quant_coverage_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_required_quantizations(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_quantization_and_low_success_count(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_can_require_airgapped_command_telemetry(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
