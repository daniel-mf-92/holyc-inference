#!/usr/bin/env python3
"""Tests for QEMU prompt identity audits."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_id_audit


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def row(prompt: str = "smoke-short", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "prompt_sha256": prompt_sha(prompt),
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "launch_index": 1,
        "iteration": 1,
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_prompt_id_audit.build_parser().parse_args(extra)


def test_audit_accepts_stable_prompt_identities(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row("smoke-short"), row("smoke-code", launch_index=2)])
    args = parse_args([str(artifact), "--min-rows", "2"])

    rows, findings = qemu_prompt_id_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 2


def test_audit_flags_prompt_hash_drift_and_collision(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row("same-name"),
            row("same-name", prompt_sha256=prompt_sha("other"), launch_index=2),
            row("alias", prompt_sha256=prompt_sha("same-name"), launch_index=3),
        ],
    )
    args = parse_args([str(artifact)])

    _, findings = qemu_prompt_id_audit.audit([artifact], args)
    kinds = {finding.kind for finding in findings}

    assert {"prompt_hash_drift", "prompt_hash_collision"} <= kinds


def test_audit_flags_missing_and_malformed_prompt_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row("", prompt_sha256="bad")])
    args = parse_args([str(artifact)])

    _, findings = qemu_prompt_id_audit.audit([artifact], args)
    kinds = {finding.kind for finding in findings}

    assert {"missing_prompt", "invalid_prompt_sha256"} <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row()])
    output_dir = tmp_path / "out"

    status = qemu_prompt_id_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "prompt_id"])

    assert status == 0
    payload = json.loads((output_dir / "prompt_id.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["unique_prompts"] == 1
    assert "No prompt identity findings." in (output_dir / "prompt_id.md").read_text(encoding="utf-8")
    assert "prompt_sha256" in (output_dir / "prompt_id.csv").read_text(encoding="utf-8")
    assert "kind" in (output_dir / "prompt_id_findings.csv").read_text(encoding="utf-8")
    root = ET.parse(output_dir / "prompt_id_junit.xml").getroot()
    assert root.attrib["name"] == "holyc_qemu_prompt_id_audit"
    assert root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_stable_prompt_identities(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_prompt_hash_drift_and_collision(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_and_malformed_prompt_identity(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
