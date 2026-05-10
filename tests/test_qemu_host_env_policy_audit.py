#!/usr/bin/env python3
"""Tests for QEMU host environment policy audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_host_env_policy_audit


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_host_env_policy_audit.build_parser().parse_args(extra)


def test_audit_accepts_clean_captured_environment(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [{"environment": {"PATH": "/usr/bin:/bin", "QEMU_AUDIO_DRV": "none"}}])
    args = parse_args([str(artifact), "--require-environment"])

    records, findings = qemu_host_env_policy_audit.audit([artifact], args)

    assert len(records) == 1
    assert records[0].network_env_count == 0
    assert findings == []


def test_audit_flags_proxy_environment_and_url_values(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            {"environment": {"HTTP_PROXY": "http://127.0.0.1:8080", "PATH": "/usr/bin:/bin"}},
            {"host_environment": {"CUSTOM_ENDPOINT": "https://example.invalid/offline"}},
        ],
    )
    args = parse_args([str(artifact), "--fail-on-url-values", "--min-records", "2"])

    records, findings = qemu_host_env_policy_audit.audit([artifact], args)

    assert len(records) == 2
    kinds = {finding.kind for finding in findings}
    assert {"network_env_var", "url_env_value"} <= kinds
    assert any(finding.variable == "HTTP_PROXY" for finding in findings)


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_artifact(artifact, [{"environment": {"PATH": "/usr/bin:/bin"}}])

    status = qemu_host_env_policy_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "host_env", "--require-environment"]
    )

    assert status == 0
    payload = json.loads((output_dir / "host_env.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["records"] == 1
    assert "No host environment policy findings." in (output_dir / "host_env.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "host_env.csv").open(encoding="utf-8")))
    assert rows[0]["scope"] == "environment"
    finding_rows = list(csv.DictReader((output_dir / "host_env_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "host_env_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_host_env_policy_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_missing_required_environment(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_artifact(artifact, [{"profile": "smoke"}])

    status = qemu_host_env_policy_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "host_env", "--require-environment"]
    )

    assert status == 1
    payload = json.loads((output_dir / "host_env.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "missing_environment"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_clean_captured_environment(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_proxy_environment_and_url_values(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_missing_required_environment(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
