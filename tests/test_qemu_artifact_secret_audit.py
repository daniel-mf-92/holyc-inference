from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_artifact_secret_audit


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmarks": rows}, indent=2) + "\n", encoding="utf-8")


def base_row() -> dict[str, object]:
    return {
        "benchmark": "qemu-prompt",
        "command": ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"],
        "stdout_tail": "BENCH_RESULT: {\"tokens\": 8, \"elapsed_us\": 100000}\n",
        "stderr_tail": "",
        "failure_reason": "",
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C"},
    }


def test_secret_audit_passes_safe_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_safe.json"
    args = qemu_artifact_secret_audit.build_parser().parse_args([str(artifact)])
    write_artifact(artifact, [base_row()])

    records, findings = qemu_artifact_secret_audit.audit([artifact], args)

    assert len(records) == 1
    assert findings == []
    assert records[0].sensitive_fields_checked == 0


def test_secret_audit_flags_tokens_urls_and_sensitive_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_unsafe.json"
    unsafe_row = base_row()
    unsafe_row.update(
        {
            "stdout_tail": "auth failed for sk-proj-abcdefghijklmnopqrstuvwxyz123456",
            "stderr_tail": "clone https://user:pass@example.invalid/repo.git failed",
            "authorization_header": "Bearer manually-configured-token",
            "set_cookie": "session=abc123",
            "x_api_key": "not-a-real-key-but-sensitive",
        }
    )
    write_artifact(artifact, [unsafe_row])
    args = qemu_artifact_secret_audit.build_parser().parse_args([str(artifact)])

    records, findings = qemu_artifact_secret_audit.audit([artifact], args)
    kinds = {finding.kind for finding in findings}
    paths = {finding.json_path for finding in findings}

    assert len(records) == 1
    assert records[0].sensitive_fields_checked == 3
    assert "openai_api_key" in kinds
    assert "url_embedded_credentials" in kinds
    assert "sensitive_field_populated" in kinds
    assert "$.authorization_header" in paths
    assert "$.set_cookie" in paths
    assert "$.x_api_key" in paths
    assert all("abcdefghijklmnopqrstuvwxyz123456" not in finding.redacted_sample for finding in findings)


def test_secret_audit_cli_writes_sidecars(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_safe.json"
    output_dir = tmp_path / "out"
    write_artifact(artifact, [base_row()])

    status = qemu_artifact_secret_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "secret_audit"]
    )

    assert status == 0
    payload = json.loads((output_dir / "secret_audit.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "secret_audit.csv").open(encoding="utf-8")))
    findings = list(csv.DictReader((output_dir / "secret_audit_findings.csv").open(encoding="utf-8")))
    junit = ET.parse(output_dir / "secret_audit_junit.xml").getroot()
    markdown = (output_dir / "secret_audit.md").read_text(encoding="utf-8")

    assert payload["status"] == "pass"
    assert len(rows) == 1
    assert findings == []
    assert junit.attrib["name"] == "holyc_qemu_artifact_secret_audit"
    assert "No secret-like artifact text findings." in markdown


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory(prefix="test-qemu-artifact-secret-audit-") as tmp:
        root = Path(tmp)
        test_secret_audit_passes_safe_artifact(root / "safe")
        test_secret_audit_flags_tokens_urls_and_sensitive_fields(root / "unsafe")
        test_secret_audit_cli_writes_sidecars(root / "cli")
    print("test_qemu_artifact_secret_audit=ok")
