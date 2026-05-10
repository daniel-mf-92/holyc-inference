from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_artifact_reference_audit
import qemu_prompt_bench


COMMAND = ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"]


def write_artifact(path: Path, *, command: list[str] | None = None, prompt_source: str = "bench/prompts/smoke.jsonl") -> None:
    row_command = command or COMMAND
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:00:00Z",
                "status": "pass",
                "command": row_command,
                "command_sha256": qemu_prompt_bench.command_hash(row_command),
                "command_airgap": qemu_prompt_bench.command_airgap_metadata(row_command),
                "prompt_suite": {"source": prompt_source, "prompt_count": 1},
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "phase": "measured",
                        "prompt": "smoke",
                        "model": "synthetic-smoke",
                        "command": row_command,
                        "stdout_tail": "prompt text may mention https://example.invalid without being a resource",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_accepts_local_artifact_references(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact)
    args = qemu_artifact_reference_audit.build_parser().parse_args([str(artifact)])

    record, findings = qemu_artifact_reference_audit.audit_artifact(artifact, args)

    assert record.status == "pass"
    assert record.command_arrays_checked == 2
    assert record.remote_reference_count == 0
    assert findings == []


def test_audit_rejects_remote_resources_and_network_commands(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, command=["qemu-system-x86_64", "-netdev", "user,id=n0"], prompt_source="https://example.invalid/prompts.jsonl")
    args = qemu_artifact_reference_audit.build_parser().parse_args([str(artifact)])

    record, findings = qemu_artifact_reference_audit.audit_artifact(artifact, args)

    assert record.status == "fail"
    kinds = {finding.kind for finding in findings}
    assert "remote_uri" in kinds
    assert "missing_nic_none" in kinds
    assert "command_airgap_violation" in kinds


def test_cli_writes_reports_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_artifact(artifact)

    status = qemu_artifact_reference_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "refs"]
    )

    assert status == 0
    report = json.loads((output_dir / "refs.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["summary"]["artifacts"] == 1
    assert "QEMU Artifact Reference Audit" in (output_dir / "refs.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "refs.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "pass"
    findings = list(csv.DictReader((output_dir / "refs_findings.csv").open(encoding="utf-8")))
    assert findings == []
    junit = ET.parse(output_dir / "refs_junit.xml").getroot()
    assert junit.attrib["name"] == "holyc_qemu_artifact_reference_audit"
    assert junit.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    output_dir = tmp_path / "out"
    empty.mkdir()

    status = qemu_artifact_reference_audit.main(
        [str(empty), "--output-dir", str(output_dir), "--output-stem", "refs", "--min-artifacts", "1"]
    )

    assert status == 1
    report = json.loads((output_dir / "refs.json").read_text(encoding="utf-8"))
    assert report["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_local_artifact_references(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_rejects_remote_resources_and_network_commands(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_reports_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_artifacts_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
