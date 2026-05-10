#!/usr/bin/env python3
"""Tests for QEMU replay manifest tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_bench
import qemu_replay_manifest


COMMAND = ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio", "-display", "none"]


def write_artifact(path: Path, *, command: list[str] | None = None, hash_override: str | None = None) -> None:
    argv = command or COMMAND
    command_airgap = qemu_prompt_bench.command_airgap_metadata(argv)
    command_sha256 = qemu_prompt_bench.command_hash(argv) if hash_override is None else hash_override
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-30T00:00:00Z",
                "status": "pass",
                "commit": "abc123",
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "command": argv,
                "command_sha256": command_sha256,
                "command_airgap": command_airgap,
                "prompt_suite": {
                    "source": "bench/prompts/smoke.jsonl",
                    "suite_sha256": "a" * 64,
                    "prompt_count": 2,
                },
                "launch_plan_sha256": "b" * 64,
                "expected_launch_sequence_sha256": "c" * 64,
                "launch_plan": [
                    {"phase": "warmup", "prompt_id": "alpha"},
                    {"phase": "measured", "prompt_id": "alpha"},
                ],
                "benchmarks": [
                    {
                        "phase": "measured",
                        "command": argv,
                        "command_sha256": command_sha256,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_accepts_airgapped_replay_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact)

    entry, findings = qemu_replay_manifest.audit_artifact(artifact)

    assert findings == []
    assert entry is not None
    assert entry.airgap_ok is True
    assert entry.explicit_nic_none is True
    assert entry.legacy_net_none is False
    assert entry.measured_rows == 1
    assert entry.launch_plan_entries == 2
    assert entry.key.endswith(f"{'a' * 64}/{qemu_prompt_bench.command_hash(COMMAND)}")


def test_audit_flags_hash_and_airgap_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, command=["qemu-system-x86_64", "-netdev", "user,id=n0"], hash_override="bad")

    entry, findings = qemu_replay_manifest.audit_artifact(artifact)
    kinds = {finding.kind for finding in findings}

    assert entry is not None
    assert entry.status == "fail"
    assert "command_hash" in kinds
    assert "command_airgap" in kinds
    assert "row_command_hash" in kinds


def test_cli_writes_manifest_reports(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_artifact(artifact)

    status = qemu_replay_manifest.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "replay"]
    )

    assert status == 0
    payload = json.loads((output_dir / "replay.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["artifacts"] == 1
    assert payload["entries"][0]["argv"] == COMMAND
    rows = list(csv.DictReader((output_dir / "replay.csv").open(encoding="utf-8")))
    assert rows[0]["airgap_ok"] == "True"
    argv_rows = [json.loads(line) for line in (output_dir / "replay_argv.jsonl").read_text(encoding="utf-8").splitlines()]
    assert argv_rows[0]["argv"] == COMMAND
    assert "QEMU Replay Manifest" in (output_dir / "replay.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "replay_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_replay_manifest"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    status = qemu_replay_manifest.main(
        [str(input_dir), "--output-dir", str(output_dir), "--output-stem", "replay", "--min-artifacts", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "replay.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_airgapped_replay_artifact(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_hash_and_airgap_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_manifest_reports(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_artifacts_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
