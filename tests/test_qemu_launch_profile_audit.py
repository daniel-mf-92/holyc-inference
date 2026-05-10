from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_launch_profile_audit


def artifact(*, root_memory: str = "256M", row_memory: str = "256M", machine: str = "q35") -> dict[str, object]:
    root_command = ["qemu-system-x86_64", "-nic", "none", "-M", machine, "-cpu", "max", "-m", root_memory]
    row_command = ["qemu-system-x86_64", "-nic", "none", "-M", machine, "-cpu", "max", "-m", row_memory]
    return {
        "artifact_schema_version": "qemu-prompt-bench/v1",
        "status": "pass",
        "command": root_command,
        "warmups": [{"command": row_command, "prompt": "warm"}],
        "benchmarks": [{"command": row_command, "prompt": "smoke"}],
    }


def args(**overrides: object) -> argparse.Namespace:
    values = {
        "pattern": list(qemu_launch_profile_audit.DEFAULT_PATTERNS),
        "min_artifacts": 1,
        "min_commands": 1,
        "require_memory": True,
        "require_machine": False,
        "require_cpu": False,
        "require_accelerator": False,
        "fail_on_cross_artifact_drift": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_audit_accepts_stable_launch_profile(tmp_path: Path) -> None:
    (tmp_path / "qemu_prompt_bench_pass.json").write_text(json.dumps(artifact()) + "\n", encoding="utf-8")

    records, findings = qemu_launch_profile_audit.audit([tmp_path], args(require_machine=True, require_cpu=True))

    assert findings == []
    assert len(records) == 3
    assert {record.memory for record in records} == {"256M"}
    assert {record.machine for record in records} == {"q35"}


def test_audit_rejects_memory_drift_and_missing_required_fields(tmp_path: Path) -> None:
    drift = artifact(row_memory="512M")
    missing = artifact()
    missing["command"] = ["qemu-system-x86_64", "-nic", "none"]
    missing["benchmarks"] = [{"command": ["qemu-system-x86_64", "-nic", "none"], "prompt": "smoke"}]
    (tmp_path / "qemu_prompt_bench_drift.json").write_text(json.dumps(drift) + "\n", encoding="utf-8")
    (tmp_path / "qemu_prompt_bench_missing.json").write_text(json.dumps(missing) + "\n", encoding="utf-8")

    records, findings = qemu_launch_profile_audit.audit([tmp_path], args(require_machine=True, require_cpu=True))
    kinds = {finding.kind for finding in findings}

    assert len(records) == 6
    assert "profile_drift" in kinds
    assert "missing_memory" in kinds
    assert "missing_machine" in kinds
    assert "missing_cpu" in kinds


def test_audit_rejects_cross_artifact_profile_drift(tmp_path: Path) -> None:
    (tmp_path / "qemu_prompt_bench_q4.json").write_text(
        json.dumps(artifact(root_memory="256M", row_memory="256M")) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "qemu_prompt_bench_q8.json").write_text(
        json.dumps(artifact(root_memory="512M", row_memory="512M")) + "\n",
        encoding="utf-8",
    )

    records, findings = qemu_launch_profile_audit.audit([tmp_path], args(fail_on_cross_artifact_drift=True))

    assert len(records) == 6
    assert {finding.kind for finding in findings} == {"cross_artifact_profile_drift"}


def test_cli_writes_reports_and_junit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_pass.json"
    output_dir = tmp_path / "out"
    artifact_path.write_text(json.dumps(artifact()) + "\n", encoding="utf-8")

    status = qemu_launch_profile_audit.main(
        [
            str(artifact_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "launch_profile",
            "--require-memory",
            "--require-machine",
            "--require-cpu",
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "launch_profile.json").read_text(encoding="utf-8"))
    junit = ET.parse(output_dir / "launch_profile_junit.xml").getroot()
    assert report["status"] == "pass"
    assert report["summary"]["commands"] == 3
    assert junit.attrib["failures"] == "0"
    assert "QEMU Launch Profile Audit" in (output_dir / "launch_profile.md").read_text(encoding="utf-8")
