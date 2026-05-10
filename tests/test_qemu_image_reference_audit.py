from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_image_reference_audit


def artifact(image_path: str, command_path: str, *, command_style: str = "drive") -> dict[str, object]:
    if command_style == "hda":
        command = ["qemu-system-x86_64", "-nic", "none", "-hda", command_path]
    else:
        command = ["qemu-system-x86_64", "-nic", "none", "-drive", f"file={command_path},format=raw,if=ide"]
    return {
        "artifact_schema_version": "qemu-prompt-bench/v1",
        "image": {"path": image_path, "exists": True, "size_bytes": 1024, "sha256": "a" * 64},
        "command": command,
        "benchmarks": [{"command": command, "prompt": "smoke"}],
    }


def args(**overrides: object) -> argparse.Namespace:
    values = {
        "pattern": list(qemu_image_reference_audit.DEFAULT_PATTERNS),
        "min_artifacts": 1,
        "require_drive_reference": True,
        "require_existing_image": False,
        "require_image_hash": False,
        "require_image_size": False,
        "require_single_drive_path": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_audit_accepts_matching_drive_and_hda_references(tmp_path: Path) -> None:
    (tmp_path / "drive.json").write_text(json.dumps(artifact("/tmp/TempleOS.img", "/tmp/TempleOS.img")) + "\n", encoding="utf-8")
    (tmp_path / "hda.json").write_text(
        json.dumps(artifact("/tmp/TempleOS2.img", "/tmp/TempleOS2.img", command_style="hda")) + "\n",
        encoding="utf-8",
    )

    records, findings = qemu_image_reference_audit.audit([tmp_path], args())

    assert findings == []
    assert len(records) == 2
    assert {record.status for record in records} == {"pass"}
    assert {record.distinct_drive_paths for record in records} == {1}


def test_audit_rejects_image_metadata_drift_and_missing_drive(tmp_path: Path) -> None:
    (tmp_path / "mismatch.json").write_text(json.dumps(artifact("/tmp/TempleOS.img", "/tmp/Other.img")) + "\n", encoding="utf-8")
    missing = artifact("/tmp/TempleOS.img", "/tmp/TempleOS.img")
    missing["command"] = ["qemu-system-x86_64", "-nic", "none"]
    missing["benchmarks"] = [{"command": ["qemu-system-x86_64", "-nic", "none"], "prompt": "smoke"}]
    (tmp_path / "missing.json").write_text(json.dumps(missing) + "\n", encoding="utf-8")

    records, findings = qemu_image_reference_audit.audit([tmp_path], args())
    kinds = {finding.kind for finding in findings}

    assert len(records) == 2
    assert "drive_image_mismatch" in kinds
    assert "missing_drive_reference" in kinds


def test_cli_writes_reports_and_junit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "pass.json"
    output_dir = tmp_path / "out"
    artifact_path.write_text(json.dumps(artifact("/tmp/TempleOS.img", "/tmp/TempleOS.img")) + "\n", encoding="utf-8")

    status = qemu_image_reference_audit.main(
        [
            str(artifact_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "image_ref",
            "--require-drive-reference",
            "--require-image-hash",
            "--require-image-size",
            "--require-single-drive-path",
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "image_ref.json").read_text(encoding="utf-8"))
    junit = ET.parse(output_dir / "image_ref_junit.xml").getroot()
    assert report["status"] == "pass"
    assert report["summary"]["artifacts"] == 1
    assert report["summary"]["drive_references"] == 2
    assert junit.attrib["failures"] == "0"
    assert "QEMU Image Reference Audit" in (output_dir / "image_ref.md").read_text(encoding="utf-8")
