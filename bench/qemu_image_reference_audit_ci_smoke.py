#!/usr/bin/env python3
"""Smoke gate for qemu_image_reference_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench" / "qemu_image_reference_audit.py"


def artifact(image_path: str, command_path: str) -> dict[str, object]:
    command = [
        "qemu-system-x86_64",
        "-nic",
        "none",
        "-serial",
        "stdio",
        "-drive",
        f"file={command_path},format=raw,if=ide",
    ]
    return {
        "artifact_schema_version": "qemu-prompt-bench/v1",
        "status": "pass",
        "image": {"path": image_path, "exists": False, "size_bytes": None, "sha256": None},
        "command": command,
        "benchmarks": [{"command": command, "prompt": "smoke"}],
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="qemu-image-ref-smoke-") as tmp:
        root = Path(tmp)
        inputs = root / "inputs"
        output_dir = root / "out"
        inputs.mkdir()

        (inputs / "pass.json").write_text(json.dumps(artifact("/tmp/TempleOS.img", "/tmp/TempleOS.img")) + "\n", encoding="utf-8")
        pass_cmd = [
            sys.executable,
            str(SCRIPT),
            str(inputs / "pass.json"),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_image_reference_audit_smoke_pass",
            "--require-drive-reference",
            "--require-single-drive-path",
        ]
        subprocess.run(pass_cmd, cwd=ROOT, check=True)

        (inputs / "fail.json").write_text(json.dumps(artifact("/tmp/TempleOS.img", "/tmp/Other.img")) + "\n", encoding="utf-8")
        fail_cmd = [
            sys.executable,
            str(SCRIPT),
            str(inputs / "fail.json"),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_image_reference_audit_smoke_fail",
            "--require-drive-reference",
        ]
        failed = subprocess.run(fail_cmd, cwd=ROOT, check=False)
        if failed.returncode == 0:
            raise SystemExit("expected mismatched image/drive artifact to fail")

        report = json.loads((output_dir / "qemu_image_reference_audit_smoke_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        if "drive_image_mismatch" not in kinds:
            raise SystemExit(f"expected drive_image_mismatch finding, got {sorted(kinds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
