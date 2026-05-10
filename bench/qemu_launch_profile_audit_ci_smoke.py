#!/usr/bin/env python3
"""Smoke gate for qemu_launch_profile_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench" / "qemu_launch_profile_audit.py"


def artifact(memory: str = "256M") -> dict[str, object]:
    command = ["qemu-system-x86_64", "-nic", "none", "-M", "q35", "-cpu", "max", "-m", memory]
    return {
        "artifact_schema_version": "qemu-prompt-bench/v1",
        "status": "pass",
        "command": command,
        "benchmarks": [{"command": command, "prompt": "smoke"}],
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="qemu-launch-profile-smoke-") as tmp:
        root = Path(tmp)
        inputs = root / "inputs"
        output_dir = root / "out"
        inputs.mkdir()

        (inputs / "qemu_prompt_bench_pass.json").write_text(json.dumps(artifact()) + "\n", encoding="utf-8")
        pass_cmd = [
            sys.executable,
            str(SCRIPT),
            str(inputs / "qemu_prompt_bench_pass.json"),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_launch_profile_audit_smoke_pass",
            "--require-memory",
            "--require-machine",
            "--require-cpu",
        ]
        subprocess.run(pass_cmd, cwd=ROOT, check=True)

        fail_payload = artifact()
        fail_payload["benchmarks"] = [{"command": ["qemu-system-x86_64", "-nic", "none", "-M", "q35", "-cpu", "max", "-m", "512M"]}]
        (inputs / "qemu_prompt_bench_fail.json").write_text(json.dumps(fail_payload) + "\n", encoding="utf-8")
        fail_cmd = [
            sys.executable,
            str(SCRIPT),
            str(inputs / "qemu_prompt_bench_fail.json"),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_launch_profile_audit_smoke_fail",
            "--require-memory",
        ]
        failed = subprocess.run(fail_cmd, cwd=ROOT, check=False)
        if failed.returncode == 0:
            raise SystemExit("expected launch profile drift artifact to fail")

        report = json.loads((output_dir / "qemu_launch_profile_audit_smoke_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        if "profile_drift" not in kinds:
            raise SystemExit(f"expected profile_drift finding, got {sorted(kinds)}")

        cross_inputs = root / "cross_inputs"
        cross_inputs.mkdir()
        (cross_inputs / "qemu_prompt_bench_q4.json").write_text(json.dumps(artifact("256M")) + "\n", encoding="utf-8")
        (cross_inputs / "qemu_prompt_bench_q8.json").write_text(json.dumps(artifact("512M")) + "\n", encoding="utf-8")
        cross_cmd = [
            sys.executable,
            str(SCRIPT),
            str(cross_inputs),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_launch_profile_audit_smoke_cross",
            "--require-memory",
            "--fail-on-cross-artifact-drift",
        ]
        cross_failed = subprocess.run(cross_cmd, cwd=ROOT, check=False)
        if cross_failed.returncode == 0:
            raise SystemExit("expected cross-artifact launch profile drift to fail")
        cross_report = json.loads((output_dir / "qemu_launch_profile_audit_smoke_cross.json").read_text(encoding="utf-8"))
        cross_kinds = {finding["kind"] for finding in cross_report["findings"]}
        if "cross_artifact_profile_drift" not in cross_kinds:
            raise SystemExit(f"expected cross_artifact_profile_drift finding, got {sorted(cross_kinds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
