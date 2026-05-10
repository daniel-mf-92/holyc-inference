#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_identity_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_identity_audit
import qemu_prompt_bench


COMMAND = ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def write_artifact(path: Path, *, row_profile: str = "ci-airgap-smoke") -> None:
    row = {
        "benchmark": "qemu_prompt",
        "profile": row_profile,
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "launch_index": 1,
        "prompt": "alpha",
        "iteration": 1,
        "commit": "abc123",
        "command": COMMAND,
        "command_sha256": qemu_prompt_bench.command_hash(COMMAND),
    }
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:00:00Z",
                "status": "pass",
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "commit": "abc123",
                "command": COMMAND,
                "command_sha256": qemu_prompt_bench.command_hash(COMMAND),
                "benchmarks": [row],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-identity-ci-") as tmp:
        tmp_path = Path(tmp)
        good = tmp_path / "qemu_prompt_bench_good.json"
        bad = tmp_path / "qemu_prompt_bench_bad.json"
        write_artifact(good)
        write_artifact(bad, row_profile="drift")

        artifact, findings = qemu_identity_audit.audit_artifact(good)
        if rc := require(artifact.status == "pass", "consistent_identity_rejected"):
            return rc
        if rc := require(findings == [], "unexpected_consistent_identity_findings"):
            return rc

        artifact, findings = qemu_identity_audit.audit_artifact(bad)
        if rc := require(artifact.status == "fail", "drift_identity_accepted"):
            return rc
        if rc := require("identity_drift" in {finding.kind for finding in findings}, "missing_identity_drift_finding"):
            return rc

        output_dir = tmp_path / "out"
        status = qemu_identity_audit.main(
            [
                str(good),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_identity_audit_smoke",
            ]
        )
        if rc := require(status == 0, "identity_audit_cli_failed"):
            return rc
        payload = json.loads((output_dir / "qemu_identity_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(payload["summary"]["command_hashes_checked"] == 2, "missing_command_hash_coverage"):
            return rc
        if rc := require((output_dir / "qemu_identity_audit_smoke_junit.xml").exists(), "missing_junit_output"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
