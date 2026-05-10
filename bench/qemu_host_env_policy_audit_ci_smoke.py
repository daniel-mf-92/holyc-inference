#!/usr/bin/env python3
"""Smoke gate for host environment policy audits."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_host_env_policy_audit


def write_artifact(path: Path, environment: dict[str, str]) -> None:
    payload = {
        "benchmarks": [
            {
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "phase": "measured",
                "exit_class": "ok",
                "environment": environment,
            }
        ]
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-host-env-policy-") as tmp:
        root = Path(tmp)
        clean = root / "qemu_prompt_bench_clean.json"
        dirty = root / "qemu_prompt_bench_dirty.json"
        write_artifact(clean, {"PATH": "/usr/bin:/bin", "QEMU_AUDIO_DRV": "none"})
        write_artifact(dirty, {"HTTP_PROXY": "http://127.0.0.1:8080", "PATH": "/usr/bin:/bin"})
        clean_status = qemu_host_env_policy_audit.main(
            [
                str(clean),
                "--require-environment",
                "--output-dir",
                str(root / "clean"),
                "--output-stem",
                "qemu_host_env_policy_audit_clean",
            ]
        )
        dirty_status = qemu_host_env_policy_audit.main(
            [
                str(dirty),
                "--require-environment",
                "--fail-on-url-values",
                "--output-dir",
                str(root / "dirty"),
                "--output-stem",
                "qemu_host_env_policy_audit_dirty",
            ]
        )
        if clean_status != 0:
            raise SystemExit("clean host environment policy audit failed")
        if dirty_status == 0:
            raise SystemExit("dirty host environment policy audit unexpectedly passed")
    print("qemu_host_env_policy_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
