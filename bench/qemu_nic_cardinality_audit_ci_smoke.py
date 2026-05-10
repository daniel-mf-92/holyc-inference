#!/usr/bin/env python3
"""Smoke gate for QEMU NIC cardinality audit."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_nic_cardinality_audit


def row(command: list[str], launch_index: int = 1) -> dict[str, object]:
    return {
        "prompt": "smoke-short",
        "phase": "measured",
        "launch_index": launch_index,
        "command": command,
    }


def write_artifact(path: Path, command: list[str], rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"command": command, "warmups": [], "benchmarks": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        safe = tmp / "qemu_prompt_bench_safe.json"
        safe_command = ["qemu-system-x86_64", "-display", "none", "-nic", "none", "-m", "512M"]
        write_artifact(safe, safe_command, [row(safe_command)])
        safe_out = tmp / "safe_out"
        if qemu_nic_cardinality_audit.main(
            [
                str(safe),
                "--output-dir",
                str(safe_out),
                "--output-stem",
                "qemu_nic_cardinality_audit_smoke",
                "--require-top-command",
                "--require-qemu-system-executable",
            ]
        ) != 0:
            return 1

        bad = tmp / "qemu_prompt_bench_bad.json"
        write_artifact(
            bad,
            ["qemu-system-x86_64", "-nic", "none", "-nic", "none"],
            [
                row(["qemu-system-x86_64", "-m", "512M"], 1),
                row(["python3", "fake-qemu", "-nic", "none"], 2),
                row(["qemu-system-x86_64", "-nic", "user"], 3),
            ],
        )
        bad_out = tmp / "bad_out"
        if qemu_nic_cardinality_audit.main(
            [
                str(bad),
                "--output-dir",
                str(bad_out),
                "--output-stem",
                "qemu_nic_cardinality_audit_smoke",
                "--require-top-command",
                "--require-qemu-system-executable",
            ]
        ) == 0:
            return 1
        report = json.loads((bad_out / "qemu_nic_cardinality_audit_smoke.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        expected = {"nic_none_cardinality", "non_qemu_system_executable", "airgap_violation"}
        if not expected <= kinds:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
