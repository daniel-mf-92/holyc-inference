#!/usr/bin/env python3
"""Smoke gate for qemu_prompt_echo_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_prompt_echo_audit


def make_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt": "smoke-short",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "prompt_bytes": 12,
        "guest_prompt_bytes": 12,
        "guest_prompt_bytes_match": True,
        "prompt_sha256": "a" * 64,
        "guest_prompt_sha256": "a" * 64,
        "guest_prompt_sha256_match": True,
    }
    row.update(overrides)
    return row


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        write_artifact(passing, [make_row(), make_row(prompt="smoke-long", prompt_bytes=18, guest_prompt_bytes=18)])
        out = root / "out"
        status = qemu_prompt_echo_audit.main([str(passing), "--output-dir", str(out), "--output-stem", "echo"])
        if status != 0:
            return status
        payload = json.loads((out / "echo.json").read_text(encoding="utf-8"))
        if payload["status"] != "pass" or payload["summary"]["rows"] != 2:
            return 1

        failing = root / "qemu_prompt_bench_fail.json"
        write_artifact(
            failing,
            [
                make_row(
                    guest_prompt_bytes=13,
                    guest_prompt_bytes_match=True,
                    guest_prompt_sha256="b" * 64,
                    guest_prompt_sha256_match=True,
                )
            ],
        )
        fail_out = root / "fail_out"
        fail_status = qemu_prompt_echo_audit.main([str(failing), "--output-dir", str(fail_out), "--output-stem", "echo_fail"])
        if fail_status == 0:
            return 1
        fail_payload = json.loads((fail_out / "echo_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_payload["findings"]}
        expected = {"prompt_bytes_drift", "prompt_bytes_match_flag_drift", "prompt_sha256_drift", "prompt_sha256_match_flag_drift"}
        if not expected.issubset(kinds):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
