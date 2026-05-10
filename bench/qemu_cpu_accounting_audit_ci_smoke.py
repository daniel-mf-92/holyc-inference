#!/usr/bin/env python3
"""Smoke gate for qemu_cpu_accounting_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qemu_cpu_accounting_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 32,
        "wall_elapsed_us": 50_000,
        "host_child_user_cpu_us": 12_000,
        "host_child_system_cpu_us": 3_000,
        "host_child_cpu_us": 15_000,
        "host_child_cpu_pct": 30.0,
        "host_child_tok_per_cpu_s": 32 * 1_000_000.0 / 15_000,
        "host_child_peak_rss_bytes": 4_194_304,
    }
    row.update(overrides)
    return row


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        artifact = tmp / "qemu_prompt_bench_latest.json"
        output_dir = tmp / "out"
        write_artifact(artifact, [artifact_row()])
        status = qemu_cpu_accounting_audit.main(
            [
                str(artifact),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_cpu_accounting_audit_smoke",
                "--require-cpu-metrics",
                "--max-host-child-cpu-pct",
                "80",
                "--min-host-child-tok-per-cpu-s",
                "1000",
            ]
        )
        if status != 0:
            return status
        payload = json.loads((output_dir / "qemu_cpu_accounting_audit_smoke.json").read_text(encoding="utf-8"))
        if payload["status"] != "pass" or payload["summary"]["checks"] < 3:
            return 1

        write_artifact(artifact, [artifact_row(host_child_cpu_us=1, host_child_cpu_pct=95, host_child_tok_per_cpu_s=1)])
        status = qemu_cpu_accounting_audit.main(
            [
                str(artifact),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_cpu_accounting_audit_smoke_fail",
                "--require-cpu-metrics",
                "--max-host-child-cpu-pct",
                "80",
                "--min-host-child-tok-per-cpu-s",
                "1000",
            ]
        )
        if status == 0:
            return 1
        fail_payload = json.loads((output_dir / "qemu_cpu_accounting_audit_smoke_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_payload["findings"]}
        expected = {"metric_drift", "max_host_child_cpu_pct", "min_host_child_tok_per_cpu_s"}
        return 0 if expected <= kinds else 1


if __name__ == "__main__":
    raise SystemExit(main())
