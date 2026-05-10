#!/usr/bin/env python3
"""CI smoke for qemu_matrix_budget_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_matrix_budget_audit


def write_matrix(path: Path, *, missing_expected: bool = False) -> None:
    launches = []
    for build in ("baseline", "candidate"):
        for index, phase in enumerate(("warmup", "measured", "measured"), 1):
            launches.append(
                {
                    "build": build,
                    "profile": "ci-smoke",
                    "model": "synthetic",
                    "quantization": "Q4_0",
                    "launch_index": index,
                    "phase": phase,
                    "prompt_id": f"p{index}",
                    "prompt_sha256": "0" * 64,
                    "prompt_bytes": 16 + index,
                    "expected_tokens": None if missing_expected and build == "candidate" and index == 3 else 8,
                    "iteration": index,
                }
            )
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:00:00Z",
                "status": "pass",
                "summary": {"builds": 2, "launches": len(launches)},
                "builds": [
                    {
                        "build": "baseline",
                        "profile": "ci-smoke",
                        "model": "synthetic",
                        "quantization": "Q4_0",
                        "command_airgap_ok": True,
                    },
                    {
                        "build": "candidate",
                        "profile": "ci-smoke",
                        "model": "synthetic",
                        "quantization": "Q4_0",
                        "command_airgap_ok": True,
                    },
                ],
                "launches": launches,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ok_matrix = root / "qemu_benchmark_matrix_ok.json"
        bad_matrix = root / "qemu_benchmark_matrix_bad.json"
        output = Path("bench/results")
        write_matrix(ok_matrix)
        write_matrix(bad_matrix, missing_expected=True)

        ok = qemu_matrix_budget_audit.main(
            [
                str(ok_matrix),
                "--output-dir",
                str(output),
                "--output-stem",
                "qemu_matrix_budget_audit_smoke_latest",
                "--min-builds",
                "2",
                "--max-launches",
                "6",
                "--max-launches-per-build",
                "3",
                "--max-prompt-bytes-per-build",
                "60",
                "--require-expected-tokens",
                "--require-airgap",
            ]
        )
        if rc := require(ok == 0, "budget_pass_failed"):
            return rc

        payload = json.loads((output / "qemu_matrix_budget_audit_smoke_latest.json").read_text(encoding="utf-8"))
        if rc := require(payload["summary"]["launches"] == 6, "missing_launch_summary"):
            return rc
        if rc := require(payload["summary"]["expected_tokens_total"] == 48, "missing_expected_token_summary"):
            return rc
        junit = ET.parse(output / "qemu_matrix_budget_audit_smoke_latest_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_qemu_matrix_budget_audit", "missing_junit"):
            return rc

        failed = qemu_matrix_budget_audit.main(
            [
                str(bad_matrix),
                "--output-dir",
                str(root / "failed"),
                "--output-stem",
                "budget",
                "--max-launches-per-build",
                "2",
                "--require-expected-tokens",
            ]
        )
        if rc := require(failed == 1, "budget_failure_passed"):
            return rc
        failed_payload = json.loads((root / "failed" / "budget.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_payload["findings"]}
        if rc := require({"max_launches_per_build", "missing_expected_tokens"}.issubset(kinds), "missing_budget_findings"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
