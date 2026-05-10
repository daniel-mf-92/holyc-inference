#!/usr/bin/env python3
"""Smoke gate for quant_block_compare.py."""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"


def half_bits(value: float) -> bytes:
    return struct.pack("<e", value)


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        reference = RESULTS / "quant_block_compare_smoke_reference.q4_0"
        candidate_pass = RESULTS / "quant_block_compare_smoke_candidate_pass.q4_0"
        candidate_fail = tmp / "candidate-fail.q4"

        block = half_bits(1.0) + bytes([0x88] * 16)
        reference.write_bytes(block + block)
        candidate_pass.write_bytes(block + block)
        candidate_fail.write_bytes(block + half_bits(1.0) + bytes([0x89] + [0x88] * 15))

        pass_run = run(
            [
                sys.executable,
                "bench/quant_block_compare.py",
                "--format",
                "q4_0",
                "--reference",
                str(reference),
                "--candidate",
                str(candidate_pass),
                "--output",
                "bench/results/quant_block_compare_smoke_latest.json",
                "--csv",
                "bench/results/quant_block_compare_smoke_latest.csv",
                "--markdown",
                "bench/results/quant_block_compare_smoke_latest.md",
                "--junit",
                "bench/results/quant_block_compare_smoke_latest_junit.xml",
                "--max-mismatches",
                "0",
            ]
        )
        if pass_run.returncode != 0:
            sys.stderr.write(pass_run.stdout)
            sys.stderr.write(pass_run.stderr)
            return pass_run.returncode

        fail_run = run(
            [
                sys.executable,
                "bench/quant_block_compare.py",
                "--format",
                "q4_0",
                "--reference",
                str(reference),
                "--candidate",
                str(candidate_fail),
                "--max-mismatches",
                "0",
            ]
        )
        if fail_run.returncode == 0:
            sys.stderr.write("expected mismatched Q4_0 stream to fail\n")
            return 1
        if "total mismatches" not in fail_run.stdout:
            sys.stderr.write(fail_run.stdout)
            sys.stderr.write(fail_run.stderr)
            return 1

        allowed_run = run(
            [
                sys.executable,
                "bench/quant_block_compare.py",
                "--format",
                "q4_0",
                "--reference",
                str(reference),
                "--candidate",
                str(candidate_fail),
                "--allow-mismatches",
            ]
        )
        if allowed_run.returncode != 0:
            sys.stderr.write("expected --allow-mismatches to keep telemetry-only compare non-failing\n")
            sys.stderr.write(allowed_run.stdout)
            sys.stderr.write(allowed_run.stderr)
            return allowed_run.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
