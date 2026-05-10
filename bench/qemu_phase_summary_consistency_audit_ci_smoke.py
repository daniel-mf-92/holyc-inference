#!/usr/bin/env python3
"""CI smoke entry point for QEMU phase summary consistency audits."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qemu_phase_summary_consistency_audit


def main() -> int:
    return qemu_phase_summary_consistency_audit.main(
        [
            "bench/results/qemu_prompt_bench_latest.json",
            "--output-dir",
            "bench/results",
            "--output-stem",
            "qemu_phase_summary_consistency_audit_latest",
            "--min-artifacts",
            "1",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
