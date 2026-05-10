#!/usr/bin/env python3
"""CI smoke entry point for the host-side HCEval bundle audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import hceval_bundle_audit

RESULTS = ROOT / "bench" / "results" / "datasets"
SMOKE = ROOT / "bench" / "results" / "hceval_span_smoke_latest.hceval"


def main() -> int:
    return hceval_bundle_audit.main(
        [
            "--input",
            str(SMOKE),
            "--require-manifest",
            "--output",
            str(RESULTS / "hceval_bundle_audit_smoke_latest.json"),
            "--markdown",
            str(RESULTS / "hceval_bundle_audit_smoke_latest.md"),
            "--csv",
            str(RESULTS / "hceval_bundle_audit_smoke_latest.csv"),
            "--junit",
            str(RESULTS / "hceval_bundle_audit_smoke_latest_junit.xml"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
