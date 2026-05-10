#!/usr/bin/env python3
"""Smoke gate for build_pair_manifest_audit.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_pair_manifest_audit


def main() -> int:
    manifest = Path("bench/fixtures/build_pair_manifest_audit/pass.json")
    status = build_pair_manifest_audit.main(
        [
            str(manifest),
            "--output-dir",
            "bench/results",
            "--output-stem",
            "build_pair_manifest_audit_latest",
            "--min-measured-runs",
            "4",
        ]
    )
    if status != 0:
        return status
    print("build_pair_manifest_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
