#!/usr/bin/env python3
"""Smoke gate for dataset_license_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_license_audit

RESULTS = ROOT / "bench" / "results" / "datasets"


def write_manifest(path: Path, *, license_text: str, source_url: str = "https://example.org/eval") -> None:
    path.write_text(
        json.dumps(
            {
                "format": "hceval-curated-jsonl",
                "source_name": "synthetic-license-smoke",
                "source_version": "v1",
                "license": license_text,
                "source_url": source_url,
                "record_count": 2,
                "dataset_counts": {"arc-smoke": 2},
                "split_counts": {"validation": 2},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        allowed = tmp_path / "allowed.manifest.json"
        denied = tmp_path / "denied.manifest.json"
        write_manifest(allowed, license_text="Apache-2.0", source_url="https://example.org/allowed")
        write_manifest(denied, license_text="research-only", source_url="ftp://example.org/denied")

        pass_status = dataset_license_audit.main(
            [
                "--input",
                str(allowed),
                "--output-dir",
                str(RESULTS),
                "--output-stem",
                "dataset_license_audit_smoke_latest",
                "--allow-license",
                "Apache-2.0",
                "--require-source-url",
                "--min-records",
                "1",
                "--fail-on-findings",
            ]
        )
        fail_status = dataset_license_audit.main(
            [
                "--input",
                str(denied),
                "--output-dir",
                str(tmp_path / "fail"),
                "--output-stem",
                "dataset_license_audit_fail",
                "--allow-license",
                "Apache-2.0",
                "--require-source-url",
                "--fail-on-findings",
            ]
        )
        if pass_status != 0 or fail_status == 0:
            return 1
    print("dataset_license_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
