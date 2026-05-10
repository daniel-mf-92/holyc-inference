#!/usr/bin/env python3
"""Host-side checks for eval dataset license policy audits."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_license_audit.py"
spec = importlib.util.spec_from_file_location("dataset_license_audit", AUDIT_PATH)
dataset_license_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_license_audit"] = dataset_license_audit
spec.loader.exec_module(dataset_license_audit)


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_manifest() -> dict:
    return {
        "format": "hceval-curated-jsonl",
        "source_name": "arc",
        "source_version": "v1",
        "license": "Apache-2.0",
        "source_url": "https://example.org/arc",
        "record_count": 12,
        "dataset_counts": {"arc": 12},
        "split_counts": {"validation": 12},
    }


def test_license_policy_pass_writes_artifact_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        output_dir = root / "out"
        write_manifest(manifest, base_manifest())

        status = dataset_license_audit.main(
            [
                "--input",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--allow-license",
                "Apache-2.0",
                "--require-source-url",
                "--min-records",
                "1",
                "--fail-on-findings",
            ]
        )

        report = json.loads((output_dir / "dataset_license_audit_latest.json").read_text(encoding="utf-8"))
        rows = list(csv.DictReader((output_dir / "dataset_license_audit_latest.csv").open(encoding="utf-8")))
        assert status == 0
        assert report["status"] == "pass"
        assert report["finding_count"] == 0
        assert rows[0]["normalized_license"] == "apache-2.0"
        assert rows[0]["source_url_host"] == "example.org"


def test_license_policy_fails_denied_missing_and_scheme_findings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        output_dir = root / "out"
        payload = base_manifest()
        payload.update(
            {
                "source_name": "unknown",
                "license": "research-only",
                "source_url": "ftp://example.org/arc",
                "record_count": 0,
            }
        )
        write_manifest(manifest, payload)

        status = dataset_license_audit.main(
            [
                "--input",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--allow-license",
                "Apache-2.0",
                "--require-source-url",
                "--min-records",
                "1",
                "--fail-on-findings",
            ]
        )

        report = json.loads((output_dir / "dataset_license_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert {
            "missing_source_name",
            "record_count_below_min",
            "license_not_allowed",
            "license_denied",
            "source_url_scheme_not_allowed",
        }.issubset(kinds)


if __name__ == "__main__":
    test_license_policy_pass_writes_artifact_csv()
    test_license_policy_fails_denied_missing_and_scheme_findings()
    print("dataset_license_audit_tests=ok")
