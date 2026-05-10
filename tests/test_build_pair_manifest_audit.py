#!/usr/bin/env python3
"""Tests for build-pair manifest audits."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import build_pair_manifest_audit


def pair(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "key": "ci/tiny/Q4_0/suite/command/launch/env",
        "baseline_source": "bench/results/base.json",
        "candidate_source": "bench/results/head.json",
        "baseline_commit": "basecommit",
        "candidate_commit": "headcommit",
        "baseline_generated_at": "2026-04-30T00:00:00Z",
        "candidate_generated_at": "2026-05-01T00:00:00Z",
        "baseline_measured_runs": 4,
        "candidate_measured_runs": 4,
        "build_compare_args": [
            "--input",
            "base-20260430-basecommit=bench/results/base.json",
            "--input",
            "head-20260501-headcommit=bench/results/head.json",
        ],
    }
    payload.update(overrides)
    return payload


def parse_args(extra: list[str]) -> object:
    return build_pair_manifest_audit.build_parser().parse_args(extra)


def test_audit_accepts_valid_build_pair_manifest(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": [pair()]}), encoding="utf-8")

    pairs, findings = build_pair_manifest_audit.audit([manifest], parse_args([str(manifest), "--min-measured-runs", "4"]))

    assert findings == []
    assert len(pairs) == 1
    assert pairs[0].candidate_commit == "headcommit"


def test_audit_flags_pair_manifest_drift(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps(
            {
                "pairs": [
                    pair(
                        candidate_source="bench/results/base.json",
                        candidate_commit="basecommit",
                        candidate_generated_at="2026-04-29T00:00:00Z",
                        candidate_measured_runs=0,
                        build_compare_args="--input base=bench/results/base.json --input head=missing.json",
                    ),
                    pair(
                        candidate_source="bench/results/head.json",
                        build_compare_args="--input base=bench/results/base.json --input head=missing.json",
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    pairs, findings = build_pair_manifest_audit.audit([manifest], parse_args([str(manifest), "--min-measured-runs", "2"]))

    assert len(pairs) == 2
    assert {
        "same_source",
        "same_commit",
        "candidate_older_than_baseline",
        "insufficient_runs",
        "missing_build_compare_source",
    } <= {finding.kind for finding in findings}


def test_cli_writes_json_csv_markdown_and_junit(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "pairs.csv"
    output = tmp_path / "out"
    fieldnames = list(pair().keys())
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = pair()
        row["build_compare_args"] = "--input base=bench/results/base.json --input head=bench/results/head.json"
        writer.writerow(row)

    status = build_pair_manifest_audit.main([str(manifest), "--output-dir", str(output), "--output-stem", "pair_manifest"])

    assert status == 0
    payload = json.loads((output / "pair_manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output / "pair_manifest.csv").open(encoding="utf-8", newline="")))
    finding_rows = list(csv.DictReader((output / "pair_manifest_findings.csv").open(encoding="utf-8", newline="")))
    junit = ET.parse(output / "pair_manifest_junit.xml").getroot()

    assert payload["status"] == "pass"
    assert payload["summary"]["pairs"] == 1
    assert rows[0]["baseline_source"] == "bench/results/base.json"
    assert finding_rows == []
    assert "No build-pair manifest findings." in (output / "pair_manifest.md").read_text(encoding="utf-8")
    assert junit.attrib["name"] == "holyc_build_pair_manifest_audit"
    assert junit.attrib["failures"] == "0"


def test_cli_fails_when_min_pairs_not_met(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "pairs.json"
    output = tmp_path / "out"
    manifest.write_text(json.dumps({"pairs": []}), encoding="utf-8")

    status = build_pair_manifest_audit.main([str(manifest), "--output-dir", str(output), "--output-stem", "pair_manifest"])

    assert status == 1
    payload = json.loads((output / "pair_manifest.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_pairs"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory(prefix="holyc-build-pair-manifest-audit-") as tmp:
        root = Path(tmp)
        test_audit_accepts_valid_build_pair_manifest(root / "valid")
        test_audit_flags_pair_manifest_drift(root / "drift")
        test_cli_writes_json_csv_markdown_and_junit(root / "cli")
        test_cli_fails_when_min_pairs_not_met(root / "empty")
    print("test_build_pair_manifest_audit=ok")
