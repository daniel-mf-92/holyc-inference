#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_provenance_audit.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def run_command(command: list[str], *, expected_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0 and not expected_failure:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def finding_kinds(report_path: Path) -> set[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        finding["kind"]
        for artifact in report["artifacts"]
        for finding in artifact["findings"]
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-provenance-") as tmp:
        tmp_path = Path(tmp)
        curated = tmp_path / "smoke_curated.jsonl"
        manifest = tmp_path / "smoke_curated.manifest.json"
        pass_dir = tmp_path / "pass"

        curate_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_curate.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(curated),
            "--manifest",
            str(manifest),
            "--source-name",
            "smoke-eval",
            "--source-version",
            "synthetic",
            "--source-license",
            "synthetic-smoke",
            "--source-url",
            "https://datasets.example/eval/smoke-eval",
            "--min-choices",
            "4",
            "--max-choices",
            "4",
            "--max-records",
            "3",
            "--max-records-per-provenance",
            "1",
            "--dedupe-within-split-payloads",
        ]
        completed = run_command(curate_command)
        if completed.returncode != 0:
            return completed.returncode

        pass_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(manifest),
            "--output-dir",
            str(pass_dir),
            "--allow-license",
            "synthetic-smoke",
            "--allow-source-url-scheme",
            "https",
            "--allow-source-url-host",
            "datasets.example",
            "--allow-source-url-path-prefix",
            "/eval/",
            "--max-provenance-pct",
            "40",
            "--max-majority-answer-pct",
            "100",
            "--max-dataset-majority-answer-pct",
            "100",
            "--max-split-majority-answer-pct",
            "100",
            "--max-dataset-split-majority-answer-pct",
            "100",
            "--fail-on-findings",
        ]
        completed = run_command(pass_command)
        if completed.returncode != 0:
            return completed.returncode

        report_path = pass_dir / "dataset_provenance_audit_latest.json"
        markdown_path = pass_dir / "dataset_provenance_audit_latest.md"
        csv_path = pass_dir / "dataset_provenance_audit_latest.csv"
        record_csv_path = pass_dir / "dataset_provenance_audit_records_latest.csv"
        junit_path = pass_dir / "dataset_provenance_audit_junit_latest.xml"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_provenance_status"):
            return rc
        if rc := require(len(report["artifacts"]) == 1, "unexpected_provenance_artifact_count"):
            return rc
        artifact = report["artifacts"][0]
        if rc := require(artifact["source_url_scheme"] == "https", "missing_source_url_scheme"):
            return rc
        if rc := require(artifact["source_url_host"] == "datasets.example", "missing_source_url_host"):
            return rc
        if rc := require(artifact["source_url_path"] == "/eval/smoke-eval", "missing_source_url_path"):
            return rc
        if rc := require(artifact["provenance_counts"], "missing_provenance_counts"):
            return rc
        if rc := require(artifact["majority_provenance_pct"] <= 40.0, "unexpected_provenance_skew"):
            return rc
        if rc := require("Dataset Provenance Audit" in markdown_path.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(csv_rows) == 1, "unexpected_csv_rows"):
            return rc
        if rc := require(csv_rows[0]["source_url_host"] == "datasets.example", "unexpected_csv_source_host"):
            return rc
        record_rows = list(csv.DictReader(record_csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(record_rows) == 3, "unexpected_record_csv_rows"):
            return rc
        if rc := require(record_rows[0]["input_sha256"], "missing_record_input_sha256"):
            return rc
        if rc := require(record_rows[0]["record_payload_bytes"], "missing_record_payload_bytes"):
            return rc
        if rc := require(record_rows[0]["provenance_pct"], "missing_record_provenance_pct"):
            return rc
        junit_root = ET.parse(junit_path).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_dataset_provenance_audit", "missing_junit"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        bad_license_dir = tmp_path / "bad-license"
        bad_license_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(manifest),
            "--output-dir",
            str(bad_license_dir),
            "--deny-license",
            "synthetic-smoke",
            "--fail-on-findings",
        ]
        completed = run_command(bad_license_command, expected_failure=True)
        if completed.returncode == 0:
            print("denied_license_not_rejected=true", file=sys.stderr)
            return 1
        if rc := require(
            "license_denied" in finding_kinds(bad_license_dir / "dataset_provenance_audit_latest.json"),
            "missing_license_denied_finding",
        ):
            return rc

        bad_host_manifest = tmp_path / "bad_host.manifest.json"
        bad_host_report = json.loads(manifest.read_text(encoding="utf-8"))
        bad_host_report["source_url"] = "https://mirror.example/eval/smoke-eval"
        bad_host_manifest.write_text(json.dumps(bad_host_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bad_host_dir = tmp_path / "bad-host"
        bad_host_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(bad_host_manifest),
            "--output-dir",
            str(bad_host_dir),
            "--allow-source-url-host",
            "datasets.example",
            "--fail-on-findings",
        ]
        completed = run_command(bad_host_command, expected_failure=True)
        if completed.returncode == 0:
            print("unallowed_source_url_host_not_rejected=true", file=sys.stderr)
            return 1
        if rc := require(
            "source_url_host_not_allowed" in finding_kinds(bad_host_dir / "dataset_provenance_audit_latest.json"),
            "missing_source_url_host_not_allowed_finding",
        ):
            return rc

    print("dataset_provenance_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
