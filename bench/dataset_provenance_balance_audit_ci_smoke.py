#!/usr/bin/env python3
"""CI smoke gate for dataset_provenance_balance_audit.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
RESULTS = BENCH / "results" / "datasets"
SAMPLE = BENCH / "datasets" / "samples" / "smoke_eval.jsonl"


def run_command(command: list[str], *, expected_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0 and not expected_failure:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def audit_command(*extra_args: str, output_dir: Path = RESULTS, stem: str = "dataset_provenance_balance_audit_smoke_latest") -> list[str]:
    record_stem = stem.removesuffix("_latest") + "_records_latest" if stem.endswith("_latest") else f"{stem}_records"
    return [
        sys.executable,
        str(BENCH / "dataset_provenance_balance_audit.py"),
        "--input",
        str(SAMPLE),
        "--output",
        str(output_dir / f"{stem}.json"),
        "--markdown",
        str(output_dir / f"{stem}.md"),
        "--csv",
        str(output_dir / f"{stem}.csv"),
        "--record-csv",
        str(output_dir / f"{record_stem}.csv"),
        "--findings-csv",
        str(output_dir / f"{stem}_findings.csv"),
        "--junit",
        str(output_dir / f"{stem}_junit.xml"),
        "--require-provenance",
        "--require-provenance-source",
        "synthetic HellaSwag-shaped smoke row",
        "--require-provenance-source",
        "synthetic ARC-shaped smoke row",
        "--require-provenance-source",
        "synthetic TruthfulQA-shaped smoke row",
        "--min-provenance-sources",
        "3",
        "--min-records-per-provenance",
        "1",
        "--max-provenance-pct",
        "34",
        "--max-dataset-split-provenance-pct",
        "100",
        *extra_args,
    ]


def assert_pass_outputs() -> int:
    report = json.loads((RESULTS / "dataset_provenance_balance_audit_smoke_latest.json").read_text(encoding="utf-8"))
    if rc := require(report["status"] == "pass", "unexpected_provenance_balance_status"):
        return rc
    if rc := require(report["record_count"] == 3, "unexpected_provenance_balance_record_count"):
        return rc
    if rc := require(report["provenance_source_count"] == 3, "unexpected_provenance_source_count"):
        return rc
    if rc := require(
        report["dataset_split_counts"] == {
            "arc-smoke": {"validation": 1},
            "hellaswag-smoke": {"validation": 1},
            "truthfulqa-smoke": {"validation": 1},
        },
        "unexpected_dataset_split_counts",
    ):
        return rc
    if rc := require(len(report["distribution"]) == 6, "unexpected_provenance_distribution_rows"):
        return rc
    if rc := require(len(report["record_telemetry"]) == 3, "unexpected_provenance_record_rows"):
        return rc
    if rc := require(not report["findings"], "unexpected_provenance_findings"):
        return rc
    if rc := require(
        "No provenance balance findings." in (RESULTS / "dataset_provenance_balance_audit_smoke_latest.md").read_text(encoding="utf-8"),
        "missing_provenance_balance_markdown",
    ):
        return rc

    csv_rows = list(csv.DictReader((RESULTS / "dataset_provenance_balance_audit_smoke_latest.csv").open(encoding="utf-8", newline="")))
    if rc := require({row["scope"] for row in csv_rows} == {"provenance", "dataset_split_provenance"}, "unexpected_csv_scopes"):
        return rc

    record_rows = list(
        csv.DictReader((RESULTS / "dataset_provenance_balance_audit_smoke_records_latest.csv").open(encoding="utf-8", newline=""))
    )
    if rc := require(all(row["normalized_payload_sha256"] for row in record_rows), "missing_record_payload_hash"):
        return rc

    junit = (RESULTS / "dataset_provenance_balance_audit_smoke_latest_junit.xml").read_text(encoding="utf-8")
    if rc := require('name="dataset_provenance_balance_audit"' in junit, "missing_provenance_balance_junit"):
        return rc
    if rc := require('failures="0"' in junit, "unexpected_provenance_balance_junit_failure"):
        return rc
    return 0


def assert_failure_gates() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-provenance-balance-") as tmp:
        out = Path(tmp)
        failed = run_command(
            audit_command("--max-provenance-pct", "30", output_dir=out, stem="skewed"),
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "provenance_balance_skew_not_rejected"):
            return rc
        failed_report = json.loads((out / "skewed.json").read_text(encoding="utf-8"))
        if rc := require(
            {finding["kind"] for finding in failed_report["findings"]} == {"max_provenance_pct"},
            "missing_max_provenance_pct_finding",
        ):
            return rc

        missing_source = run_command(
            audit_command(
                "--require-provenance-source",
                "missing smoke provenance source",
                output_dir=out,
                stem="missing-source",
            ),
            expected_failure=True,
        )
        if rc := require(missing_source.returncode == 1, "missing_required_provenance_not_rejected"):
            return rc
        missing_report = json.loads((out / "missing-source.json").read_text(encoding="utf-8"))
        if rc := require(
            "missing_provenance_source" in {finding["kind"] for finding in missing_report["findings"]},
            "missing_required_source_finding",
        ):
            return rc
    return 0


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    passed = run_command(audit_command())
    if passed.returncode != 0:
        return passed.returncode
    if rc := assert_pass_outputs():
        return rc
    if rc := assert_failure_gates():
        return rc

    print("dataset_provenance_balance_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
