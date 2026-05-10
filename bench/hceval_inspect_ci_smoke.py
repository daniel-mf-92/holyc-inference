#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for hceval_inspect.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
FIXTURE = ROOT / "bench" / "results" / "datasets" / "smoke_eval.hceval"
MANIFEST = ROOT / "bench" / "results" / "datasets" / "smoke_eval.manifest.json"


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def run_inspect(
    output_dir: Path,
    *,
    manifest: Path = MANIFEST,
    expected_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(BENCH / "hceval_inspect.py"),
        "--input",
        str(FIXTURE),
        "--manifest",
        str(manifest),
        "--output",
        str(output_dir / "hceval_inspect_smoke_latest.json"),
        "--markdown",
        str(output_dir / "hceval_inspect_smoke_latest.md"),
        "--csv",
        str(output_dir / "hceval_inspect_smoke_latest.csv"),
        "--fingerprints-csv",
        str(output_dir / "hceval_inspect_smoke_fingerprints_latest.csv"),
        "--junit",
        str(output_dir / "hceval_inspect_smoke_latest_junit.xml"),
        "--max-prompt-bytes",
        "4096",
        "--max-choice-bytes",
        "1024",
        "--max-record-payload-bytes",
        "8192",
    ]
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0 and not expected_failure:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-hceval-inspect-") as tmp:
        root = Path(tmp)
        pass_dir = root / "pass"
        pass_dir.mkdir()
        completed = run_inspect(pass_dir)
        if completed.returncode:
            return completed.returncode

        report = json.loads((pass_dir / "hceval_inspect_smoke_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_hceval_inspect_status"):
            return rc
        if rc := require(report["record_count"] == 3, "unexpected_hceval_inspect_record_count"):
            return rc
        if rc := require(report["choice_count_histogram"] == {"4": 3}, "unexpected_hceval_choice_histogram"):
            return rc
        if rc := require(report["binary_layout"]["binary_bytes"] == FIXTURE.stat().st_size, "hceval_binary_size_mismatch"):
            return rc
        if rc := require(len(report["record_fingerprints"]) == 3, "missing_hceval_record_fingerprints"):
            return rc
        rows = list(csv.DictReader((pass_dir / "hceval_inspect_smoke_latest.csv").open(encoding="utf-8", newline="")))
        if rc := require(len(rows) == 3, "unexpected_hceval_inspect_csv_rows"):
            return rc
        fingerprint_rows = list(
            csv.DictReader((pass_dir / "hceval_inspect_smoke_fingerprints_latest.csv").open(encoding="utf-8", newline=""))
        )
        if rc := require(len(fingerprint_rows) == 3, "unexpected_hceval_fingerprint_csv_rows"):
            return rc
        if rc := require(
            "No findings." in (pass_dir / "hceval_inspect_smoke_latest.md").read_text(encoding="utf-8"),
            "missing_hceval_inspect_markdown_pass",
        ):
            return rc
        junit_text = (pass_dir / "hceval_inspect_smoke_latest_junit.xml").read_text(encoding="utf-8")
        if rc := require('failures="0"' in junit_text, "unexpected_hceval_inspect_junit_failure"):
            return rc

        bad_manifest = root / "bad_manifest.json"
        manifest_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest_payload["binary_sha256"] = "0" * 64
        bad_manifest.write_text(json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="utf-8")

        fail_dir = root / "fail"
        fail_dir.mkdir()
        failed = run_inspect(fail_dir, manifest=bad_manifest, expected_failure=True)
        if rc := require(failed.returncode == 1, "hceval_manifest_mismatch_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "hceval_inspect_smoke_latest.json").read_text(encoding="utf-8"))
        if rc := require("manifest binary_sha256 does not match input" in fail_report["findings"], "missing_manifest_mismatch_finding"):
            return rc
        fail_junit_text = (fail_dir / "hceval_inspect_smoke_latest_junit.xml").read_text(encoding="utf-8")
        if rc := require('failures="1"' in fail_junit_text, "missing_hceval_inspect_junit_failure"):
            return rc

    print("hceval_inspect_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
