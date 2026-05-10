#!/usr/bin/env python3
"""CI smoke test for dataset_manifest_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
RESULTS = BENCH / "results" / "datasets"
MANIFEST = RESULTS / "smoke_curated.manifest.json"
PACK_MANIFEST = RESULTS / "smoke_curated.hceval.manifest.json"


def run_smoke() -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(BENCH / "dataset_manifest_audit.py"),
        "--manifest",
        str(MANIFEST),
        "--pack-manifest",
        str(PACK_MANIFEST),
        "--root",
        str(ROOT),
        "--output",
        str(RESULTS / "dataset_manifest_audit_smoke_latest.json"),
        "--csv",
        str(RESULTS / "dataset_manifest_audit_smoke_latest.csv"),
        "--markdown",
        str(RESULTS / "dataset_manifest_audit_smoke_latest.md"),
        "--junit",
        str(RESULTS / "dataset_manifest_audit_smoke_latest_junit.xml"),
        "--require-pack-manifest",
        "--fail-on-findings",
    ]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def assert_smoke_outputs() -> None:
    report = json.loads((RESULTS / "dataset_manifest_audit_smoke_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass", report
    assert report["record_count"] == 3, report
    assert (
        report["curated_output"]["actual_normalized_sha256"]
        == report["curated_output"]["expected_normalized_sha256"]
    ), report
    assert report["curated_output"]["actual_record_ids"] == report["pack_manifest"]["records"], report
    assert report["source"]["actual_sha256"] == report["source"]["expected_sha256"], report
    assert report["pack_manifest"]["pack_output"]["actual_sha256"] == report["pack_manifest"]["binary_sha256"], report
    assert not report["findings"], report
    assert (RESULTS / "dataset_manifest_audit_smoke_latest.csv").exists()
    assert (RESULTS / "dataset_manifest_audit_smoke_latest.md").exists()
    assert (RESULTS / "dataset_manifest_audit_smoke_latest_junit.xml").exists()


def assert_failure_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="dataset-manifest-audit-") as tmp:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["pack_output"] = "missing.hceval"
        manifest_path = Path(tmp) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        output = Path(tmp) / "audit.json"
        cmd = [
            sys.executable,
            str(BENCH / "dataset_manifest_audit.py"),
            "--manifest",
            str(manifest_path),
            "--root",
            str(ROOT),
            "--output",
            str(output),
            "--require-pack-manifest",
            "--fail-on-findings",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        assert any(item["kind"] == "missing_pack_manifest" for item in report["findings"]), report


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    proc = run_smoke()
    if proc.returncode:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    assert_smoke_outputs()
    assert_failure_gate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
