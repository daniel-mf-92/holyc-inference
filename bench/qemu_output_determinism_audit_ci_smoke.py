#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU output determinism audit."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt": "short",
        "commit": "abc123",
        "seed": 0,
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 8,
        "output": "hello world",
    }
    payload.update(overrides)
    return payload


def run_audit(artifact: Path, output_dir: Path, stem: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_output_determinism_audit.py"),
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            stem,
            "--require-output-hash",
            "--require-tokens",
            "--min-repeats",
            "2",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-output-determinism-ci-") as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "out"
        passing = tmp_path / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps({"benchmarks": [row(), row()]}) + "\n", encoding="utf-8")
        completed = run_audit(passing, output_dir, "qemu_output_determinism_audit_latest")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        payload = json.loads((output_dir / "qemu_output_determinism_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_output_determinism_status"):
            return rc
        if rc := require(payload["summary"]["groups"] == 1, "unexpected_output_determinism_groups"):
            return rc
        rows = list(csv.DictReader((output_dir / "qemu_output_determinism_audit_latest.csv").open(encoding="utf-8")))
        if rc := require(len(rows) == 2 and rows[0]["output_sha256"], "missing_output_determinism_rows"):
            return rc
        findings = list(csv.DictReader((output_dir / "qemu_output_determinism_audit_latest_findings.csv").open(encoding="utf-8")))
        if rc := require(findings == [], "unexpected_output_determinism_findings"):
            return rc
        junit = ET.parse(output_dir / "qemu_output_determinism_audit_latest_junit.xml").getroot()
        if rc := require(junit.attrib["failures"] == "0", "unexpected_output_determinism_junit_failure"):
            return rc

        failing = tmp_path / "qemu_prompt_bench_fail.json"
        failing.write_text(json.dumps({"benchmarks": [row(), row(output="different", tokens=9)]}) + "\n", encoding="utf-8")
        failed = run_audit(failing, output_dir, "bad")
        if rc := require(failed.returncode == 1, "expected_output_determinism_failure"):
            return rc
        bad_payload = json.loads((output_dir / "bad.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in bad_payload["findings"]}
        if rc := require("output_hash_drift" in kinds, "missing_output_hash_drift"):
            return rc
        if rc := require("token_count_drift" in kinds, "missing_token_count_drift"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
