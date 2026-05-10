#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU prompt schema audit artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "bench" / "prompts" / "smoke.jsonl"
SYNTHETIC_QEMU = ROOT / "bench" / "fixtures" / "qemu_synthetic_bench.py"
SYNTHETIC_IMAGE = Path("/tmp/TempleOS.synthetic.img")


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-schema-ci-") as tmp:
        tmp_path = Path(tmp)
        bench_dir = tmp_path / "bench"
        audit_dir = tmp_path / "schema"

        bench_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_bench.py"),
            "--image",
            str(SYNTHETIC_IMAGE),
            "--prompts",
            str(PROMPTS),
            "--qemu-bin",
            str(SYNTHETIC_QEMU),
            "--warmup",
            "1",
            "--repeat",
            "1",
            "--timeout",
            "5",
            "--output-dir",
            str(bench_dir),
            "--profile",
            "ci-airgap-smoke",
            "--model",
            "synthetic-smoke",
            "--quantization",
            "Q8_0",
            "--require-tokens",
            "--require-tok-per-s",
            "--require-expected-tokens",
            "--require-expected-tokens-match",
            "--require-guest-prompt-sha256-match",
            "--require-guest-prompt-bytes-match",
            "--max-launches",
            "4",
            "--min-prompt-count",
            "2",
            "--qemu-arg=-m",
            "--qemu-arg=256M",
        ]
        completed = run(bench_command, cwd=ROOT)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        audit_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_schema_audit.py"),
            str(bench_dir / "qemu_prompt_bench_latest.json"),
            "--output-dir",
            str(audit_dir),
            "--output-stem",
            "schema",
            "--min-artifacts",
            "1",
            "--min-measured-rows",
            "2",
            "--require-success",
            "--require-ok-telemetry",
            "ttft_us",
            "--require-ok-telemetry",
            "memory_bytes",
        ]
        completed = run(audit_command, cwd=ROOT)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((audit_dir / "schema.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_schema_status"):
            return rc
        if rc := require(report["summary"]["artifacts"] == 1, "unexpected_artifact_count"):
            return rc
        if rc := require(report["summary"]["measured_rows"] == 2, "unexpected_measured_row_count"):
            return rc
        if rc := require(report["summary"]["warmup_rows"] == 2, "unexpected_warmup_row_count"):
            return rc
        if rc := require(report["artifacts"][0]["command_airgap_ok"] is True, "airgap_not_checked"):
            return rc
        if rc := require(
            "QEMU Prompt Schema Audit" in (audit_dir / "schema.md").read_text(encoding="utf-8"),
            "missing_markdown_title",
        ):
            return rc
        artifact_rows = list(csv.DictReader((audit_dir / "schema.csv").open(encoding="utf-8", newline="")))
        if rc := require(artifact_rows[0]["status"] == "pass", "unexpected_csv_status"):
            return rc
        finding_rows = list(csv.DictReader((audit_dir / "schema_findings.csv").open(encoding="utf-8", newline="")))
        if rc := require(finding_rows == [], "unexpected_findings_rows"):
            return rc
        junit_root = ET.parse(audit_dir / "schema_junit.xml").getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_qemu_prompt_schema_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        bad_path = tmp_path / "bad.json"
        bad_payload = json.loads((bench_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
        bad_payload["benchmarks"][0]["command"] = ["qemu-system-x86_64", "-net", "none"]
        bad_payload["benchmarks"][0]["command_has_legacy_net_none"] = True
        bad_payload["benchmarks"][0]["command_has_explicit_nic_none"] = False
        bad_path.write_text(json.dumps(bad_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fail_dir = tmp_path / "fail"
        completed = run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_prompt_schema_audit.py"),
                str(bad_path),
                "--output-dir",
                str(fail_dir),
                "--output-stem",
                "schema",
            ],
            cwd=ROOT,
        )
        if completed.returncode == 0:
            print("schema_bad_airgap_gate_did_not_fail=true", file=sys.stderr)
            return 1
        fail_report = json.loads((fail_dir / "schema.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"legacy_net_none", "missing_nic_none", "command_hash"} <= kinds, "missing_bad_airgap_findings"):
            return rc

    print("qemu_prompt_schema_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
