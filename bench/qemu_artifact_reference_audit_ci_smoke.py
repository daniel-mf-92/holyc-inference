#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU artifact reference audit."""

from __future__ import annotations

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
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-ref-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        bench_dir = tmp_path / "bench"
        audit_dir = tmp_path / "audit"

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
            "Q4_0",
            "--max-launches",
            "4",
            "--min-prompt-count",
            "1",
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
            str(ROOT / "bench" / "qemu_artifact_reference_audit.py"),
            str(bench_dir / "qemu_prompt_bench_latest.json"),
            "--output-dir",
            str(audit_dir),
            "--output-stem",
            "refs",
            "--min-artifacts",
            "1",
        ]
        completed = run(audit_command, cwd=ROOT)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((audit_dir / "refs.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_reference_audit_status"):
            return rc
        if rc := require(report["summary"]["artifacts"] == 1, "unexpected_artifact_count"):
            return rc
        if rc := require(report["summary"]["command_arrays_checked"] >= 1, "missing_command_array_coverage"):
            return rc
        junit_root = ET.parse(audit_dir / "refs_junit.xml").getroot()
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        bad_path = tmp_path / "bad.json"
        bad_payload = json.loads((bench_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
        bad_payload["prompt_suite"]["source"] = "https://example.invalid/prompts.jsonl"
        bad_payload["benchmarks"][0]["command"] = ["qemu-system-x86_64", "-netdev", "user,id=n0"]
        bad_path.write_text(json.dumps(bad_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fail_dir = tmp_path / "fail"
        completed = run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_artifact_reference_audit.py"),
                str(bad_path),
                "--output-dir",
                str(fail_dir),
                "--output-stem",
                "refs",
            ],
            cwd=ROOT,
        )
        if completed.returncode == 0:
            print("reference_bad_airgap_gate_did_not_fail=true", file=sys.stderr)
            return 1
        fail_report = json.loads((fail_dir / "refs.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"remote_uri", "missing_nic_none", "command_airgap_violation"} <= kinds, "missing_failure_findings"):
            return rc

    print("qemu_artifact_reference_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
