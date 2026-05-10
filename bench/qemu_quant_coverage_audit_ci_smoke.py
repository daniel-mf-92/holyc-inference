#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_quant_coverage_audit.py."""

from __future__ import annotations

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


def row(prompt: str, quantization: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": quantization,
        "phase": "measured",
        "exit_class": "ok",
        "returncode": 0,
        "timed_out": False,
        "tok_per_s": 128.0,
        "command_airgap_ok": True,
        "command_has_explicit_nic_none": True,
    }
    payload.update(overrides)
    return payload


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_quant_coverage_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_quant_coverage_audit_latest",
            "--min-rows-per-quant",
            "2",
            "--min-ok-rows-per-quant",
            "2",
            "--min-prompts-per-quant",
            "2",
            "--require-airgap-command",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-quant-coverage-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row("smoke-short", "Q4_0"),
                        row("smoke-long", "Q4_0"),
                        row("smoke-short", "Q8_0"),
                        row("smoke-long", "Q8_0"),
                    ]
                }
            ),
            encoding="utf-8",
        )
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((pass_dir / "qemu_quant_coverage_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_quant_coverage_pass_status"):
            return rc
        if rc := require(report["summary"]["groups"] == 2, "missing_quant_coverage_groups"):
            return rc
        if rc := require(report["summary"]["airgap_ok_rows"] == 4, "missing_quant_airgap_rows"):
            return rc
        if rc := require("No quantization coverage findings." in (pass_dir / "qemu_quant_coverage_audit_latest.md").read_text(encoding="utf-8"), "missing_quant_coverage_markdown"):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_quant_coverage_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_quant_coverage_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row("smoke-short", "Q4_0"),
                        row("smoke-long", "Q4_0", exit_class="timeout", returncode=124, timed_out=True),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "quant_coverage_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_quant_coverage_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"min_ok_rows_per_quant", "missing_required_quantization"} <= kinds, "quant_coverage_findings_not_reported"):
            return rc

        airgap_failing = root / "qemu_prompt_bench_airgap_fail.json"
        airgap_failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row("smoke-short", "Q4_0", command_airgap_ok=False),
                        row("smoke-long", "Q4_0"),
                        row("smoke-short", "Q8_0"),
                        row("smoke-long", "Q8_0", command_has_explicit_nic_none=False),
                    ]
                }
            ),
            encoding="utf-8",
        )
        airgap_fail_dir = root / "airgap-fail"
        airgap_failed = run_audit(airgap_failing, airgap_fail_dir)
        if rc := require(airgap_failed.returncode == 1, "quant_coverage_airgap_bad_artifact_not_rejected"):
            return rc
        airgap_fail_report = json.loads((airgap_fail_dir / "qemu_quant_coverage_audit_latest.json").read_text(encoding="utf-8"))
        airgap_kinds = {finding["kind"] for finding in airgap_fail_report["findings"]}
        if rc := require({"missing_airgap_command", "missing_explicit_nic_none"} <= airgap_kinds, "quant_coverage_airgap_findings_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
