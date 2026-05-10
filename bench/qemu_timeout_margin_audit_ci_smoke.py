#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_timeout_margin_audit.py."""

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


def row(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "iteration": 1,
        "exit_class": "ok",
        "timed_out": False,
        "timeout_seconds": 1.0,
        "wall_elapsed_us": 250_000,
        "wall_timeout_pct": 25.0,
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_timeout_margin_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_timeout_margin_audit_latest",
            "--min-rows",
            "1",
            "--max-ok-timeout-pct",
            "80",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-timeout-margin-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps({"benchmarks": [row()]}), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((pass_dir / "qemu_timeout_margin_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_timeout_margin_pass_status"):
            return rc
        if rc := require(report["summary"]["max_wall_timeout_pct"] == 25.0, "missing_timeout_margin_summary"):
            return rc
        if rc := require(
            "No timeout margin findings." in (pass_dir / "qemu_timeout_margin_audit_latest.md").read_text(encoding="utf-8"),
            "missing_timeout_margin_markdown",
        ):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_timeout_margin_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_timeout_margin_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row(prompt="too-close", wall_elapsed_us=950_000, wall_timeout_pct=95.0),
                        row(prompt="drift", wall_timeout_pct=1.0),
                        row(prompt="missing", timeout_seconds=None),
                        row(prompt="early-timeout", exit_class="timeout", timed_out=True, wall_elapsed_us=500_000, wall_timeout_pct=50.0),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "timeout_margin_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_timeout_margin_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require(
            {"ok_timeout_margin_too_small", "wall_timeout_pct_drift", "missing_timeout_seconds", "timeout_budget_underused"} <= kinds,
            "timeout_margin_findings_not_reported",
        ):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
