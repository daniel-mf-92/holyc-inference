#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_ttft_audit.py."""

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
        "tokens": 32,
        "ttft_us": 12_000,
        "elapsed_us": 200_000,
        "wall_elapsed_us": 240_000,
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_ttft_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_ttft_audit_latest",
            "--min-rows",
            "1",
            "--max-ttft-us",
            "50000",
            "--max-ttft-elapsed-pct",
            "30",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-ttft-audit-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps({"benchmarks": [row()]}), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((pass_dir / "qemu_ttft_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_ttft_pass_status"):
            return rc
        if rc := require(report["summary"]["median_ttft_us"] == 12_000, "missing_ttft_summary"):
            return rc
        if rc := require(
            "No TTFT findings." in (pass_dir / "qemu_ttft_audit_latest.md").read_text(encoding="utf-8"),
            "missing_ttft_markdown",
        ):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_ttft_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_ttft_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row(prompt="missing", ttft_us=""),
                        row(prompt="negative", ttft_us=-1),
                        row(prompt="late", ttft_us=250_000),
                        row(prompt="too-high", ttft_us=80_000),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "ttft_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_ttft_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {"missing_ttft_us", "negative_ttft_us", "ttft_after_guest_elapsed", "ttft_after_wall_elapsed", "max_ttft_us", "max_ttft_elapsed_pct"}
        if rc := require(expected <= kinds, "ttft_findings_not_reported"):
            return rc

    print("qemu_ttft_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
