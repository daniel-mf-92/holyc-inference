#!/usr/bin/env python3
"""Stdlib-only smoke gate for qemu_quant_pairing_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"generated_at": "2026-05-02T00:00:00Z", "benchmarks": rows}) + "\n", encoding="utf-8")


def row(quantization: str, prompt: str = "smoke-short", iteration: int = 1, exit_class: str = "ok") -> dict[str, object]:
    return {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "prompt": prompt,
        "phase": "measured",
        "iteration": iteration,
        "commit": "abc123",
        "quantization": quantization,
        "exit_class": exit_class,
        "timed_out": False,
        "failure_reason": None,
    }


def run_audit(input_path: Path, output_dir: Path, stem: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_quant_pairing_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            stem,
            "--min-pairs",
            "1",
            *extra,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-quant-pairing-audit-") as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "out"
        passing = tmp_path / "passing.json"
        write_artifact(passing, [row("Q4_0"), row("Q8_0")])
        completed = run_audit(passing, output_dir, "pass", "--require-success")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((output_dir / "pass.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "pass_report_failed"):
            return rc
        if rc := require(report["summary"]["complete_pairs"] == 1, "complete_pair_count_mismatch"):
            return rc
        junit = ET.parse(output_dir / "pass_junit.xml").getroot()
        if rc := require(junit.attrib["failures"] == "0", "unexpected_junit_failure"):
            return rc

        failing = tmp_path / "failing.json"
        write_artifact(failing, [row("Q4_0"), row("Q8_0", prompt="other"), row("Q8_0", prompt="failed", exit_class="timeout")])
        failed = run_audit(failing, output_dir, "fail", "--require-success")
        if rc := require(failed.returncode == 1, "failing_report_passed"):
            return rc
        failed_report = json.loads((output_dir / "fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require("missing_quant_pair" in kinds, "missing_quant_pair_finding"):
            return rc
        if rc := require("incomplete_success_pair" in kinds, "missing_success_pair_finding"):
            return rc

    print("qemu_quant_pairing_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
