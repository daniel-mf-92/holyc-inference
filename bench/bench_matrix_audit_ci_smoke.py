#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for benchmark matrix audits."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "bench" / "fixtures" / "bench_matrix_smoke.json"


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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-bench-matrix-audit-ci-") as tmp:
        output_dir = Path(tmp)
        stem = "bench_matrix_audit_smoke"
        completed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "bench_matrix_audit.py"),
                str(MATRIX),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                stem,
                "--expect-cells",
                "2",
                "--min-profiles",
                "1",
                "--min-models",
                "1",
                "--min-quantizations",
                "2",
                "--require-quantization",
                "Q4_0",
                "--require-quantization",
                "Q8_0",
            ]
        )
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads((output_dir / f"{stem}.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_audit_status"):
            return rc
        if rc := require(report["matrix_count"] == 1 and report["cell_count"] == 2, "unexpected_cell_count"):
            return rc
        matrix_report = report["matrices"][0]
        if rc := require(matrix_report["axis_counts"]["quantizations"] == 2, "unexpected_quantization_count"):
            return rc
        if rc := require(matrix_report["prompt_suite"]["prompt_count"] == 2, "unexpected_prompt_count"):
            return rc
        if rc := require(all(cell["command_has_explicit_nic_none"] for cell in matrix_report["cells"]), "missing_nic_none"):
            return rc
        if rc := require(all(cell["command_airgap_ok"] for cell in matrix_report["cells"]), "airgap_failure"):
            return rc

        rows = list(csv.DictReader((output_dir / f"{stem}.csv").open(newline="", encoding="utf-8")))
        if rc := require(len([row for row in rows if row["row_type"] == "cell"]) == 2, "missing_cell_rows"):
            return rc
        if rc := require("Benchmark Matrix Audit" in (output_dir / f"{stem}.md").read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        junit = ET.parse(output_dir / f"{stem}_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_bench_matrix_audit", "missing_junit"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_junit_failure"):
            return rc

        bad_matrix = output_dir / "bad_matrix.json"
        bad_payload = json.loads(MATRIX.read_text(encoding="utf-8"))
        bad_payload["qemu_args"] = ["-net", "none"]
        bad_payload["expect_cells"] = 3
        bad_matrix.write_text(json.dumps(bad_payload, indent=2) + "\n", encoding="utf-8")
        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "bench_matrix_audit.py"),
                str(bad_matrix),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "bench_matrix_audit_bad",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 2, "expected_bad_matrix_failure"):
            return rc
        failed_report = json.loads((output_dir / "bench_matrix_audit_bad.json").read_text(encoding="utf-8"))
        messages = [finding["message"] for matrix in failed_report["matrices"] for finding in matrix["findings"]]
        if rc := require(any("legacy `-net none`" in message for message in messages), "missing_net_none_rejection"):
            return rc
        if rc := require(any("expected 3" in message for message in messages), "missing_expect_cells_failure"):
            return rc

    print("bench_matrix_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
