#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_status_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def row(prompt: str = "smoke-short", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "phase": "measured",
        "returncode": 0,
        "timed_out": False,
        "exit_class": "ok",
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, *, status: str, rows: list[dict[str, object]], **overrides: object) -> None:
    ok_rows = sum(1 for item in rows if item.get("exit_class") == "ok" and not item.get("timed_out") and item.get("returncode") == 0)
    payload: dict[str, object] = {
        "generated_at": "2026-05-01T00:00:00Z",
        "status": status,
        "warmups": [],
        "benchmarks": rows,
        "command_airgap": {"ok": True, "violations": []},
        "telemetry_findings": [],
        "variability_findings": [],
        "suite_summary": {
            "runs": len(rows),
            "ok_runs": ok_rows,
            "failed_runs": len(rows) - ok_rows,
            "timed_out_runs": sum(1 for item in rows if item.get("timed_out") is True),
            "nonzero_exit_runs": sum(1 for item in rows if item.get("returncode") not in (None, 0)),
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-status-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing = tmp_path / "passing"
        passing.mkdir()
        write_artifact(passing / "qemu_prompt_bench_pass.json", status="pass", rows=[row("a"), row("b")])
        output_dir = tmp_path / "out"
        completed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_status_audit.py"),
                str(passing),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_status_audit_smoke",
            ]
        )
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads((output_dir / "qemu_status_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_status_audit_pass_status"):
            return rc
        if rc := require(report["summary"]["rows"] == 2, "unexpected_status_audit_row_count"):
            return rc
        if rc := require(
            "No status consistency findings." in (output_dir / "qemu_status_audit_smoke.md").read_text(encoding="utf-8"),
            "missing_status_audit_markdown",
        ):
            return rc
        if rc := require("expected_status" in (output_dir / "qemu_status_audit_smoke.csv").read_text(encoding="utf-8"), "missing_status_csv"):
            return rc
        junit = ET.parse(output_dir / "qemu_status_audit_smoke_junit.xml").getroot()
        if rc := require(junit.attrib["name"] == "holyc_qemu_status_audit", "missing_status_junit"):
            return rc

        failing = tmp_path / "failing"
        failing.mkdir()
        write_artifact(
            failing / "qemu_prompt_bench_fail.json",
            status="pass",
            rows=[row("timeout", returncode=124, timed_out=True, exit_class="timeout")],
            command_airgap={"ok": False, "violations": ["missing -nic none"]},
            telemetry_findings=[{"metric": "wall_tok_per_s"}],
            suite_summary={"runs": 1, "ok_runs": 1, "failed_runs": 0, "timed_out_runs": 0, "nonzero_exit_runs": 0},
        )
        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_status_audit.py"),
                str(failing),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_status_audit_failing",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "status_audit_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((output_dir / "qemu_status_audit_failing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {
            "pass_with_failed_rows",
            "pass_with_telemetry_findings",
            "pass_without_airgap_ok",
            "status_mismatch",
            "suite_ok_runs_mismatch",
            "suite_failed_runs_mismatch",
            "suite_timed_out_runs_mismatch",
            "suite_nonzero_exit_runs_mismatch",
        }
        if rc := require(expected <= kinds, "status_audit_findings_not_reported"):
            return rc

    print("qemu_status_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
