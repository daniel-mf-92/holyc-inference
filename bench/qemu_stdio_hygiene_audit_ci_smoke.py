#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_stdio_hygiene_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def row(prompt: str, exit_class: str = "ok", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "exit_class": exit_class,
        "timed_out": exit_class == "timeout",
        "stdout_bytes": 96,
        "stderr_bytes": 0,
        "stdout_tail": "BENCH_RESULT: {}\n" if exit_class == "ok" else "timeout\n",
        "stderr_tail": "",
        "failure_reason": "" if exit_class == "ok" else exit_class,
    }
    payload.update(overrides)
    return payload


def run_audit(input_path: Path, output_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_stdio_hygiene_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_stdio_hygiene_audit_latest",
            "--min-rows",
            "2",
            *extra,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-stdio-hygiene-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(
            json.dumps({"benchmarks": [row("smoke-short"), row("smoke-timeout", "timeout", stdout_bytes=8)]}),
            encoding="utf-8",
        )
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_stdio_hygiene_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_stdio_hygiene_pass_status"):
            return rc
        if rc := require(report["summary"]["rows"] == 2, "missing_stdio_hygiene_row_count"):
            return rc
        for suffix in (".md", ".csv", "_rows.csv", "_findings.csv", "_junit.xml"):
            path = pass_dir / f"qemu_stdio_hygiene_audit_latest{suffix}"
            if rc := require(path.exists(), f"missing_stdio_hygiene_sidecar_{suffix}"):
                return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row("noisy-ok", stderr_bytes=4, stderr_tail="warn"),
                        row("silent-timeout", "timeout", stdout_bytes=0, stdout_tail="", failure_reason=""),
                        row("tail-drift", stdout_bytes=1, stdout_tail="long tail"),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "stdio_hygiene_bad_rows_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_stdio_hygiene_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {"ok_stderr_noise", "silent_failure", "tail_exceeds_counter"}
        if rc := require(expected <= kinds, "stdio_hygiene_findings_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
