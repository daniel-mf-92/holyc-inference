#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_seed_audit.py."""

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


def row(prompt: str = "smoke-short", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "benchmark": "qemu_prompt",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "prompt": prompt,
        "iteration": 1,
        "commit": "abc1234",
        "seed": 42,
        "returncode": 0,
        "timed_out": False,
        "exit_class": "ok",
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"status": "pass", "benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_seed_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_seed_audit_latest",
            "--min-rows",
            "2",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-seed-audit-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        write_artifact(passing, [row("smoke-short"), row("smoke-code", iteration=2, seed="", rng_seed=99)])
        completed = run_audit(passing, ROOT / "bench" / "results")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((ROOT / "bench" / "results" / "qemu_seed_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_qemu_seed_audit_pass_status"):
            return rc
        if rc := require(report["summary"]["seeded_rows"] == 2, "missing_qemu_seed_rollup"):
            return rc
        if rc := require((ROOT / "bench" / "results" / "qemu_seed_audit_latest.md").exists(), "missing_qemu_seed_markdown"):
            return rc
        if rc := require((ROOT / "bench" / "results" / "qemu_seed_audit_latest.csv").exists(), "missing_qemu_seed_csv"):
            return rc
        if rc := require((ROOT / "bench" / "results" / "qemu_seed_audit_latest_findings.csv").exists(), "missing_qemu_seed_findings_csv"):
            return rc
        if rc := require((ROOT / "bench" / "results" / "qemu_seed_audit_latest_junit.xml").exists(), "missing_qemu_seed_junit"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        write_artifact(
            failing,
            [
                row("same", seed=7),
                row("same", seed=8),
                row("missing", seed=""),
                row("bad", seed="not-int"),
                row("negative", seed=-1),
            ],
        )
        failed = run_audit(failing, root / "fail")
        if rc := require(failed.returncode == 1, "qemu_seed_drift_not_rejected"):
            return rc
        fail_report = json.loads((root / "fail" / "qemu_seed_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {"seed_drift", "missing_seed", "invalid_seed", "negative_seed"}
        if rc := require(expected <= kinds, "qemu_seed_findings_not_reported"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
