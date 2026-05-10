#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_throughput_stability_audit.py."""

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


def artifact(*, unstable: bool = False, missing: bool = False) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    elapsed_values = (32_000, 33_000, 34_000) if not unstable else (32_000, 96_000, 34_000)
    for index, wall_elapsed_us in enumerate(elapsed_values, 1):
        row: dict[str, object] = {
            "prompt": "smoke-short",
            "profile": "ci-airgap-smoke",
            "model": "synthetic-smoke",
            "quantization": "Q4_0",
            "commit": "abc123",
            "phase": "measured",
            "exit_class": "ok",
            "tokens": 32,
            "elapsed_us": wall_elapsed_us - 2_000,
            "wall_elapsed_us": wall_elapsed_us,
            "tok_per_s": 32_000_000.0 / (wall_elapsed_us - 2_000),
            "wall_tok_per_s": 32_000_000.0 / wall_elapsed_us,
        }
        if missing and index == 1:
            row.pop("wall_tok_per_s")
        rows.append(row)
    return {"status": "pass", "benchmarks": rows}


def run_audit(input_path: Path, output_dir: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_throughput_stability_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_throughput_stability_audit_latest",
            "--min-rows",
            "3",
            "--min-samples-per-group",
            "3",
            *(extra or []),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-throughput-audit-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps(artifact()), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir, ["--min-wall-tok-per-s", "900", "--max-wall-tok-per-s-cv", "0.05"])
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_throughput_stability_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_throughput_audit_pass_status"):
            return rc
        if rc := require(report["summary"]["groups"] == 1, "missing_throughput_group_rollup"):
            return rc
        if rc := require((pass_dir / "qemu_throughput_stability_audit_latest.md").exists(), "missing_throughput_markdown"):
            return rc
        if rc := require((pass_dir / "qemu_throughput_stability_audit_latest_samples.csv").exists(), "missing_throughput_samples_csv"):
            return rc
        if rc := require((pass_dir / "qemu_throughput_stability_audit_latest_junit.xml").exists(), "missing_throughput_junit"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(json.dumps(artifact(unstable=True, missing=True)), encoding="utf-8")
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir, ["--min-wall-tok-per-s", "900", "--max-wall-tok-per-s-cv", "0.05"])
        if rc := require(failed.returncode == 1, "throughput_audit_unstable_or_missing_metric_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_throughput_stability_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"missing_metric", "min_wall_tok_per_s", "max_wall_tok_per_s_cv"} <= kinds, "throughput_audit_findings_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
