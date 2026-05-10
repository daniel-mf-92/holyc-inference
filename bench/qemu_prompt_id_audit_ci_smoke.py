#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_prompt_id_audit.py."""

from __future__ import annotations

import hashlib
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


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def row(prompt: str = "smoke-short", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "benchmark": "qemu_prompt",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "prompt": prompt,
        "prompt_sha256": prompt_sha(prompt),
        "launch_index": 1,
        "iteration": 1,
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
            str(ROOT / "bench" / "qemu_prompt_id_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_prompt_id_audit_latest",
            "--min-rows",
            "2",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-prompt-id-audit-") as tmp:
        root = Path(tmp)
        passing = ROOT / "bench" / "fixtures" / "qemu_prompt_id_audit" / "pass.json"
        completed = run_audit(passing, ROOT / "bench" / "results")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((ROOT / "bench" / "results" / "qemu_prompt_id_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_prompt_id_audit_pass_status"):
            return rc
        if rc := require(report["summary"]["unique_prompts"] == 2, "missing_prompt_identity_rollup"):
            return rc
        if rc := require((ROOT / "bench" / "results" / "qemu_prompt_id_audit_latest.md").exists(), "missing_prompt_id_markdown"):
            return rc
        if rc := require((ROOT / "bench" / "results" / "qemu_prompt_id_audit_latest.csv").exists(), "missing_prompt_id_csv"):
            return rc
        if rc := require((ROOT / "bench" / "results" / "qemu_prompt_id_audit_latest_findings.csv").exists(), "missing_prompt_id_findings_csv"):
            return rc
        if rc := require((ROOT / "bench" / "results" / "qemu_prompt_id_audit_latest_junit.xml").exists(), "missing_prompt_id_junit"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        write_artifact(
            failing,
            [
                row("same-name"),
                row("same-name", prompt_sha256=prompt_sha("different-text"), launch_index=2),
                row("alias", prompt_sha256=prompt_sha("same-name"), launch_index=3),
                row("", prompt_sha256="not-a-sha", launch_index=4),
            ],
        )
        failed = run_audit(failing, root / "fail")
        if rc := require(failed.returncode == 1, "prompt_id_drift_not_rejected"):
            return rc
        fail_report = json.loads((root / "fail" / "qemu_prompt_id_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {"prompt_hash_drift", "prompt_hash_collision", "missing_prompt", "invalid_prompt_sha256"}
        if rc := require(expected <= kinds, "prompt_id_findings_not_reported"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
