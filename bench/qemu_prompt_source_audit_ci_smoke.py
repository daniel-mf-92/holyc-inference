#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_prompt_source_audit.py."""

from __future__ import annotations

import hashlib
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


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def write_prompt_suite(path: Path) -> None:
    rows = [
        {"prompt_id": "short", "prompt": "Summarize integer-only inference.", "expected_tokens": 8},
        {"prompt_id": "code", "prompt": "Write a tiny loop.", "expected_tokens": 12},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def artifact(path: Path, prompt_source: Path, *, bad_hash: bool = False, unknown_prompt: bool = False) -> None:
    prompt = "Summarize integer-only inference."
    prompt_id = "missing" if unknown_prompt else "short"
    path.write_text(
        json.dumps(
            {
                "prompt_suite": {"source": str(prompt_source)},
                "benchmarks": [
                    {
                        "prompt": prompt_id,
                        "phase": "measured",
                        "prompt_sha256": "bad" if bad_hash else prompt_sha(prompt),
                        "prompt_bytes": len(prompt.encode("utf-8")),
                        "expected_tokens": 8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_source_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_prompt_source_audit_latest",
            "--require-expected-tokens",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-prompt-source-") as tmp:
        root = Path(tmp)
        prompts = root / "prompts.jsonl"
        write_prompt_suite(prompts)

        passing = root / "qemu_prompt_bench_pass.json"
        artifact(passing, prompts)
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_prompt_source_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_prompt_source_pass_status"):
            return rc
        if rc := require(report["summary"]["checks"] == 3, "unexpected_prompt_source_check_count"):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_prompt_source_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_prompt_source_junit_failure"):
            return rc

        bad = root / "qemu_prompt_bench_bad_hash.json"
        artifact(bad, prompts, bad_hash=True)
        bad_dir = root / "bad"
        failed = run_audit(bad, bad_dir)
        if rc := require(failed.returncode == 1, "bad_prompt_hash_not_rejected"):
            return rc
        failed_report = json.loads((bad_dir / "qemu_prompt_source_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require("prompt_sha256_mismatch" in {finding["kind"] for finding in failed_report["findings"]}, "bad_prompt_hash_not_reported"):
            return rc

        unknown = root / "qemu_prompt_bench_unknown.json"
        artifact(unknown, prompts, unknown_prompt=True)
        unknown_dir = root / "unknown"
        unknown_run = run_audit(unknown, unknown_dir)
        if rc := require(unknown_run.returncode == 1, "unknown_prompt_not_rejected"):
            return rc
        unknown_report = json.loads((unknown_dir / "qemu_prompt_source_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require("unknown_prompt_id" in {finding["kind"] for finding in unknown_report["findings"]}, "unknown_prompt_not_reported"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
