#!/usr/bin/env python3
"""Stdlib-only smoke gate for quant_pair_size_audit.py."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_manifest(path: Path, artifacts: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"model": "smoke-llm", "artifacts": artifacts}, indent=2) + "\n", encoding="utf-8")


def run_audit(manifest: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "quant_pair_size_audit.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "quant_pair_size_audit_smoke_latest",
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


def junit_failures(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'\bfailures="(\d+)"', text)
    return int(match.group(1)) if match else -1


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-quant-pair-size-") as tmp:
        root = Path(tmp)
        passing = root / "pass.json"
        write_manifest(
            passing,
            [
                {"name": "tok_embeddings", "format": "q4_0", "elements": 64, "blocks": 2, "bytes": 36},
                {"name": "tok_embeddings", "format": "q8_0", "elements": 64, "blocks": 2, "bytes": 68},
                {"name": "output", "format": "q4_0", "elements": 32, "blocks": 1, "bytes": 18},
                {"name": "output", "format": "q8_0", "elements": 32, "blocks": 1, "bytes": 34},
            ],
        )
        pass_out = root / "pass_out"
        completed = run_audit(passing, pass_out)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_out / "quant_pair_size_audit_smoke_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "passing_quant_pair_size_status_failed"):
            return rc
        if rc := require(report["summary"]["complete_pairs"] == 2, "passing_quant_pair_size_pair_count_failed"):
            return rc
        if rc := require(report["summary"]["q4_0_bytes_total"] == 54, "passing_quant_pair_size_q4_total_failed"):
            return rc
        if rc := require(report["summary"]["q8_0_bytes_total"] == 102, "passing_quant_pair_size_q8_total_failed"):
            return rc
        if rc := require(junit_failures(pass_out / "quant_pair_size_audit_smoke_latest_junit.xml") == 0, "passing_quant_pair_size_junit_failed"):
            return rc

        failing = root / "fail.json"
        write_manifest(
            failing,
            [
                {"name": "tok_embeddings", "format": "q4_0", "elements": 64, "blocks": 2, "bytes": 35},
                {"name": "tok_embeddings", "format": "q8_0", "elements": 64, "blocks": 3, "bytes": 102},
                {"name": "output", "format": "q4_0", "elements": 33, "blocks": 1, "bytes": 18},
                {"name": "output", "format": "q8_0", "elements": 32, "blocks": 1, "bytes": 34},
            ],
        )
        fail_out = root / "fail_out"
        completed = run_audit(failing, fail_out)
        if rc := require(completed.returncode == 1, "failing_quant_pair_size_not_rejected"):
            return rc
        report = json.loads((fail_out / "quant_pair_size_audit_smoke_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        expected = {
            "byte_count_mismatch",
            "block_count_mismatch",
            "element_capacity_mismatch",
            "pair_block_count_mismatch",
            "pair_element_count_mismatch",
            "pair_q4_byte_count_mismatch",
        }
        if rc := require(report["status"] == "fail" and expected <= kinds, "failing_quant_pair_size_missing_findings"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
