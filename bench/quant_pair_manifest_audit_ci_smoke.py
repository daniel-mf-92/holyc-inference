#!/usr/bin/env python3
"""Stdlib-only smoke gate for quant_pair_manifest_audit.py."""

from __future__ import annotations

import json
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
            str(ROOT / "bench" / "quant_pair_manifest_audit.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "quant_pair_manifest_audit_smoke_latest",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def junit_has_no_failures(path: Path) -> bool:
    return 'failures="0"' in path.read_text(encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-quant-pair-") as tmp:
        tmp_path = Path(tmp)
        passing = tmp_path / "pass.json"
        write_manifest(
            passing,
            [
                {"name": "tok_embeddings", "format": "q4_0", "elements": 64, "blocks": 2, "bytes": 36},
                {"name": "tok_embeddings", "format": "q8_0", "elements": 64, "blocks": 2, "bytes": 68},
                {"name": "output", "format": "q4_0", "elements": 32, "blocks": 1, "bytes": 18},
                {"name": "output", "format": "q8_0", "elements": 32, "blocks": 1, "bytes": 34},
            ],
        )
        pass_out = tmp_path / "pass_out"
        completed = run_audit(passing, pass_out)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_out / "quant_pair_manifest_audit_smoke_latest.json").read_text(encoding="utf-8"))
        if (
            report["status"] != "pass"
            or report["summary"]["complete_pairs"] != 2
            or not junit_has_no_failures(pass_out / "quant_pair_manifest_audit_smoke_latest_junit.xml")
        ):
            print("passing_quant_pair_manifest_audit_failed=true", file=sys.stderr)
            return 1

        failing = tmp_path / "fail.json"
        write_manifest(
            failing,
            [
                {"name": "tok_embeddings", "format": "q4_0", "elements": 64, "blocks": 2, "bytes": 36},
                {"name": "output", "format": "q4_0", "elements": 32, "blocks": 1, "bytes": 18},
                {"name": "output", "format": "q8_0", "elements": 64, "blocks": 2, "bytes": 68},
            ],
        )
        fail_out = tmp_path / "fail_out"
        completed = run_audit(failing, fail_out)
        if completed.returncode == 0:
            print("failing_quant_pair_manifest_audit_not_rejected=true", file=sys.stderr)
            return 1
        report = json.loads((fail_out / "quant_pair_manifest_audit_smoke_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        if report["status"] != "fail" or "missing_quant_pair" not in kinds or "element_count_mismatch" not in kinds:
            print("failing_quant_pair_manifest_audit_missing_findings=true", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
