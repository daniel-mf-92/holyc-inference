#!/usr/bin/env python3
"""Smoke gate for HCEval binary budget audit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_pack


def write_fixture(path: Path, manifest: Path) -> None:
    records = [
        dataset_pack.EvalRecord(
            record_id="smoke-1",
            dataset="smoke-eval",
            split="validation",
            prompt="Pick the warm color.",
            choices=["red", "blue", "green", "black"],
            answer_index=0,
            provenance="synthetic smoke",
        ),
        dataset_pack.EvalRecord(
            record_id="smoke-2",
            dataset="smoke-eval",
            split="validation",
            prompt="Pick the cold color.",
            choices=["orange", "blue", "yellow", "red"],
            answer_index=1,
            provenance="synthetic smoke",
        ),
    ]
    dataset_pack.write_outputs(records, path, manifest, "smoke-eval", "validation")


def run_audit(input_path: Path, output_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "hceval_budget_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "hceval_budget_audit_smoke",
            *extra,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hceval-budget-smoke-") as tmp:
        tmp_path = Path(tmp)
        hceval = tmp_path / "smoke.hceval"
        manifest = tmp_path / "smoke.manifest.json"
        write_fixture(hceval, manifest)

        pass_out = tmp_path / "pass"
        passed = run_audit(
            hceval,
            pass_out,
            "--require-manifest",
            "--max-binary-bytes",
            "4096",
            "--max-metadata-bytes",
            "512",
            "--max-record-payload-bytes",
            "512",
        )
        if passed.returncode != 0:
            print(passed.stdout)
            print(passed.stderr, file=sys.stderr)
            return passed.returncode
        pass_report = json.loads((pass_out / "hceval_budget_audit_smoke.json").read_text(encoding="utf-8"))
        pass_junit = ET.parse(pass_out / "hceval_budget_audit_smoke_junit.xml").getroot()
        if pass_report["status"] != "pass" or pass_junit.attrib["failures"] != "0":
            print("expected passing budget audit", file=sys.stderr)
            return 1

        fail_out = tmp_path / "fail"
        failed = run_audit(hceval, fail_out, "--max-binary-bytes", "64")
        if failed.returncode == 0:
            print("expected failing budget audit", file=sys.stderr)
            return 1
        fail_report = json.loads((fail_out / "hceval_budget_audit_smoke.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if "max_binary_bytes" not in kinds:
            print(f"missing max_binary_bytes finding: {kinds}", file=sys.stderr)
            return 1

    print("hceval_budget_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
