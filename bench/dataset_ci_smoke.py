#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for host-side eval dataset tooling."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-ci-") as tmp:
        tmp_path = Path(tmp)
        datasets_dir = tmp_path / "datasets"
        curated = datasets_dir / "smoke_curated.jsonl"
        curated_manifest = datasets_dir / "smoke_curated.manifest.json"
        packed = datasets_dir / "smoke_curated.hceval"
        pack_manifest = datasets_dir / "smoke_curated.hceval.manifest.json"
        inspect_json = datasets_dir / "smoke_curated.inspect.json"
        inspect_md = datasets_dir / "smoke_curated.inspect.md"
        inspect_junit = datasets_dir / "smoke_curated.inspect.junit.xml"
        leak_json = datasets_dir / "dataset_leak_audit_smoke_latest.json"
        leak_md = datasets_dir / "dataset_leak_audit_smoke_latest.md"
        leak_csv = datasets_dir / "dataset_leak_audit_smoke_latest.csv"
        leak_junit = datasets_dir / "dataset_leak_audit_smoke_latest_junit.xml"

        curate_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_curate.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(curated),
            "--manifest",
            str(curated_manifest),
            "--source-name",
            "smoke-eval",
            "--source-version",
            "synthetic",
            "--source-license",
            "synthetic-smoke",
            "--max-records",
            "3",
            "--balance-answer-index",
            "--pack-output",
            str(packed),
            "--pack-manifest",
            str(pack_manifest),
            "--pack-dataset",
            "smoke-eval",
            "--pack-split",
            "validation",
        ]
        completed = run_command(curate_command)
        if completed.returncode != 0:
            return completed.returncode

        curated_report = json.loads(curated_manifest.read_text(encoding="utf-8"))
        if rc := require(curated_report["record_count"] == 3, "unexpected_curated_record_count"):
            return rc
        if rc := require(
            curated_report["answer_histogram"] == {"0": 3},
            "unexpected_curated_answer_histogram",
        ):
            return rc
        if rc := require(curated_report["filters"]["balance_answer_index"] is True, "missing_balance_flag"):
            return rc

        inspect_command = [
            sys.executable,
            str(ROOT / "bench" / "hceval_inspect.py"),
            "--input",
            str(packed),
            "--manifest",
            str(pack_manifest),
            "--output",
            str(inspect_json),
            "--markdown",
            str(inspect_md),
            "--junit",
            str(inspect_junit),
            "--max-prompt-bytes",
            "4096",
            "--max-choice-bytes",
            "1024",
            "--max-record-payload-bytes",
            "8192",
        ]
        completed = run_command(inspect_command)
        if completed.returncode != 0:
            return completed.returncode

        inspect_report = json.loads(inspect_json.read_text(encoding="utf-8"))
        if rc := require(inspect_report["status"] == "pass", "unexpected_inspect_status"):
            return rc
        if rc := require(inspect_report["record_count"] == 3, "unexpected_inspect_record_count"):
            return rc
        if rc := require("HCEval Dataset Inspection" in inspect_md.read_text(encoding="utf-8"), "missing_inspect_md"):
            return rc
        inspect_root = ET.parse(inspect_junit).getroot()
        if rc := require(inspect_root.attrib.get("name") == "holyc_hceval_inspect", "missing_inspect_junit"):
            return rc
        if rc := require(inspect_root.attrib.get("failures") == "0", "unexpected_inspect_junit_failures"):
            return rc

        leak_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_leak_audit.py"),
            "--input",
            str(curated),
            "--output",
            str(leak_json),
            "--markdown",
            str(leak_md),
            "--csv",
            str(leak_csv),
            "--junit",
            str(leak_junit),
            "--fail-on-leaks",
        ]
        completed = run_command(leak_command)
        if completed.returncode != 0:
            return completed.returncode

        leak_report = json.loads(leak_json.read_text(encoding="utf-8"))
        if rc := require(leak_report["status"] == "pass", "unexpected_leak_status"):
            return rc
        if rc := require(leak_report["findings"] == [], "unexpected_leak_findings"):
            return rc
        if rc := require(
            "severity,kind,dataset,splits,key_sha256,record_ids,sources,detail"
            in leak_csv.read_text(encoding="utf-8"),
            "missing_leak_csv_header",
        ):
            return rc

        index_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_index.py"),
            "--input",
            str(datasets_dir),
            "--output-dir",
            str(datasets_dir),
            "--fail-on-findings",
        ]
        completed = run_command(index_command)
        if completed.returncode != 0:
            return completed.returncode

        index_report = json.loads((datasets_dir / "dataset_index_latest.json").read_text(encoding="utf-8"))
        if rc := require(index_report["status"] == "pass", "unexpected_index_status"):
            return rc
        if rc := require(len(index_report["artifacts"]) == 3, "unexpected_index_artifact_count"):
            return rc
        if rc := require(
            "Eval Dataset Artifact Index" in (datasets_dir / "dataset_index_latest.md").read_text(encoding="utf-8"),
            "missing_index_markdown",
        ):
            return rc
        index_root = ET.parse(datasets_dir / "dataset_index_junit_latest.xml").getroot()
        if rc := require(index_root.attrib.get("name") == "holyc_dataset_index", "missing_index_junit"):
            return rc
        if rc := require(index_root.attrib.get("failures") == "0", "unexpected_index_junit_failures"):
            return rc

        leak_fixture = tmp_path / "leaky.jsonl"
        leak_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "leak-train",
                            "dataset": "leak-smoke",
                            "split": "train",
                            "prompt": "Shared prompt?",
                            "choices": ["yes", "no"],
                            "answer_index": 0,
                            "provenance": "synthetic leakage smoke row",
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "id": "leak-validation",
                            "dataset": "leak-smoke",
                            "split": "validation",
                            "prompt": " Shared   prompt? ",
                            "choices": ["yes", "no"],
                            "answer_index": 0,
                            "provenance": "synthetic leakage smoke row",
                        },
                        sort_keys=True,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        unsafe_leak_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_leak_audit.py"),
            "--input",
            str(leak_fixture),
            "--output",
            str(tmp_path / "leaky_audit.json"),
            "--fail-on-leaks",
        ]
        completed = subprocess.run(
            unsafe_leak_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("leaky_dataset_not_rejected=true", file=sys.stderr)
            return 1
        leak_failure = json.loads((tmp_path / "leaky_audit.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in leak_failure["findings"]}
        if rc := require({"prompt_split_leak", "payload_split_leak"}.issubset(kinds), "missing_leak_findings"):
            return rc

    print("dataset_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
