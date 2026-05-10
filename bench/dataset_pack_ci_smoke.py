#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_pack.py and HCEval inspection."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
SAMPLE_REL = Path("bench/datasets/samples/smoke_eval.jsonl")
RESULTS_REL = Path("bench/results/datasets")
RESULTS = ROOT / RESULTS_REL


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    packed_rel = RESULTS_REL / "smoke_eval.hceval"
    manifest_rel = RESULTS_REL / "smoke_eval.manifest.json"
    inspect_json_rel = RESULTS_REL / "smoke_eval.inspect.json"
    inspect_md_rel = RESULTS_REL / "smoke_eval.inspect.md"
    inspect_csv_rel = RESULTS_REL / "smoke_eval.inspect.csv"
    inspect_fingerprints_csv_rel = RESULTS_REL / "smoke_eval.inspect.fingerprints.csv"
    inspect_junit_rel = RESULTS_REL / "smoke_eval.inspect.junit.xml"
    packed = ROOT / packed_rel
    manifest = ROOT / manifest_rel
    inspect_json = ROOT / inspect_json_rel
    inspect_md = ROOT / inspect_md_rel
    inspect_csv = ROOT / inspect_csv_rel
    inspect_fingerprints_csv = ROOT / inspect_fingerprints_csv_rel
    inspect_junit = ROOT / inspect_junit_rel

    pack_command = [
        sys.executable,
        str(BENCH / "dataset_pack.py"),
        "--input",
        str(SAMPLE_REL),
        "--output",
        str(packed_rel),
        "--manifest",
        str(manifest_rel),
        "--dataset",
        "smoke-eval",
        "--split",
        "validation",
        "--max-prompt-bytes",
        "4096",
        "--max-choice-bytes",
        "1024",
        "--max-record-payload-bytes",
        "8192",
    ]
    completed = run_command(pack_command)
    if completed.returncode:
        return completed.returncode
    if rc := require("records=3" in completed.stdout, "missing_pack_record_count"):
        return rc
    if rc := require(packed.exists() and packed.stat().st_size > 0, "missing_packed_binary"):
        return rc
    if rc := require(manifest.exists(), "missing_pack_manifest"):
        return rc

    pack_report = json.loads(manifest.read_text(encoding="utf-8"))
    if rc := require(pack_report["format"] == "hceval-mc", "unexpected_pack_format"):
        return rc
    if rc := require(pack_report["dataset"] == "smoke-eval", "unexpected_pack_dataset"):
        return rc
    if rc := require(pack_report["split"] == "validation", "unexpected_pack_split"):
        return rc
    if rc := require(pack_report["record_count"] == 3, "unexpected_pack_record_count"):
        return rc
    if rc := require(pack_report["choice_count_histogram"] == {"4": 3}, "unexpected_pack_choice_histogram"):
        return rc
    if rc := require(pack_report["answer_histogram"] == {"0": 3}, "unexpected_pack_answer_histogram"):
        return rc
    if rc := require(pack_report["binary_layout"]["binary_bytes"] == packed.stat().st_size, "binary_size_mismatch"):
        return rc
    if rc := require(len(pack_report["record_spans"]) == 3, "unexpected_record_spans"):
        return rc
    if rc := require(len(pack_report["record_fingerprints"]) == 3, "unexpected_record_fingerprints"):
        return rc
    if rc := require(len(pack_report["source_sha256"]) == 64, "missing_source_sha256"):
        return rc
    if rc := require(len(pack_report["binary_sha256"]) == 64, "missing_binary_sha256"):
        return rc

    inspect_command = [
        sys.executable,
        str(BENCH / "hceval_inspect.py"),
        "--input",
        str(packed_rel),
        "--manifest",
        str(manifest_rel),
        "--output",
        str(inspect_json_rel),
        "--markdown",
        str(inspect_md_rel),
        "--csv",
        str(inspect_csv_rel),
        "--fingerprints-csv",
        str(inspect_fingerprints_csv_rel),
        "--junit",
        str(inspect_junit_rel),
        "--max-prompt-bytes",
        "4096",
        "--max-choice-bytes",
        "1024",
        "--max-record-payload-bytes",
        "8192",
    ]
    completed = run_command(inspect_command)
    if completed.returncode:
        return completed.returncode

    inspect_report = json.loads(inspect_json.read_text(encoding="utf-8"))
    if rc := require(inspect_report["status"] == "pass", "unexpected_inspect_status"):
        return rc
    if rc := require(len(inspect_report["records"]) == 3, "unexpected_inspect_records"):
        return rc
    if rc := require(inspect_report["binary_layout"] == pack_report["binary_layout"], "inspect_layout_mismatch"):
        return rc
    if rc := require(inspect_report["record_spans"] == pack_report["record_spans"], "inspect_spans_mismatch"):
        return rc
    if rc := require(
        inspect_report["record_fingerprints"] == pack_report["record_fingerprints"],
        "inspect_fingerprints_mismatch",
    ):
        return rc
    if rc := require(inspect_report["source_sha256"] == pack_report["source_sha256"], "inspect_source_mismatch"):
        return rc
    if rc := require("HCEval Dataset Inspection" in inspect_md.read_text(encoding="utf-8"), "missing_inspect_md"):
        return rc
    inspect_rows = list(csv.DictReader(inspect_csv.open(encoding="utf-8", newline="")))
    if rc := require(len(inspect_rows) == 3, "unexpected_inspect_csv_rows"):
        return rc
    if rc := require(
        inspect_rows[0]["record_id"] == pack_report["record_spans"][0]["record_id"],
        "inspect_csv_id_mismatch",
    ):
        return rc
    fingerprint_rows = list(csv.DictReader(inspect_fingerprints_csv.open(encoding="utf-8", newline="")))
    if rc := require(len(fingerprint_rows) == 3, "unexpected_fingerprint_csv_rows"):
        return rc
    if rc := require(
        fingerprint_rows[0]["full_payload_sha256"] == pack_report["record_fingerprints"][0]["full_payload_sha256"],
        "fingerprint_csv_hash_mismatch",
    ):
        return rc
    junit_text = inspect_junit.read_text(encoding="utf-8")
    if rc := require('name="holyc_hceval_inspect"' in junit_text, "missing_inspect_junit"):
        return rc
    if rc := require('failures="0"' in junit_text, "unexpected_inspect_junit_failures"):
        return rc

    with tempfile.TemporaryDirectory(prefix="dataset-pack-ci-") as tmp:
        tmp_path = Path(tmp)
        bad_schema = tmp_path / "bad_schema.jsonl"
        bad_schema_out = tmp_path / "bad_schema.hceval"
        write_jsonl(
            bad_schema,
            [
                {
                    "id": "bad-answer",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Which answer is valid?",
                    "choices": ["A", "B"],
                    "answer_index": 4,
                    "provenance": "synthetic pack failure smoke",
                }
            ],
        )
        completed = run_command(
            [
                sys.executable,
                str(BENCH / "dataset_pack.py"),
                "--input",
                str(bad_schema),
                "--output",
                str(bad_schema_out),
            ],
            expected_failure=True,
        )
        if rc := require(completed.returncode == 2, "bad_schema_not_rejected"):
            return rc
        if rc := require("outside choice range" in completed.stderr, "missing_bad_schema_error"):
            return rc

        byte_gate = tmp_path / "byte_gate.jsonl"
        byte_gate_out = tmp_path / "byte_gate.hceval"
        write_jsonl(
            byte_gate,
            [
                {
                    "id": "too-large",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "This prompt should exceed the tiny smoke byte budget.",
                    "choices": ["yes", "no"],
                    "answer_index": 0,
                    "provenance": "synthetic pack byte-budget smoke",
                }
            ],
        )
        completed = run_command(
            [
                sys.executable,
                str(BENCH / "dataset_pack.py"),
                "--input",
                str(byte_gate),
                "--output",
                str(byte_gate_out),
                "--max-prompt-bytes",
                "8",
            ],
            expected_failure=True,
        )
        if rc := require(completed.returncode == 2, "byte_gate_not_rejected"):
            return rc
        if rc := require(
            "prompt is" in completed.stderr and "limit is 8" in completed.stderr,
            "missing_byte_gate_error",
        ):
            return rc

    print("dataset_pack_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
