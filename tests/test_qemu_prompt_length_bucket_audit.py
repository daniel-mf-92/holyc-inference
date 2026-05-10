#!/usr/bin/env python3
"""Tests for QEMU prompt length bucket audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_length_bucket_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt": "short",
        "prompt_sha256": "sha-short",
        "phase": "measured",
        "exit_class": "ok",
        "prompt_bytes": 32,
        "tokens": 16,
        "wall_tok_per_s": 120.0,
        "ttft_us": 1000.0,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_prompt_length_bucket_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_bucket_parser_accepts_open_ended_maximum() -> None:
    bucket = qemu_prompt_length_bucket_audit.parse_bucket("long:1024:")

    assert bucket.name == "long"
    assert bucket.min_bytes == 1024
    assert bucket.max_bytes is None


def test_audit_summarizes_prompt_length_buckets(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(prompt="short-a", prompt_sha256="a", prompt_bytes=32, tokens=8, wall_tok_per_s=100.0),
            artifact_row(prompt="short-b", prompt_sha256="b", prompt_bytes=64, tokens=16, wall_tok_per_s=120.0),
            artifact_row(prompt="long", prompt_sha256="c", prompt_bytes=2048, tokens=32, wall_tok_per_s=80.0),
            artifact_row(prompt="warmup", phase="warmup", prompt_bytes=4096, wall_tok_per_s=1.0),
        ],
    )
    args = parse_args([str(artifact), "--bucket", "short:0:255", "--bucket", "long:1024:", "--require-buckets"])

    samples, findings = qemu_prompt_length_bucket_audit.collect_samples(args.inputs, ["qemu_prompt_bench*.json"], args.include_warmups)
    report = qemu_prompt_length_bucket_audit.build_report(samples, args.bucket, findings, args)

    assert report["status"] == "pass"
    assert report["summary"]["samples"] == 3
    short = report["buckets"][0]
    assert short["bucket"] == "short"
    assert short["rows"] == 2
    assert short["unique_prompts"] == 2
    assert short["tokens_total"] == 24
    assert short["wall_tok_per_s_p50"] == 110.0


def test_audit_flags_empty_bucket_failures_and_missing_prompt_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(prompt_bytes=32), artifact_row(prompt="missing-bytes", prompt_bytes="")])
    args = parse_args(
        [
            str(artifact),
            "--bucket",
            "short:0:255",
            "--bucket",
            "long:1024:",
            "--require-buckets",
            "--min-successful-samples-per-bucket",
            "1",
            "--min-prompts-per-bucket",
            "1",
        ]
    )

    samples, findings = qemu_prompt_length_bucket_audit.collect_samples(args.inputs, ["qemu_prompt_bench*.json"], args.include_warmups)
    report = qemu_prompt_length_bucket_audit.build_report(samples, args.bucket, findings, args)

    kinds = {finding["kind"] for finding in report["findings"]}
    assert report["status"] == "fail"
    assert {"missing_prompt_bytes", "empty_bucket", "insufficient_successful_samples", "insufficient_unique_prompts"} <= kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(prompt="long", prompt_sha256="long", prompt_bytes=2048)])
    output_dir = tmp_path / "out"

    status = qemu_prompt_length_bucket_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "length_bucket",
            "--bucket",
            "short:0:255",
            "--bucket",
            "long:1024:",
            "--require-buckets",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "length_bucket.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["buckets"] == 2
    assert "QEMU Prompt Length Bucket Audit" in (output_dir / "length_bucket.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "length_bucket.csv").open(encoding="utf-8")))
    assert [row["bucket"] for row in rows] == ["short", "long"]
    finding_rows = list(csv.DictReader((output_dir / "length_bucket_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "length_bucket_junit.xml").getroot()
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    test_bucket_parser_accepts_open_ended_maximum()
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_summarizes_prompt_length_buckets(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_empty_bucket_failures_and_missing_prompt_bytes(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
