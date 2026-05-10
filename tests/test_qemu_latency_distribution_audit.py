#!/usr/bin/env python3
"""Tests for QEMU latency distribution audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_latency_distribution_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 32,
        "elapsed_us": 30_000,
        "wall_elapsed_us": 32_000,
        "ttft_us": 5_000,
        "us_per_token": 937.5,
        "wall_us_per_token": 1_000.0,
        "tok_per_s": 1_066.6666667,
        "wall_tok_per_s": 1_000.0,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_latency_distribution_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_latency_samples_and_groups(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(wall_elapsed_us=34_000, wall_us_per_token=1_062.5, wall_tok_per_s=941.1764706)])
    args = parse_args([str(artifact), "--min-samples-per-group", "2"])
    args.pattern = ["qemu_prompt_bench*.json"]
    args.require_metric = ["wall_elapsed_us", "wall_us_per_token", "wall_tok_per_s"]

    samples, groups, findings = qemu_latency_distribution_audit.audit([artifact], args)

    assert findings == []
    assert len(samples) == 2
    assert len(groups) == 1
    assert groups[0].samples == 2
    assert groups[0].wall_us_per_token_p95 is not None


def test_audit_flags_missing_metric_drift_and_slo_failure(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    row = artifact_row(wall_us_per_token=1.0, wall_tok_per_s="")
    slow_tail = artifact_row(prompt="smoke-short", wall_tok_per_s=1.0)
    write_artifact(artifact, [row, slow_tail])
    args = parse_args([str(artifact), "--max-p95-wall-us-per-token", "0.5", "--min-p05-wall-tok-per-s", "2000"])
    args.pattern = ["qemu_prompt_bench*.json"]
    args.require_metric = ["wall_elapsed_us", "wall_us_per_token", "wall_tok_per_s"]

    samples, groups, findings = qemu_latency_distribution_audit.audit([artifact], args)

    assert len(samples) == 2
    assert len(groups) == 1
    kinds = {finding.kind for finding in findings}
    assert {"missing_metric", "wall_us_per_token_drift", "max_p95_wall_us_per_token", "min_p05_wall_tok_per_s"} <= kinds


def test_cli_writes_json_markdown_csv_samples_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_latency_distribution_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "latency",
            "--min-rows",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "latency.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["groups"] == 1
    assert "No latency distribution findings." in (output_dir / "latency.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "latency.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    assert rows[0]["wall_tok_per_s_p05"] == "1000.0"
    sample_rows = list(csv.DictReader((output_dir / "latency_samples.csv").open(encoding="utf-8")))
    assert sample_rows[0]["wall_elapsed_us"] == "32000.0"
    finding_rows = list(csv.DictReader((output_dir / "latency_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "latency_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_latency_distribution_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_latency_distribution_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "latency",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "latency.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_latency_samples_and_groups(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_metric_drift_and_slo_failure(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_samples_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
