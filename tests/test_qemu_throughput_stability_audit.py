#!/usr/bin/env python3
"""Tests for QEMU throughput stability audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_throughput_stability_audit


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
        "tok_per_s": 1066.6666667,
        "wall_tok_per_s": 1000.0,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_throughput_stability_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_stable_throughput_samples_and_groups(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(wall_elapsed_us=33_000, wall_tok_per_s=969.6969697)])
    args = parse_args([str(artifact), "--min-samples-per-group", "2", "--max-wall-tok-per-s-cv", "0.05"])
    args.pattern = ["qemu_prompt_bench*.json"]
    args.require_metric = ["wall_elapsed_us", "wall_tok_per_s"]

    samples, groups, findings = qemu_throughput_stability_audit.audit([artifact], args)

    assert findings == []
    assert len(samples) == 2
    assert len(groups) == 1
    assert groups[0].samples == 2
    assert groups[0].wall_tok_per_s_cv is not None


def test_audit_flags_missing_metric_drift_and_cv_failure(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(wall_tok_per_s=""),
            artifact_row(wall_elapsed_us=96_000, wall_tok_per_s=333.3333333),
            artifact_row(wall_elapsed_us=34_000, wall_tok_per_s=941.1764706),
        ],
    )
    args = parse_args([str(artifact), "--min-samples-per-group", "2", "--min-wall-tok-per-s", "900", "--max-wall-tok-per-s-cv", "0.05"])
    args.pattern = ["qemu_prompt_bench*.json"]
    args.require_metric = ["wall_elapsed_us", "wall_tok_per_s"]

    samples, groups, findings = qemu_throughput_stability_audit.audit([artifact], args)

    assert len(samples) == 3
    assert len(groups) == 1
    kinds = {finding.kind for finding in findings}
    assert {"missing_metric", "min_wall_tok_per_s", "max_wall_tok_per_s_cv"} <= kinds


def test_cli_writes_json_markdown_csv_samples_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_throughput_stability_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "throughput",
            "--min-rows",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "throughput.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["groups"] == 1
    assert "No throughput stability findings." in (output_dir / "throughput.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "throughput.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    sample_rows = list(csv.DictReader((output_dir / "throughput_samples.csv").open(encoding="utf-8")))
    assert sample_rows[0]["wall_tok_per_s"] == "1000.0"
    finding_rows = list(csv.DictReader((output_dir / "throughput_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "throughput_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_throughput_stability_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_throughput_stability_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "throughput",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "throughput.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_stable_throughput_samples_and_groups(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_metric_drift_and_cv_failure(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_samples_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
