#!/usr/bin/env python3
"""Tests for QEMU prompt outlier audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_outlier_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "exit_class": "ok",
        "iteration": 1,
        "wall_elapsed_us": 32_000,
        "wall_tok_per_s": 1000.0,
        "host_overhead_pct": 5.0,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_prompt_outlier_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_repeated_stable_prompt_runs(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(iteration=1, wall_elapsed_us=32_000, wall_tok_per_s=1000.0),
            artifact_row(iteration=2, wall_elapsed_us=33_000, wall_tok_per_s=969.6969697),
            artifact_row(iteration=3, wall_elapsed_us=31_500, wall_tok_per_s=1015.8730159),
        ],
    )
    args = parse_args([str(artifact), "--min-samples-per-group", "3", "--max-relative-delta-pct", "10"])

    samples, groups, findings = qemu_prompt_outlier_audit.audit([artifact], args)

    assert findings == []
    assert len(samples) == 3
    assert len(groups) == 1
    assert groups[0].wall_elapsed_us_median == 32_000
    assert groups[0].wall_tok_per_s_mad is not None


def test_audit_flags_missing_metrics_and_relative_outliers(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(iteration=1, wall_elapsed_us=32_000, wall_tok_per_s=1000.0),
            artifact_row(iteration=2, wall_elapsed_us=33_000, wall_tok_per_s=969.6969697),
            artifact_row(iteration=3, wall_elapsed_us=96_000, wall_tok_per_s=333.3333333),
            artifact_row(iteration=4, wall_tok_per_s=""),
        ],
    )
    args = parse_args([str(artifact), "--min-samples-per-group", "3", "--max-relative-delta-pct", "20"])

    samples, groups, findings = qemu_prompt_outlier_audit.audit([artifact], args)

    assert len(samples) == 4
    assert len(groups) == 1
    kinds = {finding.kind for finding in findings}
    assert {"missing_metric", "relative_outlier"} <= kinds
    assert any(finding.metric == "wall_elapsed_us" for finding in findings)
    assert any(finding.metric == "wall_tok_per_s" for finding in findings)


def test_cli_writes_json_markdown_csv_samples_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(iteration=1),
            artifact_row(iteration=2, wall_elapsed_us=33_000, wall_tok_per_s=969.6969697),
            artifact_row(iteration=3, wall_elapsed_us=31_500, wall_tok_per_s=1015.8730159),
        ],
    )
    output_dir = tmp_path / "out"

    status = qemu_prompt_outlier_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "outlier",
            "--min-rows",
            "3",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "outlier.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["groups"] == 1
    assert "No prompt outlier findings." in (output_dir / "outlier.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "outlier.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    sample_rows = list(csv.DictReader((output_dir / "outlier_samples.csv").open(encoding="utf-8")))
    assert sample_rows[0]["wall_tok_per_s"] == "1000.0"
    finding_rows = list(csv.DictReader((output_dir / "outlier_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "outlier_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_prompt_outlier_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_group_sample_floor_is_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_prompt_outlier_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "outlier",
            "--min-samples-per-group",
            "3",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "outlier.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_samples_per_group"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_repeated_stable_prompt_runs(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_metrics_and_relative_outliers(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_samples_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_group_sample_floor_is_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
