#!/usr/bin/env python3
"""Tests for QEMU launch order audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_launch_order_audit


def row(*, phase: str, launch_index: int, timestamp: str = "2026-05-01T00:00:01Z") -> dict[str, object]:
    return {
        "phase": phase,
        "launch_index": launch_index,
        "prompt": "smoke-short",
        "iteration": 1,
        "timestamp": timestamp,
        "wall_elapsed_us": 250000,
        "exit_class": "ok",
    }


def artifact(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "planned_total_launches": 3,
        "planned_warmup_launches": 1,
        "planned_measured_launches": 2,
        "warmups": [row(phase="warmup", launch_index=1, timestamp="2026-05-01T00:00:01Z")],
        "benchmarks": [
            row(phase="measured", launch_index=2, timestamp="2026-05-01T00:00:02Z"),
            row(phase="measured", launch_index=3, timestamp="2026-05-01T00:00:03Z"),
        ],
    }
    payload.update(overrides)
    return payload


def parse_args(extra: list[str]) -> object:
    return qemu_launch_order_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_accepts_contiguous_launch_order(tmp_path: Path) -> None:
    path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(path, artifact())
    args = parse_args([str(path)])
    args.pattern = ["qemu_prompt_bench*.json"]

    artifacts, rows, findings = qemu_launch_order_audit.audit([path], args)

    assert findings == []
    assert len(artifacts) == 1
    assert len(rows) == 3
    assert artifacts[0].min_launch_index == 1
    assert artifacts[0].max_launch_index == 3


def test_audit_flags_duplicate_gapped_and_misordered_launches(tmp_path: Path) -> None:
    path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        path,
        artifact(
            planned_total_launches=4,
            warmups=[row(phase="warmup", launch_index=3, timestamp="2026-05-01T00:00:03Z")],
            benchmarks=[
                row(phase="measured", launch_index=1, timestamp="2026-05-01T00:00:01Z"),
                row(phase="measured", launch_index=3, timestamp="2026-05-01T00:00:02Z"),
            ],
        ),
    )
    args = parse_args([str(path), "--timestamp-tolerance-us", "0"])
    args.pattern = ["qemu_prompt_bench*.json"]

    artifacts, rows, findings = qemu_launch_order_audit.audit([path], args)

    assert len(artifacts) == 1
    assert len(rows) == 3
    kinds = {finding.kind for finding in findings}
    assert {"duplicate_launch_index", "launch_index_gap", "planned_total_drift", "warmup_after_measured", "timestamp_regressed"} <= kinds


def test_cli_writes_json_markdown_csv_rows_findings_and_junit(tmp_path: Path) -> None:
    path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(path, artifact())
    output_dir = tmp_path / "out"

    status = qemu_launch_order_audit.main([str(path), "--output-dir", str(output_dir), "--output-stem", "launch_order"])

    assert status == 0
    payload = json.loads((output_dir / "launch_order.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["launch_rows"] == 3
    assert "No launch order findings." in (output_dir / "launch_order.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "launch_order.csv").open(encoding="utf-8")))
    assert rows[0]["launch_rows"] == "3"
    launch_rows = list(csv.DictReader((output_dir / "launch_order_rows.csv").open(encoding="utf-8")))
    assert launch_rows[0]["phase"] == "warmup"
    finding_rows = list(csv.DictReader((output_dir / "launch_order_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "launch_order_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_launch_order_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_artifact_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    status = qemu_launch_order_audit.main([str(tmp_path), "--output-dir", str(output_dir), "--output-stem", "launch_order", "--min-artifacts", "1"])

    assert status == 1
    payload = json.loads((output_dir / "launch_order.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_contiguous_launch_order(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_duplicate_gapped_and_misordered_launches(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_rows_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_artifact_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
