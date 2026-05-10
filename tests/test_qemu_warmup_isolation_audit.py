#!/usr/bin/env python3
"""Tests for QEMU warmup isolation audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_warmup_isolation_audit


def row(*, phase: str, launch_index: int, tokens: int = 16) -> dict[str, object]:
    return {
        "prompt": "smoke-short",
        "phase": phase,
        "launch_index": launch_index,
        "tokens": tokens,
        "exit_class": "ok",
    }


def artifact(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "planned_warmup_launches": 1,
        "planned_measured_launches": 2,
        "warmups": [row(phase="warmup", launch_index=1, tokens=16)],
        "benchmarks": [row(phase="measured", launch_index=2, tokens=32), row(phase="measured", launch_index=3, tokens=32)],
        "suite_summary": {"total_tokens": 64},
        "phase_summaries": {"warmup": {"runs": 1}, "measured": {"runs": 2}},
    }
    payload.update(overrides)
    return payload


def parse_args(extra: list[str]) -> object:
    return qemu_warmup_isolation_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_accepts_isolated_warmups(tmp_path: Path) -> None:
    path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(path, artifact(phase_summaries=[{"phase": "warmup", "runs": 1}, {"phase": "measured", "runs": 2}]))
    args = parse_args([str(path), "--require-phase-summaries"])
    args.pattern = ["qemu_prompt_bench*.json"]

    artifacts, findings = qemu_warmup_isolation_audit.audit([path], args)

    assert findings == []
    assert len(artifacts) == 1
    assert artifacts[0].warmup_tokens_total == 16
    assert artifacts[0].measured_tokens_total == 64


def test_audit_flags_warmup_leaks_and_count_drift(tmp_path: Path) -> None:
    path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        path,
        artifact(
            planned_warmup_launches=2,
            benchmarks=[row(phase="warmup", launch_index=1, tokens=32)],
            suite_summary={"total_tokens": 48},
        ),
    )
    args = parse_args([str(path), "--min-measured-rows", "2"])
    args.pattern = ["qemu_prompt_bench*.json"]

    artifacts, findings = qemu_warmup_isolation_audit.audit([path], args)

    assert len(artifacts) == 1
    kinds = {finding.kind for finding in findings}
    assert {"planned_warmup_count_drift", "measured_phase_drift", "launch_index_overlap", "suite_summary_warmup_leak", "min_measured_rows"} <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(path, artifact())
    output_dir = tmp_path / "out"

    status = qemu_warmup_isolation_audit.main(
        [
            str(path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "warmup",
            "--require-phase-summaries",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "warmup.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["measured_rows"] == 2
    assert "No warmup isolation findings." in (output_dir / "warmup.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "warmup.csv").open(encoding="utf-8")))
    assert rows[0]["measured_tokens_total"] == "64"
    finding_rows = list(csv.DictReader((output_dir / "warmup_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "warmup_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_warmup_isolation_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_artifact_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    status = qemu_warmup_isolation_audit.main(
        [
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "warmup",
            "--min-artifacts",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "warmup.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_isolated_warmups(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_warmup_leaks_and_count_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_artifact_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
