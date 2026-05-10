#!/usr/bin/env python3
"""Tests for QEMU prompt balance audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_balance_audit


def row(prompt: str, iteration: int, *, phase: str = "measured", exit_class: str = "ok", timed_out: bool = False) -> dict[str, object]:
    return {
        "benchmark": "qemu_prompt",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": phase,
        "prompt": prompt,
        "iteration": iteration,
        "exit_class": exit_class,
        "timed_out": timed_out,
        "failure_reason": None if exit_class == "ok" and not timed_out else "failed",
    }


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:00:00Z",
                "status": "pass",
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "benchmarks": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(extra: list[str]) -> object:
    return qemu_prompt_balance_audit.build_parser().parse_args(extra)


def test_audit_accepts_balanced_prompt_samples(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact_path,
        [row("alpha", 1), row("alpha", 2), row("beta", 1), row("beta", 2), row("alpha", 0, phase="warmup")],
    )
    args = parse_args([str(artifact_path), "--min-prompts", "2", "--min-successful-runs-per-prompt", "2"])

    artifact, rows, findings = qemu_prompt_balance_audit.audit_artifact(artifact_path, args)

    assert artifact.status == "pass"
    assert artifact.prompts == 2
    assert artifact.successful_run_delta == 0
    assert {item.prompt: item.warmup_runs for item in rows}["alpha"] == 1
    assert findings == []


def test_audit_flags_unbalanced_and_failed_samples(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [row("alpha", 1), row("alpha", 2), row("beta", 1, exit_class="error")])
    args = parse_args([str(artifact_path), "--max-successful-run-delta", "0", "--fail-on-failed-runs"])

    artifact, _, findings = qemu_prompt_balance_audit.audit_artifact(artifact_path, args)

    assert artifact.status == "fail"
    kinds = {finding.kind for finding in findings}
    assert {"successful_run_delta", "failed_runs"} <= kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [row("alpha", 1), row("beta", 1)])
    output_dir = tmp_path / "out"

    status = qemu_prompt_balance_audit.main(
        [str(artifact_path), "--output-dir", str(output_dir), "--output-stem", "balance", "--min-prompts", "2"]
    )

    assert status == 0
    payload = json.loads((output_dir / "balance.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["prompts"] == 2
    assert "No prompt balance findings." in (output_dir / "balance.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "balance.csv").open(encoding="utf-8")))
    assert {item["prompt"] for item in rows} == {"alpha", "beta"}
    findings = list(csv.DictReader((output_dir / "balance_findings.csv").open(encoding="utf-8")))
    assert findings == []
    junit_root = ET.parse(output_dir / "balance_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_prompt_balance_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    status = qemu_prompt_balance_audit.main(
        [str(input_dir), "--output-dir", str(output_dir), "--output-stem", "balance", "--min-artifacts", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "balance.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_balanced_prompt_samples(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_unbalanced_and_failed_samples(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_artifacts_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
