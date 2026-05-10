#!/usr/bin/env python3
"""Tests for QEMU prompt coverage audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_bench
import qemu_prompt_coverage_audit


def write_prompt_suite(path: Path) -> list[qemu_prompt_bench.PromptCase]:
    rows = [
        {"id": "alpha", "prompt": "Alpha prompt", "expected_tokens": 4},
        {"id": "beta", "prompt": "Beta prompt", "expected_tokens": 8},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return qemu_prompt_bench.load_prompt_cases(path)


def bench_row(case: qemu_prompt_bench.PromptCase, *, exit_class: str = "ok") -> dict[str, object]:
    prompt_sha = qemu_prompt_bench.prompt_hash(case.prompt)
    return {
        "phase": "measured",
        "prompt": case.prompt_id,
        "prompt_sha256": prompt_sha,
        "guest_prompt_sha256": prompt_sha,
        "guest_prompt_sha256_match": True,
        "guest_prompt_bytes_match": True,
        "tokens": case.expected_tokens,
        "expected_tokens": case.expected_tokens,
        "expected_tokens_match": True,
        "exit_class": exit_class,
        "timed_out": False,
        "failure_reason": None,
    }


def write_artifact(path: Path, suite: Path, cases: list[qemu_prompt_bench.PromptCase], rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q8_0",
                "prompt_suite": {
                    "source": str(suite),
                    "prompt_count": len(cases),
                    "suite_sha256": qemu_prompt_bench.prompt_suite_hash(cases),
                },
                "benchmarks": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_accepts_full_successful_prompt_coverage(tmp_path: Path) -> None:
    suite = tmp_path / "prompts.jsonl"
    cases = write_prompt_suite(suite)
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, suite, cases, [bench_row(cases[0]), bench_row(cases[1])])
    args = qemu_prompt_coverage_audit.build_parser().parse_args(
        [str(artifact), "--require-suite-file", "--require-success", "--min-runs-per-prompt", "1"]
    )

    coverage, prompt_rows, findings = qemu_prompt_coverage_audit.audit_artifact(artifact, args)

    assert coverage.status == "pass"
    assert coverage.expected_prompts == 2
    assert coverage.measured_prompts == 2
    assert coverage.min_successful_runs_per_expected_prompt == 1
    assert len(prompt_rows) == 2
    assert findings == []


def test_audit_flags_missing_and_failed_prompt_runs(tmp_path: Path) -> None:
    suite = tmp_path / "prompts.jsonl"
    cases = write_prompt_suite(suite)
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, suite, cases, [bench_row(cases[0], exit_class="nonzero_exit")])
    args = qemu_prompt_coverage_audit.build_parser().parse_args(
        [str(artifact), "--require-suite-file", "--require-success", "--min-runs-per-prompt", "1"]
    )

    coverage, _prompt_rows, findings = qemu_prompt_coverage_audit.audit_artifact(artifact, args)

    assert coverage.status == "fail"
    assert coverage.missing_prompts == 1
    assert {finding.kind for finding in findings} == {"failed_runs", "min_runs_per_prompt", "missing_prompt"}


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    suite = tmp_path / "prompts.jsonl"
    cases = write_prompt_suite(suite)
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, suite, cases, [bench_row(cases[0]), bench_row(cases[1])])
    output_dir = tmp_path / "out"

    status = qemu_prompt_coverage_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "coverage",
            "--require-suite-file",
            "--require-success",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "coverage.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["expected_prompts"] == 2
    assert "QEMU Prompt Coverage Audit" in (output_dir / "coverage.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "coverage_prompts.csv").open(encoding="utf-8")))
    assert [row["prompt"] for row in rows] == ["alpha", "beta"]
    junit_root = ET.parse(output_dir / "coverage_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_prompt_coverage_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    status = qemu_prompt_coverage_audit.main(
        [
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "coverage",
            "--min-artifacts",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "coverage.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pass"
        path.mkdir()
        test_audit_accepts_full_successful_prompt_coverage(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fail"
        path.mkdir()
        test_audit_flags_missing_and_failed_prompt_runs(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cli"
        path.mkdir()
        test_cli_writes_json_markdown_csv_and_junit(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty"
        path.mkdir()
        test_cli_fails_when_no_artifacts_match(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
