#!/usr/bin/env python3
"""Tests for QEMU stdio hygiene audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_stdio_hygiene_audit


def artifact_row(prompt: str = "smoke-short", exit_class: str = "ok", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "exit_class": exit_class,
        "timed_out": exit_class == "timeout",
        "stdout_bytes": 90,
        "stderr_bytes": 0,
        "stdout_tail": "BENCH_RESULT: {}\n" if exit_class == "ok" else "launch failed\n",
        "stderr_tail": "",
        "failure_reason": "" if exit_class == "ok" else exit_class,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_stdio_hygiene_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_accepts_quiet_ok_rows_and_explained_failures(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        {"benchmarks": [artifact_row("ok"), artifact_row("timeout", "timeout", stdout_bytes=14)]},
    )
    args = parse_args([str(artifact), "--min-rows", "2"])

    rows, findings = qemu_stdio_hygiene_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 2
    assert rows[0].stderr_tail_bytes == 0
    assert rows[1].has_failure_signal is True


def test_audit_flags_noisy_ok_silent_failure_and_bad_counters(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        {
            "benchmarks": [
                artifact_row("noisy-ok", stderr_bytes=4, stderr_tail="warn"),
                artifact_row("silent-timeout", "timeout", stdout_bytes=0, stdout_tail="", failure_reason=""),
                artifact_row("tail-drift", stdout_bytes=1, stdout_tail="long tail"),
                artifact_row("missing-counter", stderr_bytes=None),
                artifact_row("negative-counter", stdout_bytes=-1, stdout_tail=""),
            ]
        },
    )
    args = parse_args([str(artifact)])

    rows, findings = qemu_stdio_hygiene_audit.audit([artifact], args)

    assert len(rows) == 5
    kinds = {finding.kind for finding in findings}
    assert {
        "ok_stderr_noise",
        "silent_failure",
        "tail_exceeds_counter",
        "missing_byte_counter",
        "negative_byte_counter",
    } <= kinds


def test_audit_flags_stdio_budget_overruns(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        {
            "benchmarks": [
                artifact_row(
                    "budget",
                    "timeout",
                    stdout_bytes=128,
                    stderr_bytes=64,
                    stdout_tail="x" * 33,
                    stderr_tail="y" * 17,
                    failure_reason="z" * 9,
                )
            ]
        },
    )
    args = parse_args(
        [
            str(artifact),
            "--max-stdout-bytes",
            "127",
            "--max-stderr-bytes",
            "63",
            "--max-stdout-tail-bytes",
            "32",
            "--max-stderr-tail-bytes",
            "16",
            "--max-failure-reason-bytes",
            "8",
        ]
    )

    rows, findings = qemu_stdio_hygiene_audit.audit([artifact], args)

    assert len(rows) == 1
    assert sum(1 for finding in findings if finding.kind == "stdio_budget_exceeded") == 5


def test_cli_writes_json_markdown_csv_rows_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, {"benchmarks": [artifact_row()]})
    output_dir = tmp_path / "out"

    status = qemu_stdio_hygiene_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "stdio"]
    )

    assert status == 0
    payload = json.loads((output_dir / "stdio.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["ok_rows"] == 1
    assert "No stdio hygiene findings." in (output_dir / "stdio.md").read_text(encoding="utf-8")
    summary_rows = list(csv.DictReader((output_dir / "stdio.csv").open(encoding="utf-8")))
    assert any(row["metric"] == "rows_with_stderr" for row in summary_rows)
    detail_rows = list(csv.DictReader((output_dir / "stdio_rows.csv").open(encoding="utf-8")))
    assert detail_rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "stdio_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "stdio_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_stdio_hygiene_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_quiet_ok_rows_and_explained_failures(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_noisy_ok_silent_failure_and_bad_counters(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_stdio_budget_overruns(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_rows_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
