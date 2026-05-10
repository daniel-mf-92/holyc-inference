#!/usr/bin/env python3
"""Tests for QEMU summary consistency audit tooling."""

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
import qemu_summary_consistency_audit


COMMAND = ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"]


def bench_run(prompt: str, tokens: int, elapsed_us: int) -> qemu_prompt_bench.BenchRun:
    airgap = qemu_prompt_bench.command_airgap_metadata(COMMAND)
    prompt_text = f"{prompt} prompt"
    return qemu_prompt_bench.BenchRun(
        benchmark="qemu_prompt",
        profile="ci-airgap-smoke",
        model="synthetic-smoke",
        quantization="Q4_0",
        phase="measured",
        launch_index=1,
        prompt=prompt,
        prompt_sha256=qemu_prompt_bench.prompt_hash(prompt_text),
        guest_prompt_sha256=qemu_prompt_bench.prompt_hash(prompt_text),
        guest_prompt_sha256_match=True,
        prompt_bytes=qemu_prompt_bench.prompt_bytes(prompt_text),
        guest_prompt_bytes=qemu_prompt_bench.prompt_bytes(prompt_text),
        guest_prompt_bytes_match=True,
        iteration=1,
        commit="abc123",
        timestamp="2026-04-30T00:00:00Z",
        tokens=tokens,
        expected_tokens=tokens,
        expected_tokens_match=True,
        elapsed_us=elapsed_us,
        wall_elapsed_us=elapsed_us + 1000,
        timeout_seconds=5.0,
        wall_timeout_pct=1.0,
        host_overhead_us=1000,
        host_overhead_pct=1.0,
        host_child_user_cpu_us=1000,
        host_child_system_cpu_us=500,
        host_child_cpu_us=1500,
        host_child_cpu_pct=10.0,
        host_child_tok_per_cpu_s=1000.0,
        host_child_peak_rss_bytes=1024,
        ttft_us=5000,
        tok_per_s=qemu_prompt_bench.derived_tok_per_s(tokens, elapsed_us),
        wall_tok_per_s=qemu_prompt_bench.derived_tok_per_s(tokens, elapsed_us + 1000),
        prompt_bytes_per_s=qemu_prompt_bench.derived_bytes_per_s(qemu_prompt_bench.prompt_bytes(prompt_text), elapsed_us),
        wall_prompt_bytes_per_s=qemu_prompt_bench.derived_bytes_per_s(
            qemu_prompt_bench.prompt_bytes(prompt_text), elapsed_us + 1000
        ),
        tokens_per_prompt_byte=qemu_prompt_bench.derived_tokens_per_prompt_byte(
            tokens, qemu_prompt_bench.prompt_bytes(prompt_text)
        ),
        us_per_token=qemu_prompt_bench.derived_us_per_token(tokens, elapsed_us),
        wall_us_per_token=qemu_prompt_bench.derived_us_per_token(tokens, elapsed_us + 1000),
        memory_bytes=2048,
        memory_bytes_per_token=qemu_prompt_bench.derived_memory_bytes_per_token(2048, tokens),
        stdout_bytes=50,
        stderr_bytes=0,
        serial_output_bytes=50,
        serial_output_lines=1,
        returncode=0,
        timed_out=False,
        exit_class="ok",
        failure_reason=None,
        command=COMMAND,
        command_sha256=qemu_prompt_bench.command_hash(COMMAND),
        command_airgap_ok=airgap["ok"],
        command_has_explicit_nic_none=airgap["explicit_nic_none"],
        command_has_legacy_net_none=airgap["legacy_net_none"],
        command_airgap_violations=tuple(airgap["violations"]),
        stdout_tail="",
        stderr_tail="",
    )


def write_artifact(path: Path, runs: list[qemu_prompt_bench.BenchRun]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-30T00:00:00Z",
                "status": "pass",
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "benchmarks": [qemu_prompt_bench.asdict(run) for run in runs],
                "suite_summary": qemu_prompt_bench.suite_summary(runs),
                "summaries": qemu_prompt_bench.summarize_runs(runs),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_accepts_consistent_summary(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [bench_run("alpha", 16, 100000), bench_run("beta", 32, 200000)])
    args = qemu_summary_consistency_audit.build_parser().parse_args([str(artifact_path)])

    artifact, findings = qemu_summary_consistency_audit.audit_artifact(artifact_path, args)

    assert artifact.status == "pass"
    assert artifact.measured_rows == 2
    assert artifact.expected_prompt_summaries == 2
    assert artifact.checked_fields > 20
    assert findings == []


def test_audit_flags_summary_drift(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [bench_run("alpha", 16, 100000)])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["suite_summary"]["runs"] = 99
    payload["summaries"][0]["tokens_median"] = 99
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args = qemu_summary_consistency_audit.build_parser().parse_args([str(artifact_path)])

    artifact, findings = qemu_summary_consistency_audit.audit_artifact(artifact_path, args)

    assert artifact.status == "fail"
    assert ("suite", "runs", "value_mismatch") in {
        (finding.scope, finding.field, finding.kind) for finding in findings
    }
    assert any(finding.scope == "prompt:alpha" and finding.field == "tokens_median" for finding in findings)


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [bench_run("alpha", 16, 100000)])
    output_dir = tmp_path / "out"

    status = qemu_summary_consistency_audit.main(
        [str(artifact_path), "--output-dir", str(output_dir), "--output-stem", "summary"]
    )

    assert status == 0
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["artifacts"] == 1
    assert "QEMU Summary Consistency Audit" in (output_dir / "summary.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "summary.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "pass"
    finding_rows = list(csv.DictReader((output_dir / "summary_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "summary_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_summary_consistency_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    status = qemu_summary_consistency_audit.main(
        [str(input_dir), "--output-dir", str(output_dir), "--output-stem", "summary", "--min-artifacts", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_summary(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_summary_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_artifacts_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
