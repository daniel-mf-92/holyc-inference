import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import qemu_timing_consistency_audit


def row(**overrides):
    payload = {
        "benchmark": "bench",
        "profile": "profile",
        "quantization": "Q8_0",
        "phase": "measured",
        "prompt": "prompt",
        "iteration": 1,
        "launch_index": 7,
        "tokens": 8,
        "elapsed_us": 400_000,
        "wall_elapsed_us": 500_000,
        "timeout_seconds": 10.0,
        "tok_per_s": 20.0,
        "wall_tok_per_s": 16.0,
        "us_per_token": 50_000.0,
        "wall_us_per_token": 62_500.0,
        "host_overhead_us": 100_000,
        "host_overhead_pct": 25.0,
        "wall_timeout_pct": 5.0,
        "host_child_user_cpu_us": 200_000,
        "host_child_system_cpu_us": 50_000,
        "host_child_cpu_us": 250_000,
        "ttft_us": 100_000,
    }
    payload.update(overrides)
    return payload


def args():
    return argparse.Namespace(pattern=["qemu_prompt_bench*.json"], measured_only=False, rel_tolerance=1e-6, abs_tolerance=1e-6)


def test_audit_artifact_passes_consistent_timing(tmp_path):
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    artifact.write_text(json.dumps({"benchmarks": [row()]}) + "\n", encoding="utf-8")

    records, findings = qemu_timing_consistency_audit.audit_artifact(artifact, args())

    assert len(records) == 1
    assert findings == []


def test_audit_artifact_flags_derived_metric_drift(tmp_path):
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    artifact.write_text(json.dumps({"benchmarks": [row(wall_tok_per_s=99.0, host_child_cpu_us=1)]}) + "\n", encoding="utf-8")

    records, findings = qemu_timing_consistency_audit.audit_artifact(artifact, args())

    assert len(records) == 1
    assert {finding.field for finding in findings} == {"wall_tok_per_s", "host_child_cpu_us"}


def test_audit_artifact_flags_wall_elapsed_before_guest_elapsed(tmp_path):
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    artifact.write_text(
        json.dumps(
            {
                "benchmarks": [
                    row(
                        wall_elapsed_us=300_000,
                        wall_tok_per_s=26.666666666666668,
                        wall_us_per_token=37_500.0,
                        host_overhead_us=-100_000,
                        host_overhead_pct=-25.0,
                        wall_timeout_pct=3.0,
                    )
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records, findings = qemu_timing_consistency_audit.audit_artifact(artifact, args())

    assert len(records) == 1
    assert any(finding.kind == "wall_elapsed_before_guest_elapsed" for finding in findings)


def test_main_writes_failure_artifacts(tmp_path):
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    artifact.write_text(json.dumps({"benchmarks": [row(ttft_us=999_999)]}) + "\n", encoding="utf-8")

    status = qemu_timing_consistency_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "timing"])

    assert status == 1
    payload = json.loads((output_dir / "timing.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["findings"][0]["kind"] == "ttft_exceeds_elapsed"
