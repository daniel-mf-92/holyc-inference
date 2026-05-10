from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_launch_jsonl_parity_audit


def write_artifact(path: Path, *, sidecar_rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "generated_at": "2026-05-02T00:00:00Z",
        "status": "pass",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "command_sha256": "c" * 64,
        "command_airgap": {"ok": True, "explicit_nic_none": True, "legacy_net_none": False, "violations": []},
        "launch_plan_sha256": "a" * 64,
        "expected_launch_sequence_sha256": "b" * 64,
        "observed_launch_sequence_sha256": "b" * 64,
        "prompt_suite": {"suite_sha256": "d" * 64},
        "warmups": [
            {
                "benchmark": "qemu_prompt",
                "phase": "warmup",
                "launch_index": 1,
                "iteration": 1,
                "prompt": "smoke-short",
                "prompt_sha256": "e" * 64,
                "tokens": 4,
                "elapsed_us": 1000,
                "wall_elapsed_us": 1200,
                "returncode": 0,
                "timed_out": False,
                "exit_class": "ok",
            }
        ],
        "benchmarks": [
            {
                "benchmark": "qemu_prompt",
                "phase": "measured",
                "launch_index": 2,
                "iteration": 1,
                "prompt": "smoke-short",
                "prompt_sha256": "e" * 64,
                "tokens": 4,
                "elapsed_us": 1000,
                "wall_elapsed_us": 1200,
                "returncode": 0,
                "timed_out": False,
                "exit_class": "ok",
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if sidecar_rows is None:
        sidecar_rows = qemu_launch_jsonl_parity_audit.launch_rows(payload)
    path.with_name("qemu_prompt_bench_launches_latest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in sidecar_rows),
        encoding="utf-8",
    )
    return payload


def test_audit_accepts_matching_launch_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(source)
    args = qemu_launch_jsonl_parity_audit.parse_args([str(source)])

    records, findings = qemu_launch_jsonl_parity_audit.audit([source], args)

    assert findings == []
    assert len(records) == 1
    assert records[0].json_rows == 2
    assert records[0].jsonl_rows == 2


def test_audit_rejects_value_and_row_count_drift(tmp_path: Path) -> None:
    source = tmp_path / "qemu_prompt_bench_latest.json"
    payload = write_artifact(source)
    rows = qemu_launch_jsonl_parity_audit.launch_rows(payload)
    rows[0]["phase"] = "measured"
    rows.pop()
    write_artifact(source, sidecar_rows=rows)
    args = qemu_launch_jsonl_parity_audit.parse_args([str(source)])

    _records, findings = qemu_launch_jsonl_parity_audit.audit([source], args)
    kinds = {finding.kind for finding in findings}

    assert "row_count_mismatch" in kinds
    assert "value_mismatch" in kinds


def test_cli_writes_reports_and_junit(tmp_path: Path) -> None:
    source = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_artifact(source)

    status = qemu_launch_jsonl_parity_audit.main(
        [str(source), "--output-dir", str(output_dir), "--output-stem", "launch_jsonl_parity"]
    )

    assert status == 0
    report = json.loads((output_dir / "launch_jsonl_parity.json").read_text(encoding="utf-8"))
    junit = ET.parse(output_dir / "launch_jsonl_parity_junit.xml").getroot()
    assert report["status"] == "pass"
    assert report["summary"]["json_rows"] == 2
    assert junit.attrib["failures"] == "0"
    assert "QEMU Launch JSONL Parity Audit" in (output_dir / "launch_jsonl_parity.md").read_text(encoding="utf-8")
