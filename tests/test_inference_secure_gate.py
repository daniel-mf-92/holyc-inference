#!/usr/bin/env python3
"""Harness for IQ-1258 secure-local release gate script."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

GATE_SCRIPT = Path("automation/inference-secure-gate.sh")

REQUIRED_WS_IDS = {
    "WS16-03",
    "WS16-04",
    "WS16-05",
    "WS9-02",
    "WS9-08",
    "WS9-17",
    "WS9-18",
    "WS9-22",
    "WS16-08",
}

REQUIRED_ARTIFACTS = {
    "src/model/trust_manifest.HC": ["ModelTrustManifestVerifySHA256Checked"],
    "src/model/eval_gate.HC": ["ModelEvalPromotionGateChecked"],
    "src/gguf/hardening_gate.HC": ["GGUFParserHardeningGateChecked"],
    "src/gpu/policy.HC": ["GPU_POLICY_ERR_IOMMU_GUARD", "GPUPolicyAllowDispatchChecked"],
    "src/gpu/book_of_truth_bridge.HC": ["BOTGPUBridgeRecordMMIOWrite", "BOT_GPU_DMA_UNMAP"],
    "src/gpu/command_verify.HC": ["GPUCommandVerifyDescriptorChecked"],
}


def _run_gate(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(GATE_SCRIPT)],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def _parse_json_lines(stdout: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def test_source_contains_iq1258_contract() -> None:
    src = GATE_SCRIPT.read_text(encoding="utf-8")

    assert "inference-secure-local-release" in src
    assert 'check_contains "WS16-03" "src/model/trust_manifest.HC"' in src
    assert 'check_contains "WS16-04" "src/model/eval_gate.HC"' in src
    assert 'check_contains "WS16-05" "src/gguf/hardening_gate.HC"' in src
    assert 'check_contains "WS9-22" "src/gpu/policy.HC"' in src
    assert 'check_contains "WS9-18" "src/gpu/command_verify.HC"' in src


def test_gate_emits_machine_readable_output_on_repo() -> None:
    result = _run_gate()
    records = _parse_json_lines(result.stdout)
    assert records

    check_records = [rec for rec in records if rec.get("type") == "check"]
    summary_records = [rec for rec in records if rec.get("type") == "summary"]

    assert check_records
    assert len(summary_records) == 1

    seen_ids = {str(rec.get("id")) for rec in check_records}
    assert REQUIRED_WS_IDS.issubset(seen_ids)

    summary = summary_records[0]
    assert summary.get("status") in ("pass", "fail")
    assert int(summary.get("passed", -1)) >= 1


def test_gate_passes_when_all_required_controls_exist(tmp_path: Path) -> None:
    fake_root = tmp_path / "fake-repo"
    for rel_file, markers in REQUIRED_ARTIFACTS.items():
        target = fake_root / rel_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(["// synthetic", *markers, ""]), encoding="utf-8")

    gate_copy = fake_root / "automation" / "inference-secure-gate.sh"
    gate_copy.parent.mkdir(parents=True, exist_ok=True)
    gate_copy.write_text(GATE_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")

    result = _run_gate(extra_env={"INFERENCE_GATE_ROOT": str(fake_root)})
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    records = _parse_json_lines(result.stdout)
    summary = [rec for rec in records if rec.get("type") == "summary"]
    assert len(summary) == 1
    assert summary[0].get("status") == "pass"
    assert int(summary[0].get("failed", -1)) == 0


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    test_source_contains_iq1258_contract()
    test_gate_emits_machine_readable_output_on_repo()
    with TemporaryDirectory() as tmp:
        test_gate_passes_when_all_required_controls_exist(Path(tmp))
    print("ok")
