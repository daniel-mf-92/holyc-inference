#!/usr/bin/env python3
"""Tests for QEMU benchmark seed metadata audits."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_seed_audit


def row(prompt: str = "smoke-short", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "iteration": 1,
        "commit": "abc1234",
        "seed": 42,
        "exit_class": "ok",
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_seed_audit.build_parser().parse_args(extra)


def test_audit_accepts_seeded_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row("smoke-short"), row("smoke-code", iteration=2, seed="", rng_seed=99)])
    args = parse_args([str(artifact), "--min-rows", "2"])

    rows, findings = qemu_seed_audit.audit([artifact], args)

    assert findings == []
    assert [item.seed for item in rows] == [42, 99]


def test_audit_flags_missing_invalid_negative_and_drifted_seeds(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row("same", seed=1),
            row("same", seed=2),
            row("missing", seed=""),
            row("bad", seed="1.5"),
            row("negative", seed=-1),
        ],
    )
    args = parse_args([str(artifact)])

    _, findings = qemu_seed_audit.audit([artifact], args)
    kinds = {finding.kind for finding in findings}

    assert {"seed_drift", "missing_seed", "invalid_seed", "negative_seed"} <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row()])
    output_dir = tmp_path / "out"

    status = qemu_seed_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "seed"])

    assert status == 0
    payload = json.loads((output_dir / "seed.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["seeded_rows"] == 1
    assert "No seed metadata findings." in (output_dir / "seed.md").read_text(encoding="utf-8")
    assert "seed_source" in (output_dir / "seed.csv").read_text(encoding="utf-8")
    assert "kind" in (output_dir / "seed_findings.csv").read_text(encoding="utf-8")
    root = ET.parse(output_dir / "seed_junit.xml").getroot()
    assert root.attrib["name"] == "holyc_qemu_seed_audit"
    assert root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_seeded_rows(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_invalid_negative_and_drifted_seeds(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
