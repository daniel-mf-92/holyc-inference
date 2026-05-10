#!/usr/bin/env python3
"""Tests for host-side eval reproducibility metadata audits."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_repro_audit


def write_predictions(path: Path, *, seed: int = 1234, temperature: float = 0.0, omit_key: str = "") -> None:
    rows = []
    for index in range(2):
        metadata = {
            "seed": seed,
            "temperature": temperature,
            "top_k": 1,
            "top_p": 1.0,
            "max_tokens": 16,
        }
        metadata.pop(omit_key, None)
        rows.append({"record_id": f"smoke-{index}", "predicted_index": index % 2, "metadata": metadata})
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_evaluate_passes_matching_deterministic_metadata(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_predictions(holyc)
    write_predictions(llama)
    args = eval_repro_audit.parse_args([str(holyc), str(llama), "--require-metadata", "--require-deterministic"])

    artifacts = [eval_repro_audit.summarize_artifact(path, eval_repro_audit.REPRO_KEYS) for path in [holyc, llama]]
    findings = eval_repro_audit.evaluate(artifacts, args)

    assert findings == []
    assert artifacts[0].metadata["seed"] == ["1234"]
    assert artifacts[0].metadata["temperature"] == ["0"]


def test_evaluate_flags_cross_engine_mismatch_and_missing_metadata(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_predictions(holyc, seed=1234)
    write_predictions(llama, seed=9999, omit_key="top_p")
    args = eval_repro_audit.parse_args([str(holyc), str(llama), "--require-metadata"])

    artifacts = [eval_repro_audit.summarize_artifact(path, eval_repro_audit.REPRO_KEYS) for path in [holyc, llama]]
    gates = {finding.gate for finding in eval_repro_audit.evaluate(artifacts, args)}

    assert {"cross_engine_mismatch", "missing_metadata"}.issubset(gates)


def test_cli_writes_repro_artifacts(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    output_dir = tmp_path / "out"
    write_predictions(holyc)
    write_predictions(llama)

    status = eval_repro_audit.main(
        [
            str(holyc),
            str(llama),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "repro",
            "--require-metadata",
            "--require-deterministic",
            "--expect",
            "seed=1234",
            "--expect",
            "temperature=0",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "repro.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["artifacts"] == 2
    assert "Eval Reproducibility Audit" in (output_dir / "repro.md").read_text(encoding="utf-8")
    assert (output_dir / "repro_findings.csv").exists()
    root = ET.parse(output_dir / "repro_junit.xml").getroot()
    assert root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_evaluate_passes_matching_deterministic_metadata(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_evaluate_flags_cross_engine_mismatch_and_missing_metadata(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_repro_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
