#!/usr/bin/env python3
"""Tests for eval identity audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_identity_audit


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def row(record_id: str, **metadata: object) -> dict[str, object]:
    full_metadata: dict[str, object] = {
        "model": "smoke-model",
        "model_sha256": "abc",
        "tokenizer_sha256": "tok",
        "quantization": "Q4_0",
        "prompt_template_sha256": "prompt",
    }
    full_metadata.update(metadata)
    return {"id": record_id, "scores": [1.0, 0.0], "metadata": full_metadata}


def test_summarize_accepts_consistent_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.jsonl"
    write_predictions(
        artifact,
        [
            row("a", model_sha256="abc", tokenizer_sha256="tok", quantization="Q4_0"),
            row("b", model_sha256="abc", tokenizer_sha256="tok", quantization="Q4_0"),
        ],
    )

    summary = eval_identity_audit.summarize_artifact(artifact, ["model_sha256", "tokenizer_sha256", "quantization"])

    assert summary.rows == 2
    assert summary.metadata["model_sha256"] == ["abc"]
    assert summary.missing_counts["tokenizer_sha256"] == 0
    assert summary.inconsistent_keys == []


def test_evaluate_flags_missing_inconsistent_and_cross_engine_drift(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_predictions(
        holyc,
        [
            row("a", model_sha256="abc", tokenizer_sha256="tok", quantization="Q4_0"),
            row("b", model_sha256="def", tokenizer_sha256="", quantization="Q4_0"),
        ],
    )
    write_predictions(llama, [row("a", model_sha256="xyz", tokenizer_sha256="tok", quantization="Q4_0")])
    args = eval_identity_audit.build_parser().parse_args(
        [
            str(holyc),
            str(llama),
            "--require-identity",
            "--compare-key",
            "model_sha256",
        ]
    )
    artifacts = [eval_identity_audit.summarize_artifact(path, ["model_sha256", "tokenizer_sha256"]) for path in [holyc, llama]]

    findings = eval_identity_audit.evaluate(artifacts, args)
    gates = {finding.gate for finding in findings}

    assert {"inconsistent_identity", "missing_identity", "cross_engine_mismatch"} <= gates


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    rows = [
        row("a", model_sha256="abc", tokenizer_sha256="tok", quantization="Q4_0"),
        row("b", model_sha256="abc", tokenizer_sha256="tok", quantization="Q4_0"),
    ]
    write_predictions(holyc, rows)
    write_predictions(llama, rows)
    output_dir = tmp_path / "out"

    status = eval_identity_audit.main(
        [
            str(holyc),
            str(llama),
            "--require-identity",
            "--compare-key",
            "model_sha256",
            "--compare-key",
            "tokenizer_sha256",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "identity",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "identity.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["artifacts"] == 2
    assert "No identity findings." in (output_dir / "identity.md").read_text(encoding="utf-8")
    rows_csv = list(csv.DictReader((output_dir / "identity.csv").open(encoding="utf-8")))
    assert rows_csv[0]["model_sha256"] == "abc"
    finding_rows = list(csv.DictReader((output_dir / "identity_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "identity_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_eval_identity_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_summarize_accepts_consistent_identity(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_evaluate_flags_missing_inconsistent_and_cross_engine_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
