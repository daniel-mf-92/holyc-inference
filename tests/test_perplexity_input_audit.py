#!/usr/bin/env python3
"""Tests for perplexity input artifact audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import perplexity_input_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return perplexity_input_audit.build_parser().parse_args(extra)


def test_audit_accepts_logprob_and_nll_records(tmp_path: Path) -> None:
    artifact = tmp_path / "ppl.jsonl"
    write_jsonl(
        artifact,
        [
            {"id": "one", "dataset": "arc", "split": "validation", "token_logprobs": [-0.1, -0.2]},
            {"id": "two", "dataset": "arc", "split": "validation", "token_count": 3, "mean_nll": 0.5, "perplexity": 1.6487212707001282},
        ],
    )
    args = parse_args([str(artifact), "--require-dataset", "--require-split", "--min-records", "2", "--min-tokens", "5"])

    records, summaries, findings = perplexity_input_audit.audit([artifact], args)

    assert findings == []
    assert len(records) == 2
    assert summaries[0].token_count == 5
    assert summaries[0].datasets == "arc"
    assert records[0].derived_total_nll == 0.30000000000000004


def test_audit_flags_duplicate_ids_missing_metadata_and_metric_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "ppl.jsonl"
    write_jsonl(
        artifact,
        [
            {"id": "one", "token_count": 2, "total_nll": 1.0, "mean_nll": 0.1},
            {"id": "one", "token_count": 1, "token_logprobs": [0.2]},
            {"id": "three", "token_count": 2, "mean_nll": 0.5, "perplexity": 9.0},
        ],
    )
    args = parse_args([str(artifact), "--require-dataset", "--require-split"])

    records, summaries, findings = perplexity_input_audit.audit([artifact], args)

    assert len(records) == 3
    assert summaries[0].duplicate_ids == 1
    kinds = {finding.kind for finding in findings}
    assert {"duplicate_record_id", "missing_dataset", "missing_split", "mean_nll_drift", "positive_logprob", "perplexity_drift"} <= kinds


def test_cli_writes_json_markdown_csv_sources_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "ppl.jsonl"
    write_jsonl(artifact, [{"id": "one", "dataset": "smoke", "split": "validation", "token_count": 2, "mean_nll": 0.5}])
    output_dir = tmp_path / "out"

    status = perplexity_input_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "input_audit",
            "--require-dataset",
            "--require-split",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "input_audit.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["records"] == 1
    assert "No perplexity input findings." in (output_dir / "input_audit.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "input_audit.csv").open(encoding="utf-8")))
    assert rows[0]["record_id"] == "one"
    source_rows = list(csv.DictReader((output_dir / "input_audit_sources.csv").open(encoding="utf-8")))
    assert source_rows[0]["valid_rows"] == "1"
    finding_rows = list(csv.DictReader((output_dir / "input_audit_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "input_audit_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_perplexity_input_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_on_minimum_token_gate(tmp_path: Path) -> None:
    artifact = tmp_path / "ppl.jsonl"
    write_jsonl(artifact, [{"id": "one", "token_count": 2, "mean_nll": 0.5}])
    output_dir = tmp_path / "out"

    status = perplexity_input_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "input_audit",
            "--min-tokens-per-source",
            "3",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "input_audit.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_tokens_per_source"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_logprob_and_nll_records(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_duplicate_ids_missing_metadata_and_metric_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_sources_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_on_minimum_token_gate(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
