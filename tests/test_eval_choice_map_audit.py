#!/usr/bin/env python3
"""Tests for host-side eval choice-map audit."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import eval_choice_map_audit


def smoke_args(output_dir: Path, output_stem: str = "audit") -> list[str]:
    return [
        "--gold",
        str(BENCH / "datasets" / "samples" / "smoke_eval.jsonl"),
        "--holyc",
        str(BENCH / "eval" / "samples" / "holyc_smoke_predictions.jsonl"),
        "--llama",
        str(BENCH / "eval" / "samples" / "llama_smoke_predictions.jsonl"),
        "--dataset",
        "smoke-eval",
        "--split",
        "validation",
        "--output-dir",
        str(output_dir),
        "--output-stem",
        output_stem,
    ]


def test_smoke_choice_map_passes(tmp_path: Path) -> None:
    assert eval_choice_map_audit.main(smoke_args(tmp_path)) == 0

    report = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["gold"]["records"] == 3
    assert report["gold"]["choice_count_histogram"] == {"4": 3}
    assert report["summary"]["holyc"]["format_histogram"] == {"alpha": 1, "index": 1, "scores_only": 1}
    assert report["summary"]["llama.cpp"]["format_histogram"] == {"alpha": 1, "index": 1, "scores_only": 1}
    assert report["summary"]["holyc"]["valid_pct"] == 100.0
    assert len(report["rows"]) == 6

    rows = list(csv.DictReader((tmp_path / "audit.csv").open(newline="", encoding="utf-8")))
    assert {row["source"] for row in rows} == {"holyc", "llama.cpp"}
    assert (tmp_path / "audit_findings.csv").read_text(encoding="utf-8").startswith("severity,source,record_id")
    assert "Eval Choice Map Audit" in (tmp_path / "audit.md").read_text(encoding="utf-8")
    junit_root = ET.parse(tmp_path / "audit_junit.xml").getroot()
    assert junit_root.attrib["failures"] == "0"


def test_engine_format_parity_gate_fails(tmp_path: Path) -> None:
    llama = tmp_path / "llama_index_only.jsonl"
    llama.write_text(
        "\n".join(
            [
                '{"id":"smoke-hellaswag-1","prediction":0}',
                '{"id":"smoke-arc-1","prediction":0}',
                '{"id":"smoke-truthfulqa-1","prediction":0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = smoke_args(tmp_path, "parity")
    args[args.index(str(BENCH / "eval" / "samples" / "llama_smoke_predictions.jsonl"))] = str(llama)
    args.append("--require-engine-format-parity")

    assert eval_choice_map_audit.main(args) == 2
    report = json.loads((tmp_path / "parity.json").read_text(encoding="utf-8"))
    assert any(finding["kind"] == "engine_format_parity" for finding in report["findings"])


def test_mixed_format_gate_fails(tmp_path: Path) -> None:
    args = smoke_args(tmp_path, "mixed")
    args.append("--fail-mixed-formats")

    assert eval_choice_map_audit.main(args) == 2
    report = json.loads((tmp_path / "mixed.json").read_text(encoding="utf-8"))
    assert {finding["source"] for finding in report["findings"] if finding["kind"] == "mixed_formats"} == {
        "holyc",
        "llama.cpp",
    }


def test_unmapped_prediction_fails_with_record_detail(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc_bad.jsonl"
    holyc.write_text(
        "\n".join(
            [
                '{"id":"smoke-hellaswag-1","prediction":"not a choice"}',
                '{"id":"smoke-arc-1","prediction":"A"}',
                '{"id":"smoke-truthfulqa-1","prediction":"A"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = smoke_args(tmp_path, "bad")
    args[args.index(str(BENCH / "eval" / "samples" / "holyc_smoke_predictions.jsonl"))] = str(holyc)

    assert eval_choice_map_audit.main(args) == 2
    report = json.loads((tmp_path / "bad.json").read_text(encoding="utf-8"))
    assert any(
        finding["record_id"] == "smoke-hellaswag-1" and finding["kind"] == "unmapped"
        for finding in report["findings"]
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_smoke_choice_map_passes(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_engine_format_parity_gate_fails(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_mixed_format_gate_fails(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_unmapped_prediction_fails_with_record_detail(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
