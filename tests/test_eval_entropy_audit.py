#!/usr/bin/env python3
"""Host-side checks for eval entropy audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
AUDIT_PATH = BENCH_PATH / "eval_entropy_audit.py"
spec = importlib.util.spec_from_file_location("eval_entropy_audit", AUDIT_PATH)
eval_entropy_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_entropy_audit"] = eval_entropy_audit
spec.loader.exec_module(eval_entropy_audit)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_entropy_audit_passes_paired_scored_predictions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(
            holyc,
            [
                {"id": "a", "scores": [4.0, 1.0, 0.0, -1.0]},
                {"id": "b", "scores": [0.5, 2.0, 0.0, -0.5]},
            ],
        )
        write_jsonl(
            llama,
            [
                {"id": "a", "scores": [4.1, 1.1, 0.0, -1.0]},
                {"id": "b", "scores": [0.4, 2.1, 0.0, -0.4]},
            ],
        )

        assert eval_entropy_audit.main(
            [
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "audit",
                "--min-records",
                "2",
                "--fail-on-findings",
            ]
        ) == 0

        payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        assert payload["status"] == "pass"
        assert payload["summaries"][0]["engine"] == "holyc"
        assert payload["summaries"][0]["scored_count"] == 2
        assert payload["records"][0]["top_probability"] > 0.9
        assert payload["findings"] == []
        assert (tmp_path / "audit_records.csv").read_text(encoding="utf-8").startswith("source,")
        assert "No entropy findings" in (tmp_path / "audit.md").read_text(encoding="utf-8")
        junit = ET.parse(tmp_path / "audit_junit.xml").getroot()
        assert junit.attrib["name"] == "holyc_eval_entropy_audit"
        assert junit.attrib["failures"] == "0"


def test_entropy_audit_fails_collapse_uniformity_and_pairing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(
            holyc,
            [
                {"id": "a", "scores": [50.0, 0.0, 0.0, 0.0]},
                {"id": "b", "scores": [0.0, 0.0, 0.0, 0.0]},
            ],
        )
        write_jsonl(
            llama,
            [
                {"id": "a", "scores": [1.0, 0.9, 0.8, 0.7]},
                {"id": "c", "scores": [1.0, 0.0, 0.0, 0.0]},
            ],
        )

        assert eval_entropy_audit.main(
            [
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "audit",
                "--min-records",
                "2",
                "--min-normalized-entropy",
                "0.05",
                "--max-normalized-entropy",
                "0.98",
                "--max-record-entropy-delta",
                "0.2",
                "--fail-on-findings",
            ]
        ) == 2

        payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in payload["findings"]}
        assert payload["status"] == "fail"
        assert "normalized_entropy_low" in metrics
        assert "normalized_entropy_high" in metrics
        assert "missing_llama" in metrics
        assert "missing_holyc" in metrics
        assert "record_entropy_delta" in metrics
        assert "entropy finding" in (tmp_path / "audit_junit.xml").read_text(encoding="utf-8")


def test_entropy_audit_rejects_invalid_scores() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(holyc, [{"id": "a", "scores": [1.0, "bad"]}])
        write_jsonl(llama, [{"id": "a", "scores": [1.0, 0.0]}])

        assert eval_entropy_audit.main(
            [
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--output-dir",
                str(tmp_path),
                "--fail-on-findings",
            ]
        ) == 2

        payload = json.loads((tmp_path / "eval_entropy_audit_latest.json").read_text(encoding="utf-8"))
        assert payload["findings"][0]["metric"] == "scores"
