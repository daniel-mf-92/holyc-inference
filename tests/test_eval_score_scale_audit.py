#!/usr/bin/env python3
"""Host-side checks for eval score-scale audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
AUDIT_PATH = BENCH_PATH / "eval_score_scale_audit.py"
spec = importlib.util.spec_from_file_location("eval_score_scale_audit", AUDIT_PATH)
eval_score_scale_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_score_scale_audit"] = eval_score_scale_audit
spec.loader.exec_module(eval_score_scale_audit)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_score_scale_audit_passes_paired_finite_scores() -> None:
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

        assert eval_score_scale_audit.main(
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
        assert payload["summaries"][0]["record_count"] == 2
        assert payload["findings"] == []
        assert (tmp_path / "audit_records.csv").read_text(encoding="utf-8").startswith("source,")
        assert "No score-scale findings" in (tmp_path / "audit.md").read_text(encoding="utf-8")
        junit = ET.parse(tmp_path / "audit_junit.xml").getroot()
        assert junit.attrib["name"] == "holyc_eval_score_scale_audit"
        assert junit.attrib["failures"] == "0"


def test_score_scale_audit_fails_constants_pairing_and_scale_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(
            holyc,
            [
                {"id": "a", "scores": [1.0, 1.0, 1.0, 1.0]},
                {"id": "b", "scores": [1000.0, 900.0, 800.0, 700.0]},
            ],
        )
        write_jsonl(
            llama,
            [
                {"id": "a", "scores": [4.0, 1.0, 0.0, -1.0]},
                {"id": "c", "scores": [0.5, 2.0, 0.0, -0.5]},
            ],
        )

        assert eval_score_scale_audit.main(
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
                "--min-score-span",
                "0.01",
                "--max-mean-abs-score-ratio",
                "5",
                "--fail-on-findings",
            ]
        ) == 2

        payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in payload["findings"]}
        assert payload["status"] == "fail"
        assert "score_span" in metrics
        assert "missing_llama" in metrics
        assert "missing_holyc" in metrics
        assert "mean_abs_score_ratio" in metrics
        assert "score-scale finding" in (tmp_path / "audit_junit.xml").read_text(encoding="utf-8")


def test_score_scale_audit_rejects_invalid_scores() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(holyc, [{"id": "a", "scores": [1.0, "nan"]}])
        write_jsonl(llama, [{"id": "a", "scores": [1.0, 0.0]}])

        assert eval_score_scale_audit.main(
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

        payload = json.loads((tmp_path / "eval_score_scale_audit_latest.json").read_text(encoding="utf-8"))
        assert payload["findings"][0]["metric"] == "scores"
