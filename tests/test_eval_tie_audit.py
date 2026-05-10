#!/usr/bin/env python3
"""Host-side checks for eval top-choice tie audits."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_tie_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_tie_audit_passes_clean_paired_scores() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(holyc, [{"id": "a", "scores": [3.0, 1.0]}, {"id": "b", "scores": [0.0, 2.0]}])
        write_jsonl(llama, [{"id": "a", "scores": [3.1, 1.0]}, {"id": "b", "scores": [0.0, 2.1]}])

        status = eval_tie_audit.main(
            [
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "ties",
                "--min-records",
                "2",
                "--max-top-index-disagreement-rate",
                "0",
            ]
        )

        assert status == 0
        payload = json.loads((tmp_path / "ties.json").read_text(encoding="utf-8"))
        assert payload["status"] == "pass"
        assert payload["summary"]["shared_records"] == 2
        assert payload["engine_summaries"][0]["top_tie_count"] == 0
        assert "No eval tie findings." in (tmp_path / "ties.md").read_text(encoding="utf-8")
        assert (tmp_path / "ties_records.csv").read_text(encoding="utf-8").startswith("source,")
        junit = ET.parse(tmp_path / "ties_junit.xml").getroot()
        assert junit.attrib["name"] == "holyc_eval_tie_audit"
        assert junit.attrib["failures"] == "0"


def test_tie_audit_flags_ties_missing_pairs_and_disagreements() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(holyc, [{"id": "a", "scores": [1.0, 1.0, 0.0]}, {"id": "b", "scores": [0.0, 4.0]}])
        write_jsonl(llama, [{"id": "a", "scores": [0.0, 2.0, 1.0]}, {"id": "c", "scores": [3.0, 0.0]}])

        status = eval_tie_audit.main(
            [
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "ties",
                "--max-top-tie-rate",
                "0",
                "--max-top-index-disagreement-rate",
                "0",
            ]
        )

        assert status == 1
        payload = json.loads((tmp_path / "ties.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in payload["findings"]}
        assert {"top_tie_rate", "missing_pair", "top_index_disagreement_rate"} <= metrics
        assert "top tie rate" in (tmp_path / "ties_junit.xml").read_text(encoding="utf-8")


def test_tie_audit_rejects_bad_thresholds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        status = eval_tie_audit.main(
            [
                "--holyc",
                str(ROOT / "bench" / "eval" / "samples" / "holyc_smoke_scored_predictions.jsonl"),
                "--llama",
                str(ROOT / "bench" / "eval" / "samples" / "llama_smoke_scored_predictions.jsonl"),
                "--output-dir",
                tmp,
                "--epsilon",
                "-1",
            ]
        )
        assert status == 2


def main() -> int:
    test_tie_audit_passes_clean_paired_scores()
    test_tie_audit_flags_ties_missing_pairs_and_disagreements()
    test_tie_audit_rejects_bad_thresholds()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
