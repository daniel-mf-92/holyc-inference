from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_score_parity_audit


GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def scored_rows() -> list[dict[str, object]]:
    return [
        {"id": "smoke-hellaswag-1", "scores": [5.0, 1.0, 0.5, 0.25]},
        {"id": "smoke-arc-1", "scores": [4.0, 1.0, 0.0, -1.0]},
        {"id": "smoke-truthfulqa-1", "scores": [3.0, 0.0, -1.0, -2.0]},
    ]


def test_build_report_accepts_paired_score_vectors(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, scored_rows())
    write_jsonl(llama, scored_rows())

    args = eval_score_parity_audit.build_parser().parse_args(
        [
            "--gold",
            str(GOLD),
            "--holyc",
            str(holyc),
            "--llama",
            str(llama),
            "--dataset",
            "smoke-eval",
            "--split",
            "validation",
            "--require-scores",
            "--min-score-parity-pct",
            "100",
            "--max-top-score-tie-pct",
            "0",
        ]
    )
    report = eval_score_parity_audit.build_report(args)

    assert report["status"] == "pass"
    assert report["summary"]["paired_rows"] == 3
    assert report["summary"]["paired_scored_rows"] == 3
    assert report["summary"]["score_shape_match_rows"] == 3
    assert report["summary"]["holyc_top_score_tie_rows"] == 0
    assert report["summary"]["llama_top_score_tie_rows"] == 0


def test_build_report_rejects_score_presence_mismatch_and_missing_rows(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(
        holyc,
        [
            {"id": "smoke-hellaswag-1", "scores": [5.0, 1.0, 0.5, 0.25]},
            {"id": "smoke-arc-1", "prediction": 0},
            {"id": "unknown-extra", "scores": [1.0, 0.0, 0.0, 0.0]},
        ],
    )
    write_jsonl(
        llama,
        [
            {"id": "smoke-hellaswag-1", "scores": [5.0, 1.0, 0.5, 0.25]},
            {"id": "smoke-arc-1", "scores": [4.0, 1.0, 0.0, -1.0]},
        ],
    )

    args = eval_score_parity_audit.build_parser().parse_args(
        [
            "--gold",
            str(GOLD),
            "--holyc",
            str(holyc),
            "--llama",
            str(llama),
            "--dataset",
            "smoke-eval",
            "--split",
            "validation",
            "--require-scores",
            "--min-paired-rows",
            "3",
        ]
    )
    report = eval_score_parity_audit.build_report(args)
    kinds = {finding["kind"] for finding in report["findings"]}

    assert report["status"] == "fail"
    assert {"missing_scores", "score_presence_mismatch", "missing_id", "extra_id", "min_paired_rows"} <= kinds


def test_build_report_rejects_top_score_tie_collapse(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(
        holyc,
        [
            {"id": "smoke-hellaswag-1", "scores": [5.0, 5.0, 0.5, 0.25]},
            {"id": "smoke-arc-1", "scores": [4.0, 1.0, 0.0, -1.0]},
            {"id": "smoke-truthfulqa-1", "scores": [3.0, 0.0, -1.0, -2.0]},
        ],
    )
    write_jsonl(llama, scored_rows())

    args = eval_score_parity_audit.build_parser().parse_args(
        [
            "--gold",
            str(GOLD),
            "--holyc",
            str(holyc),
            "--llama",
            str(llama),
            "--dataset",
            "smoke-eval",
            "--split",
            "validation",
            "--require-scores",
            "--max-top-score-tie-pct",
            "0",
        ]
    )
    report = eval_score_parity_audit.build_report(args)
    kinds = {finding["kind"] for finding in report["findings"]}

    assert report["status"] == "fail"
    assert report["summary"]["holyc_top_score_tie_rows"] == 1
    assert report["summary"]["holyc_top_score_tie_pct"] == 100.0 / 3.0
    assert "max_top_score_tie_pct" in kinds


def test_cli_writes_reports(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(holyc, scored_rows())
    write_jsonl(llama, scored_rows())

    status = eval_score_parity_audit.main(
        [
            "--gold",
            str(GOLD),
            "--holyc",
            str(holyc),
            "--llama",
            str(llama),
            "--dataset",
            "smoke-eval",
            "--split",
            "validation",
            "--require-scores",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "score_parity",
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "score_parity.json").read_text(encoding="utf-8"))
    junit = ET.parse(output_dir / "score_parity_junit.xml").getroot()
    assert report["status"] == "pass"
    assert (output_dir / "score_parity_pairs.csv").read_text(encoding="utf-8").startswith("record_id,")
    assert "HolyC top-score tie %" in (output_dir / "score_parity.md").read_text(encoding="utf-8")
    assert junit.attrib["failures"] == "0"
