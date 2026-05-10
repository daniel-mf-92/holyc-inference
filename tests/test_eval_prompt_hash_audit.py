from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_compare
import eval_prompt_hash_audit


GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def hashed_rows() -> list[dict[str, object]]:
    gold = eval_compare.load_gold(GOLD, "smoke-eval", "validation")
    rows: list[dict[str, object]] = []
    for record_id, case in gold.items():
        hashes = eval_prompt_hash_audit.expected_hashes(case)
        rows.append(
            {
                "id": record_id,
                "prediction": 0,
                "prompt_sha256": hashes.prompt_sha256,
                "choices_sha256": hashes.choices_sha256,
                "input_sha256": hashes.input_sha256,
            }
        )
    return rows


def parse_args(extra: list[str], holyc: Path, llama: Path) -> object:
    return eval_prompt_hash_audit.build_parser().parse_args(
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
            *extra,
        ]
    )


def test_build_report_accepts_matching_prediction_hashes(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(holyc, hashed_rows())
    write_jsonl(llama, hashed_rows())

    report = eval_prompt_hash_audit.build_report(
        parse_args(["--require-hashes", "--min-hashed-rows", "6"], holyc, llama)
    )

    assert report["status"] == "pass"
    assert report["summary"]["hashed_prediction_rows"] == 6
    assert report["summary"]["matched_hash_fields"] == 18
    assert report["findings"] == []


def test_build_report_flags_missing_and_mismatched_hashes(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    bad = hashed_rows()
    bad[0]["prompt_sha256"] = "0" * 64
    write_jsonl(holyc, bad)
    write_jsonl(llama, [{"id": "smoke-hellaswag-1", "prediction": 0}, {"id": "extra", "prediction": 1}])

    report = eval_prompt_hash_audit.build_report(parse_args(["--require-hashes"], holyc, llama))
    kinds = {finding["kind"] for finding in report["findings"]}

    assert report["status"] == "fail"
    assert "prompt_sha256_mismatch" in kinds
    assert "missing_prompt_sha256" in kinds
    assert "missing_prediction" in kinds
    assert "extra_prediction" in kinds


def test_cli_writes_report_artifacts(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(holyc, hashed_rows())
    write_jsonl(llama, hashed_rows())

    status = eval_prompt_hash_audit.main(
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
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "prompt_hash",
            "--require-hashes",
            "--fail-on-findings",
        ]
    )

    payload = json.loads((output_dir / "prompt_hash.json").read_text(encoding="utf-8"))
    rows_csv = list(csv.DictReader((output_dir / "prompt_hash.csv").open(encoding="utf-8")))
    findings_csv = list(csv.DictReader((output_dir / "prompt_hash_findings.csv").open(encoding="utf-8")))
    junit_root = ET.parse(output_dir / "prompt_hash_junit.xml").getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert len(rows_csv) == 6
    assert findings_csv == []
    assert "Eval Prompt Hash Audit" in (output_dir / "prompt_hash.md").read_text(encoding="utf-8")
    assert junit_root.attrib["name"] == "holyc_eval_prompt_hash_audit"
    assert junit_root.attrib["failures"] == "0"
