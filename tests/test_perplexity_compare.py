#!/usr/bin/env python3
"""Host-side checks for the offline HolyC vs llama.cpp perplexity comparator."""

from __future__ import annotations

import importlib.util
import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
COMPARE_PATH = BENCH_PATH / "perplexity_compare.py"
spec = importlib.util.spec_from_file_location("perplexity_compare", COMPARE_PATH)
perplexity_compare = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["perplexity_compare"] = perplexity_compare
spec.loader.exec_module(perplexity_compare)


def test_smoke_logprobs_compare_cleanly() -> None:
    holyc = BENCH_PATH / "eval" / "samples" / "holyc_smoke_logprobs.jsonl"
    llama = BENCH_PATH / "eval" / "samples" / "llama_smoke_logprobs.jsonl"

    holyc_records = perplexity_compare.load_records(holyc)
    llama_records = perplexity_compare.load_records(llama)
    rows, summary = perplexity_compare.compare(holyc_records, llama_records)

    assert len(rows) == 3
    assert summary["record_count"] == 3
    assert summary["holyc"]["token_count"] == 11
    assert summary["llama"]["token_count"] == 11
    assert summary["token_count_mismatches"] == 0
    assert summary["holyc"]["perplexity"] > 1.0


def test_cli_writes_json_and_markdown_report() -> None:
    holyc = BENCH_PATH / "eval" / "samples" / "holyc_smoke_logprobs.jsonl"
    llama = BENCH_PATH / "eval" / "samples" / "llama_smoke_logprobs.jsonl"

    with tempfile.TemporaryDirectory() as tmp:
        assert (
            perplexity_compare.main(
                [
                    "--holyc",
                    str(holyc),
                    "--llama",
                    str(llama),
                    "--dataset",
                    "smoke-eval",
                    "--split",
                    "validation",
                    "--model",
                    "synthetic-smoke",
                    "--quantization",
                    "Q4_0",
                    "--output-dir",
                    tmp,
                    "--output-stem",
                    "ppl",
                ]
            )
            == 0
        )
        payload = json.loads((Path(tmp) / "ppl.json").read_text(encoding="utf-8"))
        csv_rows = list(csv.DictReader((Path(tmp) / "ppl.csv").open(newline="", encoding="utf-8")))
        junit_root = ET.parse(Path(tmp) / "ppl_junit.xml").getroot()
        assert payload["summary"]["record_count"] == 3
        assert payload["status"] == "pass"
        assert payload["regressions"] == []
        markdown = (Path(tmp) / "ppl.md").read_text(encoding="utf-8")
        assert "Perplexity Compare Report" in markdown
        assert "No quality gate regressions." in markdown
        assert len(csv_rows) == 3
        assert csv_rows[0]["record_id"] == "smoke-arc-1"
        assert "nll_delta_holyc_minus_llama" in csv_rows[0]
        assert junit_root.attrib["name"] == "holyc_perplexity_compare"
        assert junit_root.attrib["failures"] == "0"


def test_cli_can_fail_on_perplexity_quality_gate_regression(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    holyc.write_text(
        "\n".join(
            [
                '{"id":"one","token_count":2,"mean_nll":1.25}',
                '{"id":"two","token_count":2,"mean_nll":1.00}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    llama.write_text(
        "\n".join(
            [
                '{"id":"one","token_count":2,"mean_nll":0.50}',
                '{"id":"two","token_count":2,"mean_nll":0.50}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status = perplexity_compare.main(
        [
            "--holyc",
            str(holyc),
            "--llama",
            str(llama),
            "--output-dir",
            str(tmp_path),
            "--output-stem",
            "gated",
            "--max-nll-delta",
            "0.1",
            "--max-perplexity-ratio",
            "1.1",
            "--max-record-nll-delta",
            "0.2",
            "--fail-on-regression",
        ]
    )
    payload = json.loads((tmp_path / "gated.json").read_text(encoding="utf-8"))
    junit_root = ET.parse(tmp_path / "gated_junit.xml").getroot()

    assert status == 1
    assert payload["status"] == "fail"
    assert {row["metric"] for row in payload["regressions"]} == {
        "max_abs_record_nll_delta",
        "nll_delta_holyc_minus_llama",
        "perplexity_ratio_holyc_over_llama",
    }
    assert junit_root.attrib["failures"] == "3"
    assert junit_root.find("./testcase/failure") is not None


def test_token_count_mismatch_fails_by_default(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    holyc.write_text('{"id":"one","token_logprobs":[-0.1,-0.2]}\n', encoding="utf-8")
    llama.write_text('{"id":"one","token_logprobs":[-0.1]}\n', encoding="utf-8")

    assert perplexity_compare.main(["--holyc", str(holyc), "--llama", str(llama), "--output-dir", str(tmp_path)]) == 2
    assert (
        perplexity_compare.main(
            [
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--allow-token-count-mismatch",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )


if __name__ == "__main__":
    test_smoke_logprobs_compare_cleanly()
    test_cli_writes_json_and_markdown_report()
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_can_fail_on_perplexity_quality_gate_regression(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_token_count_mismatch_fails_by_default(Path(tmp))
    print("perplexity_compare_tests=ok")
