#!/usr/bin/env python3
"""Host-side checks for the offline HolyC vs llama.cpp perplexity comparator."""

from __future__ import annotations

import importlib.util
import csv
import json
import sys
import tempfile
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
        assert payload["summary"]["record_count"] == 3
        assert "Perplexity Compare Report" in (Path(tmp) / "ppl.md").read_text(encoding="utf-8")
        assert len(csv_rows) == 3
        assert csv_rows[0]["record_id"] == "smoke-arc-1"
        assert "nll_delta_holyc_minus_llama" in csv_rows[0]


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
        test_token_count_mismatch_fails_by_default(Path(tmp))
    print("perplexity_compare_tests=ok")
