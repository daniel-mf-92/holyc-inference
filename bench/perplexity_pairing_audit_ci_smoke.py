#!/usr/bin/env python3
"""CI smoke for perplexity_pairing_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import perplexity_pairing_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        holyc = root / "holyc.jsonl"
        llama = root / "llama.jsonl"
        write_jsonl(
            holyc,
            [
                {"id": "a", "token_logprobs": [-0.1, -0.2], "metadata": {"dataset": "arc", "split": "validation"}},
                {"id": "b", "token_count": 3, "total_nll": 0.9, "dataset": "hellaswag", "split": "validation"},
            ],
        )
        write_jsonl(
            llama,
            [
                {"id": "a", "token_logprobs": [-0.1, -0.2], "metadata": {"dataset": "arc", "split": "validation"}},
                {"id": "b", "token_count": 3, "total_nll": 0.8, "dataset": "hellaswag", "split": "validation"},
            ],
        )
        out = root / "out"
        status = perplexity_pairing_audit.main(
            ["--holyc", str(holyc), "--llama", str(llama), "--output-dir", str(out), "--output-stem", "pairing", "--min-pairs", "2"]
        )
        if status != 0:
            return status
        payload = json.loads((out / "pairing.json").read_text(encoding="utf-8"))
        if payload["summary"]["paired_rows"] != 2 or payload["status"] != "pass":
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        holyc = root / "bad_holyc.jsonl"
        llama = root / "bad_llama.jsonl"
        write_jsonl(
            holyc,
            [
                {"id": "a", "token_count": 2, "total_nll": 0.7, "dataset": "arc", "split": "validation"},
                {"id": "a", "token_count": 2, "total_nll": 0.6, "dataset": "arc", "split": "validation"},
                {"id": "only-holyc", "token_count": 1, "total_nll": 0.1},
            ],
        )
        write_jsonl(llama, [{"id": "a", "token_count": 3, "total_nll": 0.9, "dataset": "arc", "split": "test"}])
        out = root / "out"
        status = perplexity_pairing_audit.main(["--holyc", str(holyc), "--llama", str(llama), "--output-dir", str(out), "--output-stem", "pairing"])
        if status == 0:
            return 1
        payload = json.loads((out / "pairing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in payload["findings"]}
        required = {"duplicate_record_id", "missing_llama_record", "token_count_mismatch", "split_mismatch"}
        if not required <= kinds:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
