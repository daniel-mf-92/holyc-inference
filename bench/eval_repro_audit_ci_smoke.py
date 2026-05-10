#!/usr/bin/env python3
"""CI smoke gate for eval reproducibility metadata audits."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_repro_audit


def write_predictions(path: Path, *, seed: int = 1234, temperature: float = 0.0) -> None:
    rows = [
        {
            "record_id": "smoke-1",
            "predicted_index": 0,
            "metadata": {
                "seed": seed,
                "temperature": temperature,
                "top_k": 1,
                "top_p": 1.0,
                "max_tokens": 16,
            },
        },
        {
            "record_id": "smoke-2",
            "predicted_index": 1,
            "metadata": {
                "seed": seed,
                "temperature": temperature,
                "top_k": 1,
                "top_p": 1.0,
                "max_tokens": 16,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_predictions(holyc)
        write_predictions(llama)
        return eval_repro_audit.main(
            [
                str(holyc),
                str(llama),
                "--output-dir",
                "bench/results",
                "--output-stem",
                "eval_repro_audit_smoke_latest",
                "--require-metadata",
                "--require-deterministic",
                "--expect",
                "seed=1234",
                "--expect",
                "temperature=0",
                "--expect",
                "top_k=1",
                "--expect",
                "top_p=1",
                "--expect",
                "max_tokens=16",
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())
