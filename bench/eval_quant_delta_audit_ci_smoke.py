#!/usr/bin/env python3
"""CI smoke gate for eval quantization delta audits."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_quant_delta_audit


def write_eval(path: Path, *, quantization: str, holyc_accuracy: float, llama_accuracy: float, agreement: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": quantization,
                "regressions": [],
                "summary": {
                    "record_count": 4,
                    "holyc_accuracy": holyc_accuracy,
                    "llama_accuracy": llama_accuracy,
                    "agreement": agreement,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_eval(tmp_path / "q4.json", quantization="Q4_0", holyc_accuracy=0.98, llama_accuracy=1.0, agreement=0.98)
        write_eval(tmp_path / "q8.json", quantization="Q8_0", holyc_accuracy=1.0, llama_accuracy=1.0, agreement=1.0)
        return eval_quant_delta_audit.main(
            [
                str(tmp_path),
                "--output-dir",
                "bench/results",
                "--output-stem",
                "eval_quant_delta_audit_smoke_latest",
                "--fail-on-failed-eval",
                "--fail-on-regressions",
                "--min-records",
                "4",
                "--max-holyc-accuracy-drop",
                "0.05",
                "--max-llama-accuracy-drop",
                "0.01",
                "--max-agreement-delta",
                "0.05",
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())
