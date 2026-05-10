#!/usr/bin/env python3
"""CI smoke gate for eval quantization delta audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
ROOT = BENCH_DIR.parent

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
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        write_eval(safe_dir / "q4.json", quantization="Q4_0", holyc_accuracy=0.98, llama_accuracy=1.0, agreement=0.98)
        write_eval(safe_dir / "q8.json", quantization="Q8_0", holyc_accuracy=1.0, llama_accuracy=1.0, agreement=1.0)
        duplicate_dir = tmp_path / "duplicates"
        duplicate_dir.mkdir()
        write_eval(duplicate_dir / "q4_a.json", quantization="Q4_0", holyc_accuracy=0.98, llama_accuracy=1.0, agreement=0.98)
        write_eval(duplicate_dir / "q4_b.json", quantization="Q4_0", holyc_accuracy=0.97, llama_accuracy=1.0, agreement=0.97)
        write_eval(duplicate_dir / "q8.json", quantization="Q8_0", holyc_accuracy=1.0, llama_accuracy=1.0, agreement=1.0)

        duplicate_out = tmp_path / "duplicate_out"
        duplicate_run = subprocess.run(
            [
                sys.executable,
                str(BENCH_DIR / "eval_quant_delta_audit.py"),
                str(duplicate_dir),
                "--output-dir",
                str(duplicate_out),
                "--output-stem",
                "eval_quant_delta_audit_duplicate",
                "--min-records",
                "4",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if duplicate_run.returncode == 0:
            print("duplicate_quant_delta_not_rejected=true", file=sys.stderr)
            return 1
        duplicate_report = json.loads((duplicate_out / "eval_quant_delta_audit_duplicate.json").read_text(encoding="utf-8"))
        duplicate_gates = {finding["gate"] for finding in duplicate_report["findings"]}
        if "duplicate_report" not in duplicate_gates:
            print("duplicate_quant_delta_missing_finding=true", file=sys.stderr)
            return 1

        return eval_quant_delta_audit.main(
            [
                str(safe_dir),
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
