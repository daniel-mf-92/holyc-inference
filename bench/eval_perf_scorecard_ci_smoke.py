#!/usr/bin/env python3
"""CI smoke gate for eval/perf scorecards."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_perf_scorecard


def write_eval(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "regressions": [],
                "summary": {
                    "record_count": 3,
                    "holyc_accuracy": 1.0,
                    "llama_accuracy": 1.0,
                    "agreement": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_bench(path: Path) -> None:
    rows = []
    for prompt, tok_per_s, wall_tok_per_s in (
        ("smoke-short", 160.0, 120.0),
        ("smoke-code", 150.0, 115.0),
    ):
        rows.append(
            {
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "profile": "ci-airgap-smoke",
                "phase": "measured",
                "prompt": prompt,
                "exit_class": "ok",
                "timed_out": False,
                "failure_reason": None,
                "tok_per_s": tok_per_s,
                "wall_tok_per_s": wall_tok_per_s,
                "memory_bytes": 67174400,
                "command": ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio", "-display", "none"],
                "command_airgap_ok": True,
                "command_airgap_violations": [],
            }
        )
    path.write_text(json.dumps({"benchmarks": rows}, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        eval_path = tmp_path / "eval_compare_smoke.json"
        bench_path = tmp_path / "qemu_prompt_bench_smoke.json"
        write_eval(eval_path)
        write_bench(bench_path)
        status = eval_perf_scorecard.main(
            [
                "--eval",
                str(eval_path),
                "--bench",
                str(bench_path),
                "--output-dir",
                "bench/results",
                "--output-stem",
                "eval_perf_scorecard_smoke_latest",
                "--require-perf-match",
                "--fail-on-failed-eval",
                "--fail-on-regressions",
                "--min-records",
                "3",
                "--min-holyc-accuracy",
                "0.95",
                "--min-agreement",
                "0.95",
                "--min-prompts",
                "2",
                "--min-ok-runs",
                "2",
                "--min-tok-per-s",
                "100",
                "--min-wall-tok-per-s",
                "100",
                "--max-memory-bytes",
                "80000000",
            ]
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
