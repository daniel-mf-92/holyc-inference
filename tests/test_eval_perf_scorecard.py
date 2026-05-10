#!/usr/bin/env python3
"""Tests for host-side eval/perf scorecards."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_perf_scorecard


def write_eval(path: Path, *, status: str = "pass", accuracy: float = 1.0, agreement: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "regressions": [] if status == "pass" else [{"metric": "accuracy"}],
                "summary": {
                    "record_count": 4,
                    "holyc_accuracy": accuracy,
                    "llama_accuracy": 1.0,
                    "agreement": agreement,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_bench(path: Path, *, tok_per_s: float = 160.0, wall_tok_per_s: float = 120.0) -> None:
    rows = []
    for prompt in ("short", "code"):
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
                "command": ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"],
                "command_airgap_ok": True,
                "command_airgap_violations": [],
            }
        )
    path.write_text(json.dumps({"benchmarks": rows}) + "\n", encoding="utf-8")


def test_scorecard_joins_eval_and_perf_by_model_quantization(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    bench_path = tmp_path / "bench.json"
    write_eval(eval_path)
    write_bench(bench_path)

    reports = [eval_perf_scorecard.load_eval_report(eval_path)]
    summaries = eval_perf_scorecard.load_perf_summaries(bench_path)
    rows = eval_perf_scorecard.build_scorecard(reports, summaries)

    assert len(rows) == 1
    assert rows[0].model == "synthetic-smoke"
    assert rows[0].prompt_count == 2
    assert rows[0].ok_runs == 2
    assert rows[0].min_tok_per_s == 160.0
    assert rows[0].max_memory_bytes == 67174400


def test_evaluate_flags_quality_perf_and_airgap_failures(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    bench_path = tmp_path / "bench.json"
    write_eval(eval_path, status="fail", accuracy=0.5, agreement=0.5)
    write_bench(bench_path, tok_per_s=25.0, wall_tok_per_s=20.0)
    payload = json.loads(bench_path.read_text(encoding="utf-8"))
    payload["benchmarks"][0]["command_airgap_ok"] = False
    payload["benchmarks"][0]["command_airgap_violations"] = ["missing -nic none"]
    bench_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    args = eval_perf_scorecard.parse_args(
        [
            "--eval",
            str(eval_path),
            "--bench",
            str(bench_path),
            "--min-holyc-accuracy",
            "0.9",
            "--min-agreement",
            "0.9",
            "--min-tok-per-s",
            "100",
            "--min-wall-tok-per-s",
            "100",
            "--min-ok-runs",
            "3",
            "--fail-on-failed-eval",
            "--fail-on-regressions",
            "--require-perf-match",
        ]
    )
    reports = [eval_perf_scorecard.load_eval_report(eval_path)]
    rows = eval_perf_scorecard.build_scorecard(reports, eval_perf_scorecard.load_perf_summaries(bench_path))

    findings = eval_perf_scorecard.evaluate(rows, reports, args)

    assert {
        "eval_status",
        "eval_regressions",
        "min_holyc_accuracy",
        "min_agreement",
        "min_ok_runs",
        "min_tok_per_s",
        "min_wall_tok_per_s",
        "airgap",
    }.issubset({finding.gate for finding in findings})


def test_cli_writes_scorecard_artifacts(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    bench_path = tmp_path / "bench.json"
    output_dir = tmp_path / "out"
    write_eval(eval_path)
    write_bench(bench_path)

    status = eval_perf_scorecard.main(
        [
            "--eval",
            str(eval_path),
            "--bench",
            str(bench_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "scorecard",
            "--require-perf-match",
            "--min-holyc-accuracy",
            "0.95",
            "--min-agreement",
            "0.95",
            "--min-tok-per-s",
            "100",
            "--min-wall-tok-per-s",
            "100",
            "--min-ok-runs",
            "2",
            "--min-prompts",
            "2",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["scorecard_rows"] == 1
    rows = list(csv.DictReader((output_dir / "scorecard.csv").open(encoding="utf-8")))
    assert rows[0]["model"] == "synthetic-smoke"
    markdown = (output_dir / "scorecard.md").read_text(encoding="utf-8")
    assert "Eval Perf Scorecard" in markdown
    junit_root = ET.parse(output_dir / "scorecard_junit.xml").getroot()
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_scorecard_joins_eval_and_perf_by_model_quantization(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_evaluate_flags_quality_perf_and_airgap_failures(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_scorecard_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
