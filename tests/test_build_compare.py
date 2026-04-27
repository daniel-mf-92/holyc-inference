#!/usr/bin/env python3
"""Tests for host-side build benchmark comparison tooling."""

from __future__ import annotations

import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import build_compare


def write_report(path: Path, commit: str, tok_per_s: float, elapsed_us: int) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T10:00:00Z",
                "benchmarks": [
                    {
                        "commit": commit,
                        "benchmark": "qemu_prompt",
                        "profile": "secure-local",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "prompt": "smoke",
                        "tokens": 32,
                        "elapsed_us": elapsed_us,
                        "tok_per_s": tok_per_s,
                        "returncode": 0,
                        "timed_out": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_compare_builds_computes_tok_per_s_and_elapsed_deltas(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    write_report(baseline, "base", 100.0, 200000)
    write_report(candidate, "head", 125.0, 160000)

    metrics = build_compare.load_build_metrics([f"base={baseline}", f"head={candidate}"])
    deltas = build_compare.compare_builds(metrics, "base")

    assert len(metrics) == 2
    assert len(deltas) == 1
    assert deltas[0].candidate_build == "head"
    assert deltas[0].tok_per_s_delta_pct == 25.0
    assert deltas[0].elapsed_delta_pct == -20.0
    assert deltas[0].key == "qemu_prompt/secure-local/tiny/Q4_0/smoke"


def test_cli_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "results"
    write_report(baseline, "base", 100.0, 200000)
    write_report(candidate, "head", 90.0, 220000)

    status = build_compare.main(
        [
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "build_compare_latest.md").read_text(encoding="utf-8")

    assert payload["baseline_build"] == "base"
    assert payload["deltas"][0]["tok_per_s_delta_pct"] == -10.0
    assert "Build Benchmark Compare" in markdown
    assert "| head | qemu_prompt/secure-local/tiny/Q4_0/smoke | 100.000 | 90.000 | -10.000 |" in markdown


def test_missing_baseline_returns_error(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    write_report(report, "head", 100.0, 200000)

    status = build_compare.main(["--input", f"head={report}", "--baseline", "missing"])

    assert status == 2


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_compare_builds_computes_tok_per_s_and_elapsed_deltas(tmp_path)
        test_cli_writes_json_and_markdown_reports(tmp_path)
        test_missing_baseline_returns_error(tmp_path)
    print("build_compare_tests=ok")
