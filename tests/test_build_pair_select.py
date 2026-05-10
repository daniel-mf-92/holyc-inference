#!/usr/bin/env python3
"""Tests for benchmark build-pair selection tooling."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import build_pair_select


def artifact(
    source: str,
    commit: str,
    generated_at: str,
    tok_per_s: float,
    *,
    measured_runs: int = 4,
    wall_tok_per_s: float | None = 90.0,
    memory_bytes: int = 4096,
) -> dict[str, object]:
    return {
        "source": source,
        "artifact_type": "qemu_prompt",
        "status": "pass",
        "generated_at": generated_at,
        "commit": commit,
        "profile": "ci",
        "model": "tiny",
        "quantization": "Q4_0",
        "prompt_suite_sha256": "suite",
        "command_sha256": "command",
        "launch_plan_sha256": "launch",
        "environment_sha256": "env",
        "measured_runs": measured_runs,
        "median_tok_per_s": tok_per_s,
        "wall_tok_per_s_median": wall_tok_per_s,
        "max_memory_bytes": memory_bytes,
    }


def test_select_pairs_uses_latest_two_distinct_commits() -> None:
    artifacts = [
        build_pair_select.parse_artifact(artifact("old.json", "base", "2026-04-29T00:00:00Z", 100.0)),
        build_pair_select.parse_artifact(artifact("head-a.json", "head", "2026-04-30T00:00:00Z", 110.0)),
        build_pair_select.parse_artifact(artifact("head-b.json", "head", "2026-05-01T00:00:00Z", 120.0)),
    ]

    pairs, findings = build_pair_select.select_pairs(artifacts)

    assert not [finding for finding in findings if finding.severity == "error"]
    assert len(pairs) == 1
    assert pairs[0].baseline_source == "old.json"
    assert pairs[0].candidate_source == "head-b.json"
    assert pairs[0].median_tok_per_s_delta_pct == 20.0
    assert pairs[0].max_memory_delta_pct == 0.0
    assert pairs[0].build_compare_args == [
        "--input",
        "base-20260429-base=old.json",
        "--input",
        "head-20260501-head=head-b.json",
    ]


def test_cli_writes_json_csv_markdown_and_junit(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    output = tmp_path / "out"
    index.write_text(
        json.dumps(
            {
                "artifacts": [
                    artifact("base.json", "basecommit", "2026-04-29T00:00:00Z", 100.0, memory_bytes=4000),
                    artifact("head.json", "headcommit", "2026-05-01T00:00:00Z", 125.0, wall_tok_per_s=100.0, memory_bytes=5000),
                ]
            }
        ),
        encoding="utf-8",
    )

    status = build_pair_select.main([str(index), "--output-dir", str(output), "--output-stem", "pairs"])

    assert status == 0
    payload = json.loads((output / "pairs.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output / "pairs.csv").open(encoding="utf-8", newline="")))
    markdown = (output / "pairs.md").read_text(encoding="utf-8")
    junit = ET.parse(output / "pairs_junit.xml").getroot()

    assert payload["status"] == "pass"
    assert payload["pairs"][0]["median_tok_per_s_delta_pct"] == 25.0
    assert payload["pairs"][0]["wall_tok_per_s_delta_pct"] == 11.111111
    assert payload["pairs"][0]["max_memory_delta_pct"] == 25.0
    assert rows[0]["candidate_source"] == "head.json"
    assert "Build Pair Select" in markdown
    assert junit.attrib["name"] == "holyc_build_pair_select"
    assert junit.attrib["failures"] == "0"


def test_cli_fails_when_no_comparable_pair(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    output = tmp_path / "out"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(artifact("one.json", "one", "2026-05-01T00:00:00Z", 100.0)))
        writer.writeheader()
        writer.writerow(artifact("one.json", "one", "2026-05-01T00:00:00Z", 100.0))

    status = build_pair_select.main([str(index), "--output-dir", str(output), "--output-stem", "pairs"])

    payload = json.loads((output / "pairs.json").read_text(encoding="utf-8"))
    assert status == 2
    assert payload["status"] == "fail"
    assert any(finding["kind"] == "min_pairs" for finding in payload["findings"])


if __name__ == "__main__":
    test_select_pairs_uses_latest_two_distinct_commits()
    tmp_root = Path("/tmp/holyc-build-pair-select-tests")
    tmp_root.mkdir(parents=True, exist_ok=True)
    writes = tmp_root / "writes"
    fails = tmp_root / "fails"
    writes.mkdir(parents=True, exist_ok=True)
    fails.mkdir(parents=True, exist_ok=True)
    test_cli_writes_json_csv_markdown_and_junit(writes)
    test_cli_fails_when_no_comparable_pair(fails)
    print("build_pair_select_tests=ok")
