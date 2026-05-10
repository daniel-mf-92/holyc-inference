#!/usr/bin/env python3
"""Smoke gate for build_pair_select.py."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_pair_select


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(message, file=sys.stderr)
        return 1
    return 0


def artifact(source: str, commit: str, generated_at: str, tok_per_s: float, *, artifact_type: str = "qemu_prompt") -> dict[str, object]:
    return {
        "source": source,
        "artifact_type": artifact_type,
        "status": "pass",
        "generated_at": generated_at,
        "commit": commit,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt_suite_sha256": "suite",
        "command_sha256": "command",
        "launch_plan_sha256": "launch",
        "environment_sha256": "env",
        "measured_runs": 4,
        "median_tok_per_s": tok_per_s,
        "wall_tok_per_s_median": tok_per_s * 0.8,
        "max_memory_bytes": 4096,
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-build-pair-select-") as tmp:
        tmp_path = Path(tmp)
        index = tmp_path / "bench_result_index.json"
        index.write_text(
            json.dumps(
                {
                    "artifacts": [
                        artifact("bench/results/base.json", "basecommit", "2026-05-01T00:00:00Z", 100.0),
                        artifact("bench/results/head.json", "headcommit", "2026-05-02T00:00:00Z", 125.0),
                        artifact(
                            "bench/results/dry_run.json",
                            "dryruncommit",
                            "2026-05-03T00:00:00Z",
                            130.0,
                            artifact_type="qemu_prompt_dry_run",
                        ),
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )

        output_dir = ROOT / "bench" / "results"
        status = build_pair_select.main(
            [
                str(index),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "build_pair_select_smoke_latest",
                "--min-measured-runs",
                "4",
                "--require-wall-tok-per-s",
            ]
        )
        if status != 0:
            return status

        payload = json.loads((output_dir / "build_pair_select_smoke_latest.json").read_text(encoding="utf-8"))
        rows = list(csv.DictReader((output_dir / "build_pair_select_smoke_latest.csv").open(encoding="utf-8", newline="")))
        junit = (output_dir / "build_pair_select_smoke_latest_junit.xml").read_text(encoding="utf-8")
        markdown = (output_dir / "build_pair_select_smoke_latest.md").read_text(encoding="utf-8")

        checks = [
            require(payload["status"] == "pass", "build_pair_select_smoke_status_not_pass=true"),
            require(payload["eligible_pair_count"] == 1, "build_pair_select_smoke_pair_count_drift=true"),
            require(payload["pairs"][0]["median_tok_per_s_delta_pct"] == 25.0, "build_pair_select_smoke_delta_drift=true"),
            require(payload["pairs"][0]["wall_tok_per_s_delta_pct"] == 25.0, "build_pair_select_smoke_wall_delta_drift=true"),
            require(any(finding["kind"] == "skipped_type" for finding in payload["findings"]), "build_pair_select_smoke_missing_skip_finding=true"),
            require(rows[0]["candidate_source"] == "bench/results/head.json", "build_pair_select_smoke_csv_candidate_drift=true"),
            require("Build Pair Select" in markdown, "build_pair_select_smoke_markdown_missing_title=true"),
            require('name="holyc_build_pair_select"' in junit, "build_pair_select_smoke_junit_name_drift=true"),
            require('failures="0"' in junit, "build_pair_select_smoke_junit_failed=true"),
        ]
        if any(checks):
            return 1

        failing_index = tmp_path / "single_commit_index.json"
        failing_index.write_text(json.dumps({"artifacts": [artifact("only.json", "only", "2026-05-04T00:00:00Z", 100.0)]}) + "\n", encoding="utf-8")
        failing_status = build_pair_select.main(
            [
                str(failing_index),
                "--output-dir",
                str(tmp_path / "fail"),
                "--output-stem",
                "build_pair_select_fail",
            ]
        )
        if failing_status == 0:
            print("build_pair_select_smoke_missing_min_pair_failure=true", file=sys.stderr)
            return 1

    print("build_pair_select_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
