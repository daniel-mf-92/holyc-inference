#!/usr/bin/env python3
"""Smoke gate for HCEval binary diff reports."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_pack
import hceval_diff


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def pack(path: Path, rows: list[dict[str, object]], output: Path) -> None:
    write_jsonl(path, rows)
    records = dataset_pack.normalize_records(dataset_pack.read_jsonl(path), "smoke-eval", "validation")
    dataset_pack.write_outputs(records, output, output.with_suffix(".manifest.json"), "smoke-eval", "validation")


def smoke_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "smoke-1",
            "dataset": "smoke-eval",
            "split": "validation",
            "prompt": "Pick the warm color.",
            "choices": ["red", "blue", "green", "gray"],
            "answer_index": 0,
            "provenance": "synthetic smoke",
        },
        {
            "id": "smoke-2",
            "dataset": "smoke-eval",
            "split": "validation",
            "prompt": "Pick the even number.",
            "choices": ["one", "two", "three", "five"],
            "answer_index": 1,
            "provenance": "synthetic smoke",
        },
    ]


def main() -> int:
    output_dir = Path("bench/results/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = output_dir / "hceval_diff_smoke_reference.hceval"
    candidate = output_dir / "hceval_diff_smoke_candidate.hceval"
    rows = smoke_rows()
    pack(output_dir / "hceval_diff_smoke_reference.jsonl", rows, reference)
    pack(output_dir / "hceval_diff_smoke_candidate.jsonl", rows, candidate)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        changed = tmp / "changed.hceval"
        changed_rows = [dict(rows[1]), dict(rows[0])]
        changed_rows[1]["answer_index"] = 2
        pack(tmp / "changed.jsonl", changed_rows, changed)

        status = hceval_diff.main(
            [
                "--reference",
                str(reference),
                "--candidate",
                str(candidate),
                "--output",
                str(output_dir / "hceval_diff_smoke_latest.json"),
                "--markdown",
                str(output_dir / "hceval_diff_smoke_latest.md"),
                "--csv",
                str(output_dir / "hceval_diff_smoke_latest.csv"),
                "--findings-csv",
                str(output_dir / "hceval_diff_smoke_latest_findings.csv"),
                "--junit",
                str(output_dir / "hceval_diff_smoke_latest_junit.xml"),
            ]
        )
        if status != 0:
            return status

        failing_status = hceval_diff.main(
            ["--reference", str(reference), "--candidate", str(changed), "--allow-order-changes"]
        )
        return 0 if failing_status == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
