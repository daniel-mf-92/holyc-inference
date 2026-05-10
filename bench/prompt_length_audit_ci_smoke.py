#!/usr/bin/env python3
"""Smoke gate for prompt length coverage audit."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import prompt_length_audit


def write_suite(path: Path) -> None:
    rows = [
        {"id": "short", "prompt": "short prompt", "expected_tokens": 4},
        {"id": "medium", "prompt": "m" * 180, "expected_tokens": 16},
        {"id": "long", "prompt": "l" * 640, "expected_tokens": 32},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        suite = root / "prompts.jsonl"
        out = root / "out"
        write_suite(suite)
        status = prompt_length_audit.main(
            [
                str(suite),
                "--output-dir",
                str(out),
                "--output-stem",
                "prompt_length_audit_smoke",
                "--min-total-prompts",
                "3",
                "--min-bucket-prompts",
                "short=1",
                "--min-bucket-prompts",
                "medium=1",
                "--min-bucket-prompts",
                "long=1",
            ]
        )
        if status != 0:
            return status
        latest_dir = Path("bench/results")
        latest_dir.mkdir(parents=True, exist_ok=True)
        for suffix in (".json", ".md", ".csv", "_prompts.csv", "_findings.csv", "_junit.xml"):
            source = out / f"prompt_length_audit_smoke{suffix}"
            target = latest_dir / f"prompt_length_audit_smoke_latest{suffix}"
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
