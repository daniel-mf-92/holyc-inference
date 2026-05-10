#!/usr/bin/env python3
"""CI smoke runner for eval_tie_audit."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_tie_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_jsonl(
            holyc,
            [
                {"id": "a", "scores": [4.0, 1.0, 0.0, -1.0]},
                {"id": "b", "scores": [0.0, 3.0, 1.0, -1.0]},
            ],
        )
        write_jsonl(
            llama,
            [
                {"id": "a", "scores": [4.1, 1.0, 0.0, -1.0]},
                {"id": "b", "scores": [0.0, 3.2, 1.0, -1.0]},
            ],
        )
        assert eval_tie_audit.main(
            [
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--output-dir",
                str(tmp_path / "pass"),
                "--min-records",
                "2",
                "--max-top-index-disagreement-rate",
                "0",
            ]
        ) == 0

        tied = tmp_path / "holyc_tied.jsonl"
        write_jsonl(tied, [{"id": "a", "scores": [1.0, 1.0, 0.0, -1.0]}])
        assert eval_tie_audit.main(
            [
                "--holyc",
                str(tied),
                "--llama",
                str(llama),
                "--output-dir",
                str(tmp_path / "fail"),
                "--max-top-tie-rate",
                "0",
            ]
        ) == 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
