#!/usr/bin/env python3
"""Smoke test for qemu_benchmark_matrix_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_benchmark_matrix
import qemu_benchmark_matrix_audit


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prompts = root / "prompts.jsonl"
        prompts.write_text(
            "\n".join(
                [
                    json.dumps({"id": "tiny-1", "prompt": "1+1=", "expected_tokens": 2}),
                    json.dumps({"id": "tiny-2", "prompt": "Name a color.", "expected_tokens": 3}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        matrix = root / "matrix.json"
        matrix.write_text(
            json.dumps(
                {
                    "prompts": str(prompts),
                    "profile": "ci-smoke",
                    "model": "synthetic",
                    "quantization": "Q4_0",
                    "warmup": 1,
                    "repeat": 2,
                    "builds": [
                        {"build": "baseline", "image": "baseline.img", "qemu_args": ["-m", "512M"]},
                        {"build": "candidate", "image": "candidate.img", "qemu_args": ["-m", "512M"]},
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        matrix_dir = root / "matrix-out"
        if qemu_benchmark_matrix.main([str(matrix), "--output-dir", str(matrix_dir), "--output-stem", "qemu_benchmark_matrix_smoke_latest"]) != 0:
            return 1
        status = qemu_benchmark_matrix_audit.main(
            [
                str(matrix_dir / "qemu_benchmark_matrix_smoke_latest.json"),
                "--output-dir",
                "bench/results",
                "--output-stem",
                "qemu_benchmark_matrix_audit_smoke_latest",
            ]
        )
        if status != 0:
            return status
        payload = json.loads(Path("bench/results/qemu_benchmark_matrix_audit_smoke_latest.json").read_text(encoding="utf-8"))
        if payload["summary"]["artifacts"] != 1 or payload["summary"]["findings"] != 0:
            return 1
        if payload["summary"]["builds"] != 2 or payload["summary"]["launches"] != 12:
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
