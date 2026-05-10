#!/usr/bin/env python3
"""CI smoke gate for eval_artifact_identity_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_artifact_identity_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def rows(*, model_sha256: str = "m" * 64, tokenizer_sha256: str = "t" * 64, quantization: str = "Q4_0") -> list[dict[str, object]]:
    return [
        {
            "id": "arc-1",
            "prediction": 1,
            "model": "tiny-smoke",
            "model_sha256": model_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "quantization": quantization,
        },
        {
            "id": "arc-2",
            "prediction": 0,
            "metadata": {
                "model": "tiny-smoke",
                "model_sha256": model_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "quantization": quantization,
            },
        },
    ]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-artifact-identity-") as tmp:
        root = Path(tmp)
        holyc = root / "holyc.jsonl"
        llama = root / "llama.jsonl"
        write_jsonl(holyc, rows())
        write_jsonl(llama, rows())

        output_dir = root / "results"
        status = eval_artifact_identity_audit.main(
            [
                "--holyc",
                str(holyc),
                "--llama",
                str(llama),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_artifact_identity_audit_smoke",
                "--fail-on-findings",
            ]
        )
        if status != 0:
            return status
        required = [
            output_dir / "eval_artifact_identity_audit_smoke.json",
            output_dir / "eval_artifact_identity_audit_smoke.csv",
            output_dir / "eval_artifact_identity_audit_smoke_pairs.csv",
            output_dir / "eval_artifact_identity_audit_smoke_findings.csv",
            output_dir / "eval_artifact_identity_audit_smoke.md",
            output_dir / "eval_artifact_identity_audit_smoke_junit.xml",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            print(f"missing smoke outputs: {missing}", file=sys.stderr)
            return 1

        bad_llama = root / "llama_bad.jsonl"
        write_jsonl(bad_llama, rows(model_sha256="b" * 64))
        bad_status = eval_artifact_identity_audit.main(
            [
                "--holyc",
                str(holyc),
                "--llama",
                str(bad_llama),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_artifact_identity_audit_bad",
                "--fail-on-findings",
            ]
        )
        if bad_status == 0:
            print("model_sha256_mismatch_not_detected=true", file=sys.stderr)
            return 1

        missing_holyc = root / "holyc_missing.jsonl"
        missing_rows = rows()
        missing_rows[0].pop("tokenizer_sha256")
        write_jsonl(missing_holyc, missing_rows)
        missing_status = eval_artifact_identity_audit.main(
            [
                "--holyc",
                str(missing_holyc),
                "--llama",
                str(llama),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "eval_artifact_identity_audit_missing",
                "--fail-on-findings",
            ]
        )
        if missing_status == 0:
            print("missing_tokenizer_sha256_not_detected=true", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
