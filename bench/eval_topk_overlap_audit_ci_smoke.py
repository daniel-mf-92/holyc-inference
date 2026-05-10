#!/usr/bin/env python3
"""CI smoke coverage for eval_topk_overlap_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_topk_overlap_audit


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-topk-overlap-") as tmp:
        out = Path(tmp) / "out"
        status = eval_topk_overlap_audit.main(
            [
                "--gold",
                str(ROOT / "bench/datasets/samples/smoke_eval.jsonl"),
                "--holyc",
                str(ROOT / "bench/eval/samples/holyc_smoke_scored_predictions.jsonl"),
                "--llama",
                str(ROOT / "bench/eval/samples/llama_smoke_scored_predictions.jsonl"),
                "--dataset",
                "smoke-eval",
                "--split",
                "validation",
                "--top-k",
                "2",
                "--min-pair-coverage-pct",
                "100",
                "--min-topk-exact-match-pct",
                "100",
                "--min-avg-jaccard",
                "1",
                "--max-top1-disagree-pct",
                "0",
                "--output-dir",
                str(out),
                "--output-stem",
                "topk",
            ]
        )
        if status != 0:
            return status
        payload = json.loads((out / "topk.json").read_text(encoding="utf-8"))
        assert payload["status"] == "pass"
        assert payload["summary"]["paired_scored_records"] == 3
        assert payload["summary"]["topk_exact_match_pct"] == 100.0

        bad_llama = Path(tmp) / "bad_llama.jsonl"
        bad_llama.write_text(
            "\n".join(
                [
                    json.dumps({"id": "smoke-hellaswag-1", "scores": [0.0, 5.0, 0.5, 0.25]}),
                    json.dumps({"id": "smoke-arc-1", "scores": [4.1, 1.0, 0.0, -1.0]}),
                    json.dumps({"id": "smoke-truthfulqa-1", "scores": [3.2, 0.0, -1.0, -2.0]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        failing = eval_topk_overlap_audit.main(
            [
                "--gold",
                str(ROOT / "bench/datasets/samples/smoke_eval.jsonl"),
                "--holyc",
                str(ROOT / "bench/eval/samples/holyc_smoke_scored_predictions.jsonl"),
                "--llama",
                str(bad_llama),
                "--dataset",
                "smoke-eval",
                "--split",
                "validation",
                "--top-k",
                "1",
                "--min-topk-exact-match-pct",
                "100",
                "--max-top1-disagree-pct",
                "0",
                "--output-dir",
                str(out),
                "--output-stem",
                "topk_fail",
            ]
        )
        if failing == 0:
            return 1
        failed = json.loads((out / "topk_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed["findings"]}
        assert {"topk_exact_match", "top1_disagree"} <= kinds
    print("eval_topk_overlap_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
