#!/usr/bin/env python3
"""CI smoke for eval_identity_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "bench" / "eval_identity_audit.py"


def write_predictions(path: Path, *, model_sha: str = "model-a", tokenizer_sha: str = "tok-a", quantization: str = "Q4_0") -> None:
    rows = [
        {
            "id": "case-1",
            "scores": [1.0, 0.0],
            "metadata": {
                "model": "smoke-model",
                "model_sha256": model_sha,
                "tokenizer_sha256": tokenizer_sha,
                "quantization": quantization,
                "prompt_template_sha256": "prompt-a",
            },
        },
        {
            "id": "case-2",
            "scores": [0.0, 1.0],
            "metadata": {
                "model": "smoke-model",
                "model_sha256": model_sha,
                "tokenizer_sha256": tokenizer_sha,
                "quantization": quantization,
                "prompt_template_sha256": "prompt-a",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def require(condition: bool, message: str) -> int:
    if not condition:
        print(message, file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-identity-") as tmp:
        tmp_path = Path(tmp)
        holyc = tmp_path / "holyc.jsonl"
        llama = tmp_path / "llama.jsonl"
        write_predictions(holyc)
        write_predictions(llama)

        pass_dir = tmp_path / "pass"
        passed = run_command(
            [
                sys.executable,
                str(TOOL),
                str(holyc),
                str(llama),
                "--require-identity",
                "--compare-key",
                "model_sha256",
                "--compare-key",
                "tokenizer_sha256",
                "--compare-key",
                "quantization",
                "--output-dir",
                str(pass_dir),
                "--output-stem",
                "identity",
            ]
        )
        if rc := require(passed.returncode == 0, passed.stderr):
            return rc
        payload = json.loads((pass_dir / "identity.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "expected passing identity audit"):
            return rc
        for suffix in [".csv", "_findings.csv", ".md", "_junit.xml"]:
            if rc := require((pass_dir / f"identity{suffix}").exists(), f"missing output {suffix}"):
                return rc

        drifted = tmp_path / "llama_drifted.jsonl"
        write_predictions(drifted, model_sha="model-b")
        fail_dir = tmp_path / "fail"
        failed = run_command(
            [
                sys.executable,
                str(TOOL),
                str(holyc),
                str(drifted),
                "--compare-key",
                "model_sha256",
                "--output-dir",
                str(fail_dir),
                "--output-stem",
                "identity",
            ]
        )
        if rc := require(failed.returncode == 1, "expected identity drift to fail"):
            return rc
        fail_payload = json.loads((fail_dir / "identity.json").read_text(encoding="utf-8"))
        gates = {finding["gate"] for finding in fail_payload["findings"]}
        if rc := require("cross_engine_mismatch" in gates, "missing cross-engine mismatch finding"):
            return rc

    print("eval_identity_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
