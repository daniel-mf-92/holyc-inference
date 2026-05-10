#!/usr/bin/env python3
"""Smoke gate for QEMU timeout recommendation reporting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench" / "qemu_timeout_recommend.py"
RESULTS = ROOT / "bench" / "results"


def write_artifact(path: Path) -> None:
    rows = [
        {
            "benchmark": "qemu_prompt",
            "profile": "ci-airgap-smoke",
            "model": "synthetic-smoke",
            "quantization": "Q4_0",
            "phase": "measured",
            "prompt": "smoke-short",
            "wall_elapsed_us": 10_000_000,
            "timeout_seconds": 60,
            "returncode": 0,
            "timed_out": False,
            "exit_class": "ok",
        },
        {
            "benchmark": "qemu_prompt",
            "profile": "ci-airgap-smoke",
            "model": "synthetic-smoke",
            "quantization": "Q4_0",
            "phase": "measured",
            "prompt": "smoke-code",
            "wall_elapsed_us": 12_000_000,
            "timeout_seconds": 60,
            "returncode": 0,
            "timed_out": False,
            "exit_class": "ok",
        },
    ]
    path.write_text(json.dumps({"status": "pass", "benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-timeout-recommend-") as tmp:
        artifact = Path(tmp) / "qemu_prompt_bench_latest.json"
        write_artifact(artifact)
        output_dir = Path(tmp) / "out"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(artifact),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_timeout_recommend_smoke",
                "--min-samples",
                "2",
                "--require-timeout-telemetry",
            ],
            check=True,
        )
        payload = json.loads((output_dir / "qemu_timeout_recommend_smoke.json").read_text(encoding="utf-8"))
        assert payload["status"] == "pass"
        assert payload["recommendations"][0]["recommended_timeout_s"] == 41
        assert payload["recommendations"][0]["current_timeout_headroom_pct"] == 80.0

        fail_dir = Path(tmp) / "fail"
        failed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(artifact),
                "--output-dir",
                str(fail_dir),
                "--output-stem",
                "qemu_timeout_recommend_headroom_fail",
                "--min-samples",
                "2",
                "--require-timeout-telemetry",
                "--min-current-timeout-headroom-pct",
                "90",
            ],
            text=True,
            capture_output=True,
        )
        assert failed.returncode == 1, failed.stdout + failed.stderr
        fail_payload = json.loads((fail_dir / "qemu_timeout_recommend_headroom_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_payload["findings"]}
        assert "current_timeout_headroom_low" in kinds

    RESULTS.mkdir(parents=True, exist_ok=True)
    latest_artifact = RESULTS / "qemu_prompt_bench_latest.json"
    if latest_artifact.exists():
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(latest_artifact),
                "--output-dir",
                str(RESULTS),
                "--output-stem",
                "qemu_timeout_recommend_latest",
                "--no-require-rows",
            ],
            check=False,
        )
    print("qemu_timeout_recommend_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
