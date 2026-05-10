#!/usr/bin/env python3
"""CI smoke gate for qemu_replay_manifest.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_prompt_bench
import qemu_replay_manifest


def write_smoke_artifact(path: Path) -> None:
    command = ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio", "-display", "none"]
    command_airgap = qemu_prompt_bench.command_airgap_metadata(command)
    command_sha256 = qemu_prompt_bench.command_hash(command)
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-30T00:00:00Z",
                "status": "pass",
                "commit": "smoke",
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "command": command,
                "command_sha256": command_sha256,
                "command_airgap": command_airgap,
                "prompt_suite": {
                    "source": "bench/prompts/smoke.jsonl",
                    "suite_sha256": "d" * 64,
                    "prompt_count": 1,
                },
                "launch_plan_sha256": "e" * 64,
                "expected_launch_sequence_sha256": "f" * 64,
                "launch_plan": [{"phase": "measured", "prompt_id": "smoke"}],
                "benchmarks": [{"phase": "measured", "command": command, "command_sha256": command_sha256}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-replay-manifest-") as tmp:
        root = Path(tmp)
        artifact = root / "qemu_prompt_bench_latest.json"
        output_dir = root / "out"
        write_smoke_artifact(artifact)
        status = qemu_replay_manifest.main(
            [str(artifact), "--output-dir", str(output_dir), "--output-stem", "qemu_replay_manifest_smoke"]
        )
        required = [
            output_dir / "qemu_replay_manifest_smoke.json",
            output_dir / "qemu_replay_manifest_smoke.csv",
            output_dir / "qemu_replay_manifest_smoke_argv.jsonl",
            output_dir / "qemu_replay_manifest_smoke_findings.csv",
            output_dir / "qemu_replay_manifest_smoke.md",
            output_dir / "qemu_replay_manifest_smoke_junit.xml",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            print(f"missing smoke outputs: {missing}", file=sys.stderr)
            return 1
        report = json.loads((output_dir / "qemu_replay_manifest_smoke.json").read_text(encoding="utf-8"))
        entry = report["entries"][0]
        if entry["source_sha256"] != qemu_prompt_bench.file_sha256(artifact):
            print("source_sha256_mismatch=true", file=sys.stderr)
            return 1
        if entry["source_size_bytes"] != artifact.stat().st_size:
            print("source_size_bytes_mismatch=true", file=sys.stderr)
            return 1
        if entry["source_mtime_ns"] != artifact.stat().st_mtime_ns:
            print("source_mtime_ns_mismatch=true", file=sys.stderr)
            return 1
        return status


if __name__ == "__main__":
    raise SystemExit(main())
