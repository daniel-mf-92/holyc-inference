#!/usr/bin/env python3
"""CI smoke gate for qemu_replay_manifest_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_replay_manifest
import qemu_replay_manifest_audit
from qemu_replay_manifest_ci_smoke import write_smoke_artifact


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-replay-manifest-audit-") as tmp:
        root = Path(tmp)
        artifact = root / "qemu_prompt_bench_latest.json"
        manifest_dir = root / "manifest"
        output_dir = root / "audit"
        write_smoke_artifact(artifact)
        manifest_status = qemu_replay_manifest.main(
            [str(artifact), "--output-dir", str(manifest_dir), "--output-stem", "qemu_replay_manifest_smoke"]
        )
        if manifest_status != 0:
            return manifest_status
        status = qemu_replay_manifest_audit.main(
            [
                str(manifest_dir / "qemu_replay_manifest_smoke.json"),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_replay_manifest_audit_smoke",
            ]
        )
        required = [
            output_dir / "qemu_replay_manifest_audit_smoke.json",
            output_dir / "qemu_replay_manifest_audit_smoke.csv",
            output_dir / "qemu_replay_manifest_audit_smoke_findings.csv",
            output_dir / "qemu_replay_manifest_audit_smoke.md",
            output_dir / "qemu_replay_manifest_audit_smoke_junit.xml",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            print(f"missing smoke outputs: {missing}", file=sys.stderr)
            return 1
        if status != 0:
            return status

        bad_manifest = manifest_dir / "qemu_replay_manifest_bad_source_hash.json"
        bad_manifest.write_text((manifest_dir / "qemu_replay_manifest_smoke.json").read_text(encoding="utf-8"), encoding="utf-8")
        payload = qemu_replay_manifest.load_json_object(bad_manifest)[0]
        if payload is None:
            print("bad_manifest_load_failed=true", file=sys.stderr)
            return 1
        payload["entries"][0]["source_sha256"] = "0" * 64
        bad_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bad_status = qemu_replay_manifest_audit.main(
            [
                str(bad_manifest),
                "--argv-jsonl",
                str(manifest_dir / "qemu_replay_manifest_smoke_argv.jsonl"),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_replay_manifest_audit_bad_source_hash",
            ]
        )
        if bad_status == 0:
            print("source_sha256_drift_not_detected=true", file=sys.stderr)
            return 1

        bad_qemu_bin = manifest_dir / "qemu_replay_manifest_bad_qemu_bin.json"
        bad_qemu_bin.write_text(
            (manifest_dir / "qemu_replay_manifest_smoke.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        qemu_bin_payload = qemu_replay_manifest.load_json_object(bad_qemu_bin)[0]
        if qemu_bin_payload is None:
            print("bad_qemu_bin_load_failed=true", file=sys.stderr)
            return 1
        qemu_bin_payload["entries"][0]["qemu_bin"] = "qemu-system-drift"
        bad_qemu_bin.write_text(json.dumps(qemu_bin_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bad_qemu_bin_status = qemu_replay_manifest_audit.main(
            [
                str(bad_qemu_bin),
                "--argv-jsonl",
                str(manifest_dir / "qemu_replay_manifest_smoke_argv.jsonl"),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_replay_manifest_audit_bad_qemu_bin",
            ]
        )
        if bad_qemu_bin_status == 0:
            print("qemu_bin_drift_not_detected=true", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
