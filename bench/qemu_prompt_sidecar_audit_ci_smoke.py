#!/usr/bin/env python3
"""Smoke gate for qemu_prompt_sidecar_audit.py."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qemu_prompt_sidecar_audit


def write_artifact_bundle(root: Path, stem: str = "qemu_prompt_bench_latest") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / f"{stem}.json"
    base = stem[: -len("_latest")] if stem.endswith("_latest") else stem
    artifact.write_text('{"generated_at":"2026-05-01T00:00:00Z","status":"pass","benchmarks":[]}\n', encoding="utf-8")
    artifact.with_suffix(".csv").write_text("status\npass\n", encoding="utf-8")
    artifact.with_suffix(".md").write_text("# report\n", encoding="utf-8")
    artifact.with_name(f"{base}_junit_latest.xml").write_text("<testsuite failures=\"0\" />\n", encoding="utf-8")
    return artifact


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-sidecar-smoke-") as tmp:
        root = Path(tmp)
        artifact = write_artifact_bundle(root)
        output_dir = root / "out"
        status = qemu_prompt_sidecar_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "sidecar"])
        if status != 0:
            return status
        missing = root / "qemu_prompt_bench_dry_run_latest.json"
        missing.write_text('{"generated_at":"2026-05-01T00:00:00Z","status":"pass","benchmarks":[]}\n', encoding="utf-8")
        fail_status = qemu_prompt_sidecar_audit.main([str(missing), "--output-dir", str(output_dir), "--output-stem", "sidecar_fail"])
        if fail_status == 0:
            raise AssertionError("missing sidecars should fail")
    print("qemu_prompt_sidecar_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
