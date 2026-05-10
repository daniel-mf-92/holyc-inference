#!/usr/bin/env python3
"""CI smoke for qemu_result_freshness_audit."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qemu_result_freshness_audit


def write_artifact(path: Path, generated_at: str) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "status": "pass",
                "benchmarks": [{"phase": "measured", "prompt": "smoke"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "out"
        fresh = root / "fresh" / "qemu_prompt_bench_latest.json"
        stale = root / "stale" / "qemu_prompt_bench_latest.json"
        fresh.parent.mkdir()
        stale.parent.mkdir()
        write_artifact(fresh, "2026-05-01T10:30:00Z")
        write_artifact(stale, "2026-05-01T08:00:00Z")

        ok = qemu_result_freshness_audit.main(
            [
                str(fresh.parent),
                "--output-dir",
                str(output / "ok"),
                "--output-stem",
                "freshness",
                "--now",
                "2026-05-01T11:00:00Z",
                "--max-age-hours",
                "1",
            ]
        )
        if rc := require(ok == 0, "fresh_artifact_failed"):
            return rc

        failed = qemu_result_freshness_audit.main(
            [
                str(stale.parent),
                "--output-dir",
                str(output / "failed"),
                "--output-stem",
                "freshness",
                "--now",
                "2026-05-01T11:00:00Z",
                "--max-age-hours",
                "1",
            ]
        )
        if rc := require(failed == 1, "stale_artifact_passed"):
            return rc

        payload = json.loads((output / "failed" / "freshness.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in payload["findings"]}
        if rc := require("stale_artifact" in kinds, "missing_stale_finding"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
