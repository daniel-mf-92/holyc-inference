#!/usr/bin/env python3
"""CI smoke for qemu_latest_alias_audit."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qemu_latest_alias_audit


def write_artifact(path: Path, generated_at: str, prompt: str = "smoke") -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "status": "pass",
                "benchmarks": [{"phase": "measured", "prompt": prompt}],
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
        ok_dir = root / "ok"
        ok_dir.mkdir()
        latest = ok_dir / "qemu_prompt_bench_latest.json"
        stamped = ok_dir / "qemu_prompt_bench_20260501T100000Z.json"
        write_artifact(latest, "2026-05-01T10:00:00Z")
        write_artifact(stamped, "2026-05-01T10:00:00Z")

        ok = qemu_latest_alias_audit.main(
            [str(ok_dir), "--output-dir", str(output / "ok"), "--output-stem", "alias"]
        )
        if rc := require(ok == 0, "matching_latest_alias_failed"):
            return rc

        drift_dir = root / "drift"
        drift_dir.mkdir()
        write_artifact(drift_dir / "qemu_prompt_bench_latest.json", "2026-05-01T10:00:00Z", "old")
        write_artifact(drift_dir / "qemu_prompt_bench_20260501T100000Z.json", "2026-05-01T10:00:00Z", "new")

        failed = qemu_latest_alias_audit.main(
            [str(drift_dir), "--output-dir", str(output / "failed"), "--output-stem", "alias"]
        )
        if rc := require(failed == 1, "drifted_latest_alias_passed"):
            return rc

        payload = json.loads((output / "failed" / "alias.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in payload["findings"]}
        if rc := require("latest_alias_payload_drift" in kinds, "missing_payload_drift_finding"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
