#!/usr/bin/env python3
"""Stdlib-only smoke gate for quant manifest padding audit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def q4_block(values: list[int]) -> bytes:
    if len(values) != 32:
        raise ValueError("Q4_0 smoke blocks need 32 values")
    encoded = bytearray(b"\x00\x3c")
    for index in range(0, 32, 2):
        low = (values[index] + 8) & 0x0F
        high = (values[index + 1] + 8) & 0x0F
        encoded.append(low | (high << 4))
    return bytes(encoded)


def q8_block(values: list[int]) -> bytes:
    if len(values) != 32:
        raise ValueError("Q8_0 smoke blocks need 32 values")
    return b"\x00\x3c" + bytes(value & 0xFF for value in values)


def run_audit(manifest: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "quant_manifest_padding_audit.py"),
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "quant_manifest_padding_audit_smoke",
            *extra_args,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-quant-padding-smoke-") as tmp:
        root = Path(tmp)
        safe_q4 = root / "safe_q4.bin"
        safe_q8 = root / "safe_q8.bin"
        bad_q4 = root / "bad_q4.bin"
        safe_q4.write_bytes(q4_block([1] * 17 + [0] * 15))
        safe_q8.write_bytes(q8_block([1] * 17 + [0] * 15))
        bad_q4.write_bytes(q4_block([1] * 17 + [3] + [0] * 14))

        safe_manifest = root / "safe_manifest.json"
        safe_manifest.write_text(
            json.dumps(
                {
                    "artifacts": [
                        {"name": "safe.q4", "path": str(safe_q4), "format": "q4_0", "elements": 17, "blocks": 1},
                        {"name": "safe.q8", "path": str(safe_q8), "format": "q8_0", "elements": 17, "blocks": 1},
                    ]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        safe_out = root / "safe_out"
        completed = run_audit(safe_manifest, safe_out)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        safe_report = json.loads((safe_out / "quant_manifest_padding_audit_smoke.json").read_text(encoding="utf-8"))
        safe_junit_text = (safe_out / "quant_manifest_padding_audit_smoke_junit.xml").read_text(encoding="utf-8")
        checks = [
            require(safe_report["status"] == "pass", "safe_padding_audit_not_pass=true"),
            require(safe_report["summary"]["entries"] == 2, "safe_padding_audit_entry_count_drift=true"),
            require(safe_report["summary"]["padding_elements"] == 30, "safe_padding_audit_padding_count_drift=true"),
            require(safe_report["summary"]["nonzero_padding_elements"] == 0, "safe_padding_audit_nonzero_padding=true"),
            require('failures="0"' in safe_junit_text, "safe_padding_audit_junit_failures=true"),
        ]
        if not all(checks):
            return 1

        bad_manifest = root / "bad_manifest.json"
        bad_manifest.write_text(
            json.dumps({"artifacts": [{"name": "bad.q4", "path": str(bad_q4), "format": "q4_0", "elements": 17, "blocks": 1}]}, indent=2) + "\n",
            encoding="utf-8",
        )
        bad_out = root / "bad_out"
        failed = run_audit(bad_manifest, bad_out)
        if failed.returncode == 0:
            print("bad_padding_audit_not_rejected=true", file=sys.stderr)
            return 1
        bad_report = json.loads((bad_out / "quant_manifest_padding_audit_smoke.json").read_text(encoding="utf-8"))
        checks = [
            require(bad_report["status"] == "fail", "bad_padding_audit_not_fail=true"),
            require(bad_report["summary"]["nonzero_padding_elements"] == 1, "bad_padding_audit_nonzero_count_drift=true"),
            require(any(finding["kind"] == "nonzero_padding_quant" for finding in bad_report["findings"]), "bad_padding_audit_missing_finding=true"),
        ]
        if not all(checks):
            return 1

        warning_out = root / "warning_out"
        warned = run_audit(bad_manifest, warning_out, "--allow-nonzero-padding")
        if warned.returncode != 0:
            sys.stdout.write(warned.stdout)
            sys.stderr.write(warned.stderr)
            return warned.returncode
        warning_report = json.loads((warning_out / "quant_manifest_padding_audit_smoke.json").read_text(encoding="utf-8"))
        checks = [
            require(warning_report["status"] == "pass", "warning_padding_audit_not_pass=true"),
            require(any(finding["severity"] == "warning" for finding in warning_report["findings"]), "warning_padding_audit_missing_warning=true"),
        ]
        if not all(checks):
            return 1

    print("quant_manifest_padding_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
