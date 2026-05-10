#!/usr/bin/env python3
"""Smoke gate for quant_manifest_audit.py."""

from __future__ import annotations

import json
import importlib.util
import struct
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "quant_manifest_audit.py"
spec = importlib.util.spec_from_file_location("quant_manifest_audit", AUDIT_PATH)
quant_manifest_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["quant_manifest_audit"] = quant_manifest_audit
spec.loader.exec_module(quant_manifest_audit)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        q4 = root / "weights.q4_0"
        q8 = root / "weights.q8_0"
        q4.write_bytes(struct.pack("<e", 1.0) + bytes([0x88] * 16))
        q8.write_bytes(struct.pack("<e", 1.0) + bytes([0] * 32))
        manifest = root / "manifest.json"
        write_json(
            manifest,
            {
                "format": "quant-block-manifest",
                "artifacts": [
                    {
                        "path": "weights.q4_0",
                        "format": "q4_0",
                        "sha256": quant_manifest_audit.file_sha256(q4),
                        "bytes": q4.stat().st_size,
                        "block_count": 1,
                        "element_count": 32,
                    },
                    {
                        "path": "weights.q8_0",
                        "format": "q8_0",
                        "sha256": quant_manifest_audit.file_sha256(q8),
                        "bytes": q8.stat().st_size,
                        "block_count": 1,
                        "element_count": 31,
                    },
                ],
            },
        )
        output = root / "audit.json"
        csv = root / "audit.csv"
        markdown = root / "audit.md"
        junit = root / "audit.xml"
        status = quant_manifest_audit.main(
            [
                "--manifest",
                str(manifest),
                "--root",
                str(root),
                "--output",
                str(output),
                "--csv",
                str(csv),
                "--markdown",
                str(markdown),
                "--junit",
                str(junit),
                "--fail-on-findings",
            ]
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["artifact_count"] == 2
        assert "Status: PASS" in markdown.read_text(encoding="utf-8")

        bad_manifest = root / "bad_manifest.json"
        write_json(
            bad_manifest,
            {
                "path": "weights.q4_0",
                "format": "q4_0",
                "sha256": "bad",
                "bytes": 17,
                "block_count": 2,
                "element_count": 64,
            },
        )
        bad_status = quant_manifest_audit.main(
            [
                "--manifest",
                str(bad_manifest),
                "--root",
                str(root),
                "--output",
                str(root / "bad_audit.json"),
                "--fail-on-findings",
            ]
        )
        bad_report = json.loads((root / "bad_audit.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in bad_report["findings"]}
        assert bad_status == 1
        assert {"sha256_mismatch", "byte_count_mismatch", "block_count_mismatch", "element_count_over_capacity"}.issubset(kinds)
    print("quant_manifest_audit_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
