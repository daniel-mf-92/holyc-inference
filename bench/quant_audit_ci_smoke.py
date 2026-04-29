#!/usr/bin/env python3
"""Stdlib-only smoke gate for quant_audit.py raw block checks."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def write_clean_source(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "QuantSmoke.HC").write_text(
        "I64 QuantSmokeAdd(I64 a, I64 b) { return a + b; }\n",
        encoding="utf-8",
    )


def write_q4_block(path: Path, scale_bits: int) -> None:
    payload = bytes(range(16))
    path.write_bytes(struct.pack("<H", scale_bits) + payload)


def write_q8_block(path: Path, scale_bits: int) -> None:
    path.write_bytes(struct.pack("<H32b", scale_bits, *range(-16, 16)))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-quant-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        source_root = tmp_path / "src" / "quant"
        write_clean_source(source_root)

        q4_path = tmp_path / "q4_good.bin"
        q8_path = tmp_path / "q8_good.bin"
        write_q4_block(q4_path, 0x3C00)
        write_q8_block(q8_path, 0x3C00)

        pass_json = tmp_path / "quant_audit_latest.json"
        pass_md = tmp_path / "quant_audit_latest.md"
        pass_csv = tmp_path / "quant_audit_latest.csv"
        pass_junit = tmp_path / "quant_audit_junit_latest.xml"
        pass_command = [
            sys.executable,
            str(ROOT / "bench" / "quant_audit.py"),
            "--source-root",
            str(source_root),
            "--q4-block-file",
            str(q4_path),
            "--q8-block-file",
            str(q8_path),
            "--expect-blocks",
            "1",
            "--expect-elements",
            "32",
            "--min-used-quant-values",
            "16",
            "--min-scale-exponent",
            "-1",
            "--max-scale-exponent",
            "1",
            "--output",
            str(pass_json),
            "--markdown",
            str(pass_md),
            "--csv",
            str(pass_csv),
            "--junit",
            str(pass_junit),
        ]
        completed = run_command(pass_command)
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads(pass_json.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(len(report["block_audits"]) == 2, "unexpected_block_audit_count"):
            return rc
        for audit in report["block_audits"]:
            if rc := require(audit["scale_exponent_min"] == 0, "missing_scale_exponent_min"):
                return rc
            if rc := require(audit["scale_exponent_max"] == 0, "missing_scale_exponent_max"):
                return rc
            if rc := require(audit["scale_exponent_histogram"] == {"0": 1}, "missing_scale_exponent_histogram"):
                return rc
            if rc := require(audit["scale_exponent_under_limit_count"] == 0, "unexpected_under_limit"):
                return rc
            if rc := require(audit["scale_exponent_over_limit_count"] == 0, "unexpected_over_limit"):
                return rc
        if rc := require("Scale exponent min/max/under/over" in pass_md.read_text(encoding="utf-8"), "missing_markdown_exponent"):
            return rc
        if rc := require(
            "scope,path,line,column,format,kind,reason,text" in pass_csv.read_text(encoding="utf-8"),
            "missing_csv_header",
        ):
            return rc
        junit_root = ET.parse(pass_junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_quant_audit", "missing_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        bad_q4 = tmp_path / "q4_bad_scale_exponent.bin"
        write_q4_block(bad_q4, 0x0001)
        bad_json = tmp_path / "bad_quant_audit.json"
        bad_command = [
            sys.executable,
            str(ROOT / "bench" / "quant_audit.py"),
            "--source-root",
            str(source_root),
            "--format",
            "q4_0",
            "--block-file",
            str(bad_q4),
            "--min-scale-exponent",
            "0",
            "--output",
            str(bad_json),
        ]
        completed = subprocess.run(
            bad_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("bad_scale_exponent_not_rejected=true", file=sys.stderr)
            return 1
        bad_report = json.loads(bad_json.read_text(encoding="utf-8"))
        bad_audit = bad_report["block_audits"][0]
        if rc := require(bad_audit["scale_exponent_under_limit_count"] == 1, "missing_under_limit_count"):
            return rc
        if rc := require(
            any("fp16 scale exponent -14 below minimum 0" in finding for finding in bad_audit["findings"]),
            "missing_scale_exponent_finding",
        ):
            return rc

    print("quant_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
