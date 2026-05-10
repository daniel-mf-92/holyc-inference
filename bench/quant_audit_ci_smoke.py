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
    payload = bytes((value | (value << 4)) for value in range(16))
    path.write_bytes(struct.pack("<H", scale_bits) + payload)


def write_q8_block(path: Path, scale_bits: int) -> None:
    path.write_bytes(struct.pack("<H32b", scale_bits, *range(-16, 16)))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-quant-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        source_root = tmp_path / "src" / "quant"
        extra_source_root = tmp_path / "src" / "math"
        write_clean_source(source_root)
        write_clean_source(extra_source_root)

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
            "--extra-source-root",
            str(extra_source_root),
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
            "--max-duplicate-block-pct",
            "0",
            "--max-identical-block-run",
            "1",
            "--min-scale-used-values",
            "1",
            "--max-duplicate-scale-pct",
            "0",
            "--max-identical-scale-run",
            "1",
            "--min-q4-nibble-lane-used-quant-values",
            "16",
            "--min-quant-negative-count",
            "1",
            "--min-quant-positive-count",
            "1",
            "--max-quant-sign-balance-delta",
            "2",
            "--max-zero-quant-pct",
            "10",
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
        if rc := require(report["source_audit"]["files_scanned"] == 2, "unexpected_source_scan_count"):
            return rc
        if rc := require(len(report["source_roots"]) == 2, "unexpected_source_root_count"):
            return rc
        if rc := require(len(report["block_audits"]) == 2, "unexpected_block_audit_count"):
            return rc
        for audit in report["block_audits"]:
            if rc := require(audit["scale_positive_count"] == 1, "missing_scale_positive_count"):
                return rc
            if rc := require(audit["scale_negative_count"] == 0, "unexpected_scale_negative_count"):
                return rc
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
            if rc := require(audit["quant_negative_count"] == 16, "missing_negative_quant_count"):
                return rc
            if rc := require(audit["quant_zero_pct"] <= 10.0, "unexpected_zero_quant_pct"):
                return rc
            if rc := require(audit["duplicate_block_count"] == 0, "unexpected_duplicate_blocks"):
                return rc
            if rc := require(audit["repeated_block_value_count"] == 0, "unexpected_repeated_block_values"):
                return rc
            if rc := require(audit["max_identical_block_run"] == 1, "unexpected_identical_block_run"):
                return rc
            if rc := require(audit["scale_used_value_count"] == 1, "missing_scale_used_value_count"):
                return rc
            if rc := require(audit["duplicate_scale_count"] == 0, "unexpected_duplicate_scales"):
                return rc
            if rc := require(audit["max_identical_scale_run"] == 1, "unexpected_identical_scale_run"):
                return rc
        q4_audit = next(audit for audit in report["block_audits"] if audit["format"] == "q4_0")
        if rc := require(q4_audit["quant_positive_count"] == 14, "missing_q4_positive_quant_count"):
            return rc
        if rc := require(q4_audit["quant_sign_balance_delta"] == 2, "unexpected_q4_quant_sign_delta"):
            return rc
        if rc := require(q4_audit["q4_low_nibble_used_value_count"] == 16, "missing_low_lane_used_values"):
            return rc
        if rc := require(q4_audit["q4_high_nibble_used_value_count"] == 16, "missing_high_lane_used_values"):
            return rc
        if rc := require(q4_audit["q4_nibble_lane_used_value_delta"] == 0, "unexpected_lane_used_delta"):
            return rc
        q8_audit = next(audit for audit in report["block_audits"] if audit["format"] == "q8_0")
        if rc := require(q8_audit["quant_positive_count"] == 15, "missing_q8_positive_quant_count"):
            return rc
        if rc := require(q8_audit["quant_sign_balance_delta"] == 1, "unexpected_q8_quant_sign_delta"):
            return rc
        if rc := require("Scale exponent min/max/under/over" in pass_md.read_text(encoding="utf-8"), "missing_markdown_exponent"):
            return rc
        if rc := require("Duplicate scales" in pass_md.read_text(encoding="utf-8"), "missing_markdown_duplicate_scales"):
            return rc
        if rc := require("Q4 low/high lane used values" in pass_md.read_text(encoding="utf-8"), "missing_markdown_q4_lane"):
            return rc
        if rc := require("Quant sign -/+ delta" in pass_md.read_text(encoding="utf-8"), "missing_markdown_quant_sign"):
            return rc
        if rc := require("Scale sign +/-" in pass_md.read_text(encoding="utf-8"), "missing_markdown_scale_sign"):
            return rc
        if rc := require("Source roots scanned" in pass_md.read_text(encoding="utf-8"), "missing_markdown_source_roots"):
            return rc
        if rc := require("Duplicate blocks" in pass_md.read_text(encoding="utf-8"), "missing_markdown_duplicate_blocks"):
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

        bad_q8 = tmp_path / "q8_bad_negative_scale.bin"
        write_q8_block(bad_q8, 0xBC00)
        bad_negative_json = tmp_path / "bad_negative_quant_audit.json"
        bad_negative_command = [
            sys.executable,
            str(ROOT / "bench" / "quant_audit.py"),
            "--source-root",
            str(source_root),
            "--format",
            "q8_0",
            "--block-file",
            str(bad_q8),
            "--fail-negative-scales",
            "--output",
            str(bad_negative_json),
        ]
        completed = subprocess.run(
            bad_negative_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("bad_negative_scale_not_rejected=true", file=sys.stderr)
            return 1
        bad_negative_report = json.loads(bad_negative_json.read_text(encoding="utf-8"))
        bad_negative_audit = bad_negative_report["block_audits"][0]
        if rc := require(bad_negative_audit["scale_negative_count"] == 1, "missing_negative_scale_count"):
            return rc
        if rc := require(
            any("fp16 scale sign is negative" in finding for finding in bad_negative_audit["findings"]),
            "missing_negative_scale_finding",
        ):
            return rc

        duplicate_q8 = tmp_path / "q8_duplicate_blocks.bin"
        q8_block = struct.pack("<H32b", 0x3C00, *range(-16, 16))
        duplicate_q8.write_bytes(q8_block * 3)
        duplicate_json = tmp_path / "bad_duplicate_quant_audit.json"
        duplicate_command = [
            sys.executable,
            str(ROOT / "bench" / "quant_audit.py"),
            "--source-root",
            str(source_root),
            "--format",
            "q8_0",
            "--block-file",
            str(duplicate_q8),
            "--max-duplicate-block-pct",
            "50",
            "--max-identical-block-run",
            "2",
            "--output",
            str(duplicate_json),
        ]
        completed = subprocess.run(
            duplicate_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("duplicate_blocks_not_rejected=true", file=sys.stderr)
            return 1
        duplicate_report = json.loads(duplicate_json.read_text(encoding="utf-8"))
        duplicate_audit = duplicate_report["block_audits"][0]
        if rc := require(duplicate_audit["duplicate_block_count"] == 2, "missing_duplicate_block_count"):
            return rc
        if rc := require(duplicate_audit["max_identical_block_run"] == 3, "missing_identical_block_run"):
            return rc
        if rc := require(
            any("duplicate blocks 2/3" in finding for finding in duplicate_audit["findings"]),
            "missing_duplicate_block_finding",
        ):
            return rc
        if rc := require(
            any("identical block run 3 exceeds limit 2" in finding for finding in duplicate_audit["findings"]),
            "missing_identical_run_finding",
        ):
            return rc

        repeated_scale_q8 = tmp_path / "q8_repeated_scales.bin"
        repeated_scale_q8.write_bytes(
            struct.pack("<H32b", 0x3C00, *range(-16, 16))
            + struct.pack("<H32b", 0x3C00, *range(-15, 17))
            + struct.pack("<H32b", 0x4000, *range(-14, 18))
        )
        repeated_scale_json = tmp_path / "bad_repeated_scale_quant_audit.json"
        repeated_scale_command = [
            sys.executable,
            str(ROOT / "bench" / "quant_audit.py"),
            "--source-root",
            str(source_root),
            "--format",
            "q8_0",
            "--block-file",
            str(repeated_scale_q8),
            "--min-scale-used-values",
            "3",
            "--max-duplicate-scale-pct",
            "20",
            "--max-identical-scale-run",
            "1",
            "--output",
            str(repeated_scale_json),
        ]
        completed = subprocess.run(
            repeated_scale_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("repeated_scales_not_rejected=true", file=sys.stderr)
            return 1
        repeated_scale_report = json.loads(repeated_scale_json.read_text(encoding="utf-8"))
        repeated_scale_audit = repeated_scale_report["block_audits"][0]
        if rc := require(repeated_scale_audit["scale_used_value_count"] == 2, "missing_repeated_scale_used_values"):
            return rc
        if rc := require(repeated_scale_audit["duplicate_scale_count"] == 1, "missing_duplicate_scale_count"):
            return rc
        if rc := require(repeated_scale_audit["max_identical_scale_run"] == 2, "missing_identical_scale_run"):
            return rc
        if rc := require(
            any("duplicate fp16 scales 1/3" in finding for finding in repeated_scale_audit["findings"]),
            "missing_duplicate_scale_finding",
        ):
            return rc
        if rc := require(
            any("identical fp16 scale run 2 exceeds limit 1" in finding for finding in repeated_scale_audit["findings"]),
            "missing_identical_scale_finding",
        ):
            return rc

        bad_q4_lane = tmp_path / "q4_bad_nibble_lane.bin"
        bad_q4_lane.write_bytes(
            struct.pack("<H", 0x3C00)
            + bytes((low | (8 << 4)) for low in range(16))
        )
        bad_q4_lane_json = tmp_path / "bad_q4_nibble_lane_quant_audit.json"
        bad_q4_lane_command = [
            sys.executable,
            str(ROOT / "bench" / "quant_audit.py"),
            "--source-root",
            str(source_root),
            "--format",
            "q4_0",
            "--block-file",
            str(bad_q4_lane),
            "--min-q4-nibble-lane-used-quant-values",
            "4",
            "--output",
            str(bad_q4_lane_json),
        ]
        completed = subprocess.run(
            bad_q4_lane_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("bad_q4_nibble_lane_not_rejected=true", file=sys.stderr)
            return 1
        bad_q4_lane_report = json.loads(bad_q4_lane_json.read_text(encoding="utf-8"))
        bad_q4_lane_audit = bad_q4_lane_report["block_audits"][0]
        if rc := require(
            bad_q4_lane_audit["q4_low_nibble_used_value_count"] == 16,
            "missing_bad_low_lane_used_values",
        ):
            return rc
        if rc := require(
            bad_q4_lane_audit["q4_high_nibble_used_value_count"] == 1,
            "missing_bad_high_lane_used_values",
        ):
            return rc
        if rc := require(
            any("Q4_0 high nibble lane used quant values 1 below minimum 4" in finding for finding in bad_q4_lane_audit["findings"]),
            "missing_bad_high_lane_finding",
        ):
            return rc

        bad_q8_unsigned = tmp_path / "q8_bad_unsigned_payload.bin"
        bad_q8_unsigned.write_bytes(struct.pack("<H32b", 0x3C00, *([0] + list(range(1, 32)))))
        bad_q8_unsigned_json = tmp_path / "bad_q8_unsigned_payload_quant_audit.json"
        bad_q8_unsigned_command = [
            sys.executable,
            str(ROOT / "bench" / "quant_audit.py"),
            "--source-root",
            str(source_root),
            "--format",
            "q8_0",
            "--block-file",
            str(bad_q8_unsigned),
            "--min-quant-negative-count",
            "1",
            "--max-quant-sign-balance-delta",
            "8",
            "--output",
            str(bad_q8_unsigned_json),
        ]
        completed = subprocess.run(
            bad_q8_unsigned_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("bad_q8_unsigned_payload_not_rejected=true", file=sys.stderr)
            return 1
        bad_q8_unsigned_report = json.loads(bad_q8_unsigned_json.read_text(encoding="utf-8"))
        bad_q8_unsigned_audit = bad_q8_unsigned_report["block_audits"][0]
        if rc := require(
            bad_q8_unsigned_audit["quant_negative_count"] == 0,
            "unexpected_unsigned_negative_quant_count",
        ):
            return rc
        if rc := require(
            bad_q8_unsigned_audit["quant_positive_count"] == 31,
            "missing_unsigned_positive_quant_count",
        ):
            return rc
        if rc := require(
            any("negative quant payload entries 0 below minimum 1" in finding for finding in bad_q8_unsigned_audit["findings"]),
            "missing_unsigned_quant_finding",
        ):
            return rc
        if rc := require(
            any("quant sign balance delta 31 exceeds limit 8" in finding for finding in bad_q8_unsigned_audit["findings"]),
            "missing_unsigned_quant_balance_finding",
        ):
            return rc

        bad_q8_zero = tmp_path / "q8_bad_zero_payload.bin"
        bad_q8_zero.write_bytes(struct.pack("<H32b", 0x3C00, *([0] * 32)))
        bad_q8_zero_json = tmp_path / "bad_q8_zero_payload_quant_audit.json"
        bad_q8_zero_command = [
            sys.executable,
            str(ROOT / "bench" / "quant_audit.py"),
            "--source-root",
            str(source_root),
            "--format",
            "q8_0",
            "--block-file",
            str(bad_q8_zero),
            "--max-zero-quant-pct",
            "50",
            "--output",
            str(bad_q8_zero_json),
        ]
        completed = subprocess.run(
            bad_q8_zero_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("bad_q8_zero_payload_not_rejected=true", file=sys.stderr)
            return 1
        bad_q8_zero_report = json.loads(bad_q8_zero_json.read_text(encoding="utf-8"))
        bad_q8_zero_audit = bad_q8_zero_report["block_audits"][0]
        if rc := require(bad_q8_zero_audit["quant_zero_pct"] == 100.0, "missing_zero_quant_pct"):
            return rc
        if rc := require(
            any("zero quant values 100.000% exceeds limit 50.000%" in finding for finding in bad_q8_zero_audit["findings"]),
            "missing_zero_quant_finding",
        ):
            return rc

    print("quant_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
