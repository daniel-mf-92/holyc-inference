#!/usr/bin/env python3
"""Host-side checks for benchmark matrix config audits."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

AUDIT_PATH = BENCH_PATH / "bench_matrix_audit.py"
spec = importlib.util.spec_from_file_location("bench_matrix_audit", AUDIT_PATH)
bench_matrix_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["bench_matrix_audit"] = bench_matrix_audit
spec.loader.exec_module(bench_matrix_audit)


def write_matrix(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def smoke_matrix() -> dict:
    return {
        "name": "audit-smoke",
        "image": "/tmp/TempleOS.synthetic.img",
        "prompts": str(BENCH_PATH / "prompts" / "smoke.jsonl"),
        "qemu_bin": "bench/fixtures/qemu_synthetic_bench.py",
        "qemu_args": ["-m", "256M"],
        "expect_cells": 2,
        "profiles": [{"name": "ci", "qemu_args": ["-smp", "1"]}],
        "models": [{"name": "synthetic"}],
        "quantizations": [{"name": "Q4_0"}, {"name": "Q8_0"}],
    }


def test_audits_matrix_cell_coverage_and_airgap() -> None:
    with tempfile.TemporaryDirectory(prefix="holyc-bench-matrix-audit-test-") as tmp:
        tmp_path = Path(tmp)
        matrix = tmp_path / "matrix.json"
        write_matrix(matrix, smoke_matrix())
        assert bench_matrix_audit.main([str(matrix), "--output-dir", str(tmp_path), "--output-stem", "audit"]) == 0

        report = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        assert report["status"] == "pass"
        assert report["cell_count"] == 2
        assert report["matrices"][0]["axis_counts"] == {"profiles": 1, "models": 1, "quantizations": 2}
        assert report["matrices"][0]["prompt_suite"]["prompt_count"] == 2
        assert all(cell["command_airgap_ok"] for cell in report["matrices"][0]["cells"])
        assert all(cell["command_has_explicit_nic_none"] for cell in report["matrices"][0]["cells"])


def test_rejects_legacy_net_none_and_missing_quantization() -> None:
    with tempfile.TemporaryDirectory(prefix="holyc-bench-matrix-audit-test-") as tmp:
        tmp_path = Path(tmp)
        payload = smoke_matrix()
        payload["qemu_args"] = ["-net=none"]
        matrix = tmp_path / "matrix.json"
        write_matrix(matrix, payload)

        status = bench_matrix_audit.main(
            [
                str(matrix),
                "--output-dir",
                str(tmp_path),
                "--output-stem",
                "audit",
                "--require-quantization",
                "Q2_K",
            ]
        )

        report = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
        messages = [finding["message"] for matrix_report in report["matrices"] for finding in matrix_report["findings"]]
        assert status == 2
        assert report["status"] == "fail"
        assert any("legacy `-net=none`" in message for message in messages)
        assert any("missing required quantization 'Q2_K'" in message for message in messages)


def test_cli_writes_sidecars() -> None:
    with tempfile.TemporaryDirectory(prefix="holyc-bench-matrix-audit-test-") as tmp:
        tmp_path = Path(tmp)
        matrix = tmp_path / "matrix.json"
        write_matrix(matrix, smoke_matrix())

        assert bench_matrix_audit.main([str(matrix), "--output-dir", str(tmp_path), "--output-stem", "audit"]) == 0

        rows = list(csv.DictReader((tmp_path / "audit.csv").open(newline="", encoding="utf-8")))
        assert len([row for row in rows if row["row_type"] == "cell"]) == 2
        assert "Benchmark Matrix Audit" in (tmp_path / "audit.md").read_text(encoding="utf-8")
        junit = ET.parse(tmp_path / "audit_junit.xml").getroot()
        assert junit.attrib["name"] == "holyc_bench_matrix_audit"
        assert junit.attrib["failures"] == "0"


if __name__ == "__main__":
    test_audits_matrix_cell_coverage_and_airgap()
    test_rejects_legacy_net_none_and_missing_quantization()
    test_cli_writes_sidecars()
    print("bench_matrix_audit_tests=ok")
