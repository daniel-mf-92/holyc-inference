#!/usr/bin/env python3
"""Tests for QEMU benchmark matrix planning."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_benchmark_matrix


def write_inputs(tmp_path: Path, *, qemu_args: list[str] | None = None) -> Path:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            [
                json.dumps({"id": "alpha", "prompt": "Alpha?", "expected_tokens": 4}),
                json.dumps({"id": "beta", "prompt": "Beta?", "expected_tokens": 5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "prompts": str(prompts),
                "profile": "unit",
                "model": "smoke-model",
                "quantization": "Q8_0",
                "warmup": 1,
                "repeat": 2,
                "builds": [
                    {"build": "base", "image": "base.img", "qemu_args": qemu_args or ["-m", "512M"]},
                    {"build": "cand", "image": "cand.img", "qemu_args": ["-m", "512M"]},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix


def test_matrix_report_expands_builds_and_launches(tmp_path: Path) -> None:
    matrix = write_inputs(tmp_path)

    report = qemu_benchmark_matrix.build_matrix_report(matrix, "matrix")

    assert report["status"] == "pass"
    assert report["summary"]["builds"] == 2
    assert report["summary"]["prompts"] == 2
    assert report["summary"]["launches"] == 12
    assert report["summary"]["airgap_ok_builds"] == 2
    assert report["builds"][0]["command"][1:3] == ["-nic", "none"]
    assert report["launches"][0]["build"] == "base"
    assert report["launches"][0]["phase"] == "warmup"
    assert report["launches"][-1]["build"] == "cand"
    assert report["launches"][-1]["phase"] == "measured"


def test_matrix_rejects_network_qemu_args(tmp_path: Path) -> None:
    matrix = write_inputs(tmp_path, qemu_args=["-netdev", "user,id=n0"])

    try:
        qemu_benchmark_matrix.build_matrix_report(matrix, "matrix")
    except ValueError as exc:
        assert "-netdev is not allowed" in str(exc)
    else:
        raise AssertionError("expected air-gap rejection")


def test_cli_writes_matrix_artifacts(tmp_path: Path) -> None:
    matrix = write_inputs(tmp_path)
    output_dir = tmp_path / "out"

    status = qemu_benchmark_matrix.main([str(matrix), "--output-dir", str(output_dir), "--output-stem", "matrix"])

    assert status == 0
    payload = json.loads((output_dir / "matrix.json").read_text(encoding="utf-8"))
    assert payload["summary"]["builds"] == 2
    rows = list(csv.DictReader((output_dir / "matrix.csv").open(encoding="utf-8")))
    assert rows[0]["command_airgap_ok"] == "True"
    launch_rows = list(csv.DictReader((output_dir / "matrix_launches.csv").open(encoding="utf-8")))
    assert len(launch_rows) == 12
    commands = [json.loads(line) for line in (output_dir / "matrix_commands.jsonl").read_text(encoding="utf-8").splitlines()]
    assert commands[0]["argv"][1:3] == ["-nic", "none"]
    assert "QEMU Benchmark Matrix" in (output_dir / "matrix.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "matrix_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_benchmark_matrix"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_matrix_report_expands_builds_and_launches(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_matrix_rejects_network_qemu_args(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_matrix_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
