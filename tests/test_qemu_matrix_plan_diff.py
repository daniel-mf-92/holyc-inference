from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_matrix_plan_diff


def write_matrix(path: Path, *, command_hash: str = "cmd-a", launch_count: int = 2) -> None:
    launches = [
        {
            "build": "base",
            "profile": "unit",
            "model": "toy",
            "quantization": "Q4_0",
            "phase": "measured",
            "prompt_id": "alpha",
            "prompt_sha256": "prompt-alpha",
            "iteration": iteration,
        }
        for iteration in range(launch_count)
    ]
    payload = {
        "builds": [
            {
                "build": "base",
                "profile": "unit",
                "model": "toy",
                "quantization": "Q4_0",
                "command_sha256": command_hash,
                "launch_plan_sha256": f"plan-{launch_count}",
                "launch_count": launch_count,
            }
        ],
        "launches": launches,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_compare_identical_matrix_plans_passes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    write_matrix(baseline)
    write_matrix(candidate)

    report = qemu_matrix_plan_diff.compare_matrix_plans(baseline, candidate)

    assert report["status"] == "pass"
    assert report["summary"]["changed_build_rows"] == 0
    assert report["summary"]["changed_launch_rows"] == 0
    assert report["findings"] == []


def test_compare_matrix_plans_reports_launch_and_command_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    write_matrix(baseline)
    write_matrix(candidate, command_hash="cmd-b", launch_count=3)

    report = qemu_matrix_plan_diff.compare_matrix_plans(baseline, candidate)
    kinds = {finding["kind"] for finding in report["findings"]}

    assert report["status"] == "fail"
    assert "build_command_hash_drift" in kinds
    assert "build_launch_count_drift" in kinds
    assert "extra_launch" in kinds
    assert report["summary"]["changed_launch_rows"] == 1


def test_cli_writes_plan_diff_artifacts(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "out"
    write_matrix(baseline)
    write_matrix(candidate)

    status = qemu_matrix_plan_diff.main(
        [str(baseline), str(candidate), "--output-dir", str(output_dir), "--output-stem", "plan_diff"]
    )

    assert status == 0
    payload = json.loads((output_dir / "plan_diff.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    rows = list(csv.DictReader((output_dir / "plan_diff.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "pass"
    launch_rows = list(csv.DictReader((output_dir / "plan_diff_launches.csv").open(encoding="utf-8")))
    assert len(launch_rows) == 2
    assert "QEMU Matrix Plan Diff" in (output_dir / "plan_diff.md").read_text(encoding="utf-8")
    junit = ET.parse(output_dir / "plan_diff_junit.xml").getroot()
    assert junit.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_compare_identical_matrix_plans_passes(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_compare_matrix_plans_reports_launch_and_command_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_plan_diff_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
