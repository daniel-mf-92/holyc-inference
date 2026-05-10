from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_workload_estimate


def write_jsonl(path: Path) -> None:
    rows = [
        {
            "id": "arc-1",
            "dataset": "arc",
            "split": "validation",
            "question": "Which object conducts electricity?",
            "choices": ["glass", "copper", "paper", "rubber"],
            "answer": "B",
        },
        {
            "id": "arc-2",
            "dataset": "arc",
            "split": "validation",
            "question": "Water freezes at which temperature?",
            "choices": ["0 C", "25 C", "50 C", "100 C"],
            "answer": 0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def parse_args(path: Path, extra: list[str]) -> object:
    return eval_workload_estimate.build_parser().parse_args([str(path), *extra])


def test_estimate_summarizes_scored_tokens_and_launches(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    write_jsonl(source)
    args = parse_args(source, ["--bytes-per-token", "4", "--tok-per-s", "10", "--qemu-launch-overhead-s", "0.5"])

    rows, findings = eval_workload_estimate.load_records([source], args)
    scopes = eval_workload_estimate.summarize(rows, args)

    assert findings == []
    assert len(rows) == 2
    assert scopes[0].scope == "arc:validation"
    assert scopes[0].records == 2
    assert scopes[0].choices == 8
    assert scopes[0].launches_est == 2
    assert scopes[0].scored_tokens_est == sum(row.scored_tokens_est for row in rows)
    assert scopes[0].wall_seconds_est is not None


def test_gate_flags_token_launch_and_wall_budgets(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    write_jsonl(source)
    args = parse_args(
        source,
        [
            "--bytes-per-token",
            "4",
            "--tok-per-s",
            "1",
            "--qemu-launch-overhead-s",
            "1",
            "--max-scored-tokens",
            "1",
            "--max-launches",
            "1",
            "--max-wall-seconds",
            "1",
        ],
    )
    rows, _ = eval_workload_estimate.load_records([source], args)
    findings = eval_workload_estimate.gate(rows, eval_workload_estimate.summarize(rows, args), args)

    assert {"scored_tokens_budget", "launch_budget", "wall_time_budget"} <= {finding.kind for finding in findings}


def test_gate_flags_record_budgets(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    write_jsonl(source)
    args = parse_args(
        source,
        [
            "--bytes-per-token",
            "4",
            "--tok-per-s",
            "1",
            "--qemu-launch-overhead-s",
            "1",
            "--launch-mode",
            "per-choice",
            "--max-choices-per-record",
            "1",
            "--max-record-scored-tokens",
            "1",
            "--max-record-launches",
            "1",
            "--max-record-wall-seconds",
            "1",
        ],
    )
    rows, _ = eval_workload_estimate.load_records([source], args)
    findings = eval_workload_estimate.gate(rows, eval_workload_estimate.summarize(rows, args), args)

    assert {
        "choices_per_record_budget",
        "record_scored_tokens_budget",
        "record_launch_budget",
        "record_wall_time_budget",
    } <= {finding.kind for finding in findings}


def test_empty_input_fails_min_records_gate(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    args = parse_args(source, [])

    rows, _ = eval_workload_estimate.load_records([source], args)
    findings = eval_workload_estimate.gate(rows, eval_workload_estimate.summarize(rows, args), args)

    assert rows == []
    assert [finding.kind for finding in findings] == ["min_records"]


def test_cli_writes_reports(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(source)

    status = eval_workload_estimate.main(
        [
            str(source),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "estimate",
            "--tok-per-s",
            "20",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "estimate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["records"] == 2
    scopes = list(csv.DictReader((output_dir / "estimate.csv").open(encoding="utf-8")))
    assert scopes[0]["scope"] == "arc:validation"
    rows = list(csv.DictReader((output_dir / "estimate_rows.csv").open(encoding="utf-8")))
    assert rows[0]["record_id"] == "arc-1"
    assert "No eval workload budget findings." in (output_dir / "estimate.md").read_text(encoding="utf-8")
    suite = ET.parse(output_dir / "estimate_junit.xml").getroot()
    assert suite.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_estimate_summarizes_scored_tokens_and_launches(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_gate_flags_token_launch_and_wall_budgets(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_gate_flags_record_budgets(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_empty_input_fails_min_records_gate(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_reports(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
