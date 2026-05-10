#!/usr/bin/env python3
"""Host-side checks for HCEval suite audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
AUDIT_PATH = BENCH / "hceval_suite_audit.py"
PACK_PATH = BENCH / "dataset_pack.py"

pack_spec = importlib.util.spec_from_file_location("dataset_pack", PACK_PATH)
dataset_pack = importlib.util.module_from_spec(pack_spec)
assert pack_spec and pack_spec.loader
sys.modules["dataset_pack"] = dataset_pack
pack_spec.loader.exec_module(dataset_pack)

audit_spec = importlib.util.spec_from_file_location("hceval_suite_audit", AUDIT_PATH)
hceval_suite_audit = importlib.util.module_from_spec(audit_spec)
assert audit_spec and audit_spec.loader
sys.modules["hceval_suite_audit"] = hceval_suite_audit
audit_spec.loader.exec_module(hceval_suite_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def pack(path: Path, source: Path, manifest: Path) -> None:
    rows = dataset_pack.read_jsonl(source)
    records = dataset_pack.normalize_records(rows, "unit", "validation")
    dataset_pack.write_outputs(records, path, manifest, "unit", "validation")


def test_suite_audit_passes_with_manifest_and_budgets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "rows.jsonl"
        binary = root / "rows.hceval"
        manifest = root / "rows.hceval.manifest.json"
        output = root / "suite.json"
        write_jsonl(
            source,
            [
                {
                    "id": "unit-1",
                    "prompt": "Pick B.",
                    "choices": ["A", "B"],
                    "answer_index": 1,
                    "provenance": "synthetic",
                }
            ],
        )
        pack(binary, source, manifest)

        status = hceval_suite_audit.main(
            [
                "--input",
                str(root),
                "--require-manifest",
                "--max-prompt-bytes",
                "32",
                "--max-choice-bytes",
                "8",
                "--max-record-payload-bytes",
                "128",
                "--output",
                str(output),
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["input_count"] == 1
        assert report["record_count"] == 1
        assert report["dataset_counts"] == {"unit": 1}
        assert report["rows"][0]["manifest"] == str(manifest)


def test_suite_audit_reports_missing_manifest_and_budget_failures() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "rows.jsonl"
        binary = root / "rows.hceval"
        manifest = root / "rows.hceval.manifest.json"
        output = root / "suite.json"
        write_jsonl(
            source,
            [
                {
                    "id": "unit-1",
                    "prompt": "This prompt is intentionally longer than the tiny budget.",
                    "choices": ["A", "B"],
                    "answer_index": 1,
                    "provenance": "synthetic",
                }
            ],
        )
        pack(binary, source, manifest)
        manifest.unlink()

        status = hceval_suite_audit.main(
            [
                "--input",
                str(binary),
                "--require-manifest",
                "--max-prompt-bytes",
                "8",
                "--output",
                str(output),
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        details = "\n".join(finding["detail"] for finding in report["findings"])
        assert status == 1
        assert report["status"] == "fail"
        assert "missing_manifest" in kinds
        assert "prompt is" in details


if __name__ == "__main__":
    test_suite_audit_passes_with_manifest_and_budgets()
    test_suite_audit_reports_missing_manifest_and_budget_failures()
    print("hceval_suite_audit_tests=ok")
