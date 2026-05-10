#!/usr/bin/env python3
"""Host-side checks for HCEval export/repack roundtrip auditing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
sys.path.insert(0, str(BENCH))

import dataset_pack

AUDIT_PATH = BENCH / "hceval_export_roundtrip_audit.py"
spec = importlib.util.spec_from_file_location("hceval_export_roundtrip_audit", AUDIT_PATH)
hceval_export_roundtrip_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["hceval_export_roundtrip_audit"] = hceval_export_roundtrip_audit
spec.loader.exec_module(hceval_export_roundtrip_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def pack_rows(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jsonl"
    binary = tmp_path / "source.hceval"
    manifest = tmp_path / "source.hceval.manifest.json"
    write_jsonl(source, rows)
    records = dataset_pack.normalize_records(dataset_pack.read_jsonl(source), "mixed", "validation")
    dataset_pack.write_outputs(records, binary, manifest, "mixed", "validation")
    return binary, manifest


def test_export_roundtrip_passes_with_pack_manifest(tmp_path: Path) -> None:
    binary, manifest = pack_rows(
        tmp_path,
        [
            {
                "id": "arc-1",
                "dataset": "arc",
                "split": "validation",
                "prompt": "Pick B.",
                "choices": ["A", "B"],
                "answer_index": 1,
                "provenance": "synthetic arc",
            }
        ],
    )

    report = hceval_export_roundtrip_audit.build_report(binary, manifest)

    assert report["status"] == "pass"
    assert report["binary_sha256"] == report["repacked_binary_sha256"]
    assert report["source_sha256"] == report["repacked_source_sha256"]


def test_export_roundtrip_flags_missing_manifest_for_mixed_dataset_binary(tmp_path: Path) -> None:
    binary, _manifest = pack_rows(
        tmp_path,
        [
            {
                "id": "arc-1",
                "dataset": "arc",
                "split": "validation",
                "prompt": "Pick B.",
                "choices": ["A", "B"],
                "answer_index": 1,
                "provenance": "synthetic arc",
            },
            {
                "id": "truth-1",
                "dataset": "truthfulqa",
                "split": "validation",
                "prompt": "Pick evidence.",
                "choices": ["claim", "evidence"],
                "answer_index": 1,
                "provenance": "synthetic truthfulqa",
            },
        ],
    )

    report = hceval_export_roundtrip_audit.build_report(binary, None)

    assert report["status"] == "fail"
    assert any(finding["kind"] == "source_digest_mismatch" for finding in report["findings"])


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_export_roundtrip_passes_with_pack_manifest(Path(tmp) / "pass")
    with tempfile.TemporaryDirectory() as tmp:
        test_export_roundtrip_flags_missing_manifest_for_mixed_dataset_binary(Path(tmp) / "fail")
    print("hceval_export_roundtrip_audit_tests=ok")
