#!/usr/bin/env python3
"""Host-side checks for eval dataset manifest audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_manifest_audit.py"
spec = importlib.util.spec_from_file_location("dataset_manifest_audit", AUDIT_PATH)
dataset_manifest_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_manifest_audit"] = dataset_manifest_audit
spec.loader.exec_module(dataset_manifest_audit)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_manifest_digest_and_pack_consistency_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.jsonl"
        curated = root / "curated.jsonl"
        packed = root / "curated.hceval"
        source.write_text('{"id":"a"}\n', encoding="utf-8")
        curated.write_text('{"record_id":"unit-1","dataset":"unit"}\n', encoding="utf-8")
        packed.write_bytes(b"HCEVAL1\0unit")

        manifest = {
            "format": "hceval-curated-jsonl",
            "source_name": "unit",
            "source_version": "synthetic",
            "license": "synthetic",
            "source": {"path": "source.jsonl", "sha256": dataset_manifest_audit.file_sha256(source)},
            "output": "curated.jsonl",
            "normalized_sha256": dataset_manifest_audit.canonical_jsonl_sha256(curated),
            "record_count": 1,
            "selected_record_ids": ["unit-1"],
            "pack_output": "curated.hceval",
        }
        pack_manifest = {
            "format": "hceval-mc",
            "output": "curated.hceval",
            "binary_sha256": dataset_manifest_audit.file_sha256(packed),
            "source_sha256": manifest["normalized_sha256"],
            "record_count": 1,
            "records": [{"record_id": "unit-1"}],
        }
        manifest_path = root / "manifest.json"
        pack_manifest_path = root / "curated.hceval.manifest.json"
        output = root / "audit.json"
        write_json(manifest_path, manifest)
        write_json(pack_manifest_path, pack_manifest)

        status = dataset_manifest_audit.main(
            [
                "--manifest",
                str(manifest_path),
                "--pack-manifest",
                str(pack_manifest_path),
                "--root",
                str(root),
                "--output",
                str(output),
                "--require-pack-manifest",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["findings"] == []
        assert report["curated_output"]["actual_record_count"] == 1
        assert report["curated_output"]["actual_record_ids"] == ["unit-1"]
        assert report["curated_output"]["actual_dataset_counts"] == {"unit": 1}
        assert report["pack_manifest"]["record_count"] == 1


def test_manifest_gates_report_digest_and_pack_mismatches() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.jsonl"
        curated = root / "curated.jsonl"
        packed = root / "curated.hceval"
        source.write_text('{"id":"a"}\n', encoding="utf-8")
        curated.write_text('{"id":"a"}\n', encoding="utf-8")
        packed.write_bytes(b"actual")

        manifest_path = root / "manifest.json"
        pack_manifest_path = root / "pack.json"
        output = root / "audit.json"
        write_json(
            manifest_path,
            {
                "format": "hceval-curated-jsonl",
                "source_name": "unit",
                "source_version": "synthetic",
                "license": "synthetic",
                "source": {"path": "source.jsonl", "sha256": "bad-source"},
                "output": "curated.jsonl",
                "normalized_sha256": "bad-curated",
                "record_count": 2,
                "dataset_counts": {"wrong": 2},
                "split_counts": {"wrong": 2},
                "dataset_split_counts": {"wrong": {"wrong": 2}},
                "selected_record_ids": ["a", "b"],
                "pack_output": "curated.hceval",
            },
        )
        write_json(
            pack_manifest_path,
            {
                "format": "hceval-mc",
                "output": "curated.hceval",
                "binary_sha256": "bad-pack",
                "source_sha256": "different",
                "record_count": 1,
                "records": [{"record_id": "b"}, {"record_id": "a"}],
            },
        )

        status = dataset_manifest_audit.main(
            [
                "--manifest",
                str(manifest_path),
                "--pack-manifest",
                str(pack_manifest_path),
                "--root",
                str(root),
                "--output",
                str(output),
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert {
            "source_sha256_mismatch",
            "curated_output_normalized_sha256_mismatch",
            "curated_record_count_mismatch",
            "curated_dataset_counts_mismatch",
            "curated_split_counts_mismatch",
            "curated_dataset_split_counts_mismatch",
            "curated_record_id_order_mismatch",
            "pack_output_sha256_mismatch",
            "pack_source_sha256_mismatch",
            "pack_record_count_mismatch",
            "pack_record_id_order_mismatch",
        }.issubset(kinds)


if __name__ == "__main__":
    test_manifest_digest_and_pack_consistency_pass()
    test_manifest_gates_report_digest_and_pack_mismatches()
    print("dataset_manifest_audit_tests=ok")
