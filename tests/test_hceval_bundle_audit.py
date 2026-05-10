#!/usr/bin/env python3
"""Coverage for host-side HCEval bundle auditing."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_pack
import hceval_bundle_audit


def write_pack(path: Path, records: list[dataset_pack.EvalRecord], dataset: str = "smoke") -> None:
    dataset_pack.write_outputs(records, path, path.with_suffix(path.suffix + ".manifest.json"), dataset, "validation")


def record(record_id: str, prompt: str = "Pick one.") -> dataset_pack.EvalRecord:
    return dataset_pack.EvalRecord(
        record_id=record_id,
        dataset="smoke",
        split="validation",
        prompt=prompt,
        choices=["alpha", "beta"],
        answer_index=1,
        provenance="synthetic smoke",
    )


def test_bundle_audit_passes_unique_shards(tmp_path: Path) -> None:
    shard_a = tmp_path / "a.hceval"
    shard_b = tmp_path / "b.hceval"
    write_pack(shard_a, [record("a-1")])
    write_pack(shard_b, [record("b-1", "Pick two.")])

    report = hceval_bundle_audit.build_report([shard_a, shard_b], require_manifest=True)

    assert report["status"] == "pass"
    assert report["shard_count"] == 2
    assert report["record_count"] == 2
    assert report["duplicates"] == []


def test_bundle_audit_flags_duplicate_record_ids_and_payloads(tmp_path: Path) -> None:
    shard_a = tmp_path / "a.hceval"
    shard_b = tmp_path / "b.hceval"
    write_pack(shard_a, [record("same")])
    write_pack(shard_b, [record("same")])

    report = hceval_bundle_audit.build_report([shard_a, shard_b], require_manifest=True)

    assert report["status"] == "fail"
    duplicate_kinds = {item["kind"] for item in report["duplicates"]}
    assert duplicate_kinds == {"record_id", "full_payload_sha256"}
    assert any("duplicate record_id same" in finding for finding in report["findings"])


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    shard = tmp_path / "single.hceval"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    csv_path = tmp_path / "audit.csv"
    junit = tmp_path / "audit.xml"
    write_pack(shard, [record("single")])

    status = hceval_bundle_audit.main(
        [
            "--input",
            str(shard),
            "--require-manifest",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--junit",
            str(junit),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    junit_root = ET.parse(junit).getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert "HCEval Bundle Audit" in markdown.read_text(encoding="utf-8")
    assert "single.hceval" in csv_path.read_text(encoding="utf-8")
    assert junit_root.attrib["name"] == "holyc_hceval_bundle_audit"
    assert junit_root.attrib["failures"] == "0"
