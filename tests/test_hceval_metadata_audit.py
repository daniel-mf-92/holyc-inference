#!/usr/bin/env python3
"""Host-side checks for HCEval metadata audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

import dataset_pack
import hceval_metadata_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def pack_hceval(path: Path) -> None:
    source = path.with_suffix(".jsonl")
    manifest = Path(str(path) + ".manifest.json")
    write_jsonl(
        source,
        [
            {
                "id": "meta-1",
                "prompt": "Pick the tool used for measuring temperature.",
                "choices": ["thermometer", "ruler", "scale", "compass"],
                "answer_index": 0,
                "provenance": "unit metadata audit",
            }
        ],
    )
    records = dataset_pack.normalize_records(dataset_pack.read_jsonl(source), "unit", "validation")
    dataset_pack.write_outputs(records, path, manifest, "unit", "validation")


def rewrite_metadata(path: Path, metadata: dict[str, object]) -> None:
    payload = path.read_bytes()
    magic, version, flags, record_count, metadata_len, source_digest = dataset_pack.HEADER.unpack_from(payload, 0)
    old_body_offset = dataset_pack.HEADER.size + metadata_len
    new_metadata = json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8")
    header = dataset_pack.HEADER.pack(magic, version, flags, record_count, len(new_metadata), source_digest)
    path.write_bytes(header + new_metadata + payload[old_body_offset:])


def parse_args(path: Path, extra: list[str] | None = None) -> object:
    return hceval_metadata_audit.build_parser().parse_args(["--input", str(path), *(extra or [])])


def test_build_report_passes_canonical_metadata(tmp_path: Path) -> None:
    binary = tmp_path / "clean.hceval"
    pack_hceval(binary)

    report = hceval_metadata_audit.build_report(parse_args(binary))

    assert report["status"] == "pass"
    assert report["input_count"] == 1
    assert report["rows"][0]["dataset"] == "unit"
    assert report["rows"][0]["parsed_records"] == 1
    assert report["findings"] == []


def test_build_report_flags_metadata_key_and_canonical_drift(tmp_path: Path) -> None:
    binary = tmp_path / "drift.hceval"
    pack_hceval(binary)
    rewrite_metadata(
        binary,
        {
            "dataset": "unit",
            "format": "hceval-mc",
            "record_count": 1,
            "split": "validation",
            "version": dataset_pack.VERSION,
            "note": "not canonical",
        },
    )

    report = hceval_metadata_audit.build_report(parse_args(binary))
    kinds = {finding["kind"] for finding in report["findings"]}

    assert report["status"] == "fail"
    assert {"metadata_key_drift", "metadata_not_canonical"} <= kinds
    assert report["rows"][0]["finding_count"] == 2


def test_cli_writes_metadata_audit_artifacts(tmp_path: Path) -> None:
    binary = tmp_path / "clean.hceval"
    output_dir = tmp_path / "out"
    pack_hceval(binary)

    status = hceval_metadata_audit.main(
        [
            "--input",
            str(binary),
            "--output",
            str(output_dir / "metadata.json"),
            "--markdown",
            str(output_dir / "metadata.md"),
            "--csv",
            str(output_dir / "metadata.csv"),
            "--findings-csv",
            str(output_dir / "metadata_findings.csv"),
            "--junit",
            str(output_dir / "metadata_junit.xml"),
            "--fail-on-findings",
        ]
    )

    payload = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "metadata.csv").open(encoding="utf-8")))
    findings = list(csv.DictReader((output_dir / "metadata_findings.csv").open(encoding="utf-8")))
    junit_root = ET.parse(output_dir / "metadata_junit.xml").getroot()

    assert status == 0
    assert payload["status"] == "pass"
    assert len(rows) == 1
    assert findings == []
    assert "HCEval Metadata Audit" in (output_dir / "metadata.md").read_text(encoding="utf-8")
    assert junit_root.attrib["name"] == "holyc_hceval_metadata_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_build_report_passes_canonical_metadata(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_build_report_flags_metadata_key_and_canonical_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_metadata_audit_artifacts(Path(tmp))
    print("hceval_metadata_audit_tests=ok")
