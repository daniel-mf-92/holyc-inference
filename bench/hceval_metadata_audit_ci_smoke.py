#!/usr/bin/env python3
"""CI smoke gate for packed HCEval metadata auditing."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

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
                "id": "metadata-smoke-1",
                "prompt": "Choose the instrument used to measure temperature.",
                "choices": ["thermometer", "ruler", "scale", "compass"],
                "answer_index": 0,
                "provenance": "synthetic metadata smoke",
            }
        ],
    )
    records = dataset_pack.normalize_records(dataset_pack.read_jsonl(source), "metadata-smoke", "validation")
    dataset_pack.write_outputs(records, path, manifest, "metadata-smoke", "validation")


def rewrite_metadata_with_extra_key(path: Path) -> None:
    payload = path.read_bytes()
    magic, version, flags, record_count, metadata_len, source_digest = dataset_pack.HEADER.unpack_from(payload, 0)
    old_body_offset = dataset_pack.HEADER.size + metadata_len
    metadata = {
        "dataset": "metadata-smoke",
        "format": "hceval-mc",
        "note": "non-canonical smoke metadata",
        "record_count": 1,
        "split": "validation",
        "version": dataset_pack.VERSION,
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8")
    header = dataset_pack.HEADER.pack(magic, version, flags, record_count, len(metadata_bytes), source_digest)
    path.write_bytes(header + metadata_bytes + payload[old_body_offset:])


def run_audit(binary: Path, output_dir: Path, stem: str, *, fail_on_findings: bool) -> int:
    return hceval_metadata_audit.main(
        [
            "--input",
            str(binary),
            "--output",
            str(output_dir / f"{stem}.json"),
            "--markdown",
            str(output_dir / f"{stem}.md"),
            "--csv",
            str(output_dir / f"{stem}.csv"),
            "--findings-csv",
            str(output_dir / f"{stem}_findings.csv"),
            "--junit",
            str(output_dir / f"{stem}_junit.xml"),
            *(["--fail-on-findings"] if fail_on_findings else []),
        ]
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "out"
        clean = tmp_path / "clean.hceval"
        drift = tmp_path / "drift.hceval"
        pack_hceval(clean)
        drift.write_bytes(clean.read_bytes())
        rewrite_metadata_with_extra_key(drift)

        clean_status = run_audit(clean, output_dir, "clean", fail_on_findings=True)
        drift_status = run_audit(drift, output_dir, "drift", fail_on_findings=True)
        drift_report = json.loads((output_dir / "drift.json").read_text(encoding="utf-8"))

        if clean_status != 0:
            print("clean_hceval_metadata_smoke=fail")
            return 1
        if drift_status == 0:
            print("drift_hceval_metadata_smoke=fail_missing_findings")
            return 1
        kinds = {finding["kind"] for finding in drift_report["findings"]}
        required = {"metadata_key_drift", "metadata_not_canonical"}
        if not required <= kinds:
            print(f"drift_hceval_metadata_smoke=fail kinds={sorted(kinds)}")
            return 1

    print("hceval_metadata_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
