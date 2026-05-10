from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import perplexity_pairing_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_audit_accepts_paired_ids_metadata_and_token_counts(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(
        holyc,
        [
            {"id": "smoke-1", "token_logprobs": [-0.1, -0.2], "dataset": "arc", "split": "validation"},
            {"id": "smoke-2", "token_count": 3, "total_nll": 1.2, "dataset": "hellaswag", "split": "validation"},
        ],
    )
    write_jsonl(
        llama,
        [
            {"id": "smoke-1", "token_logprobs": [-0.2, -0.3], "dataset": "arc", "split": "validation"},
            {"id": "smoke-2", "token_count": 3, "perplexity": 1.5, "dataset": "hellaswag", "split": "validation"},
        ],
    )

    holyc_records, holyc_findings = perplexity_pairing_audit.load_records(holyc)
    llama_records, llama_findings = perplexity_pairing_audit.load_records(llama)
    pairs, findings = perplexity_pairing_audit.audit_pairing(holyc_records, llama_records, min_pairs=2)

    assert holyc_findings == []
    assert llama_findings == []
    assert findings == []
    assert len(pairs) == 2
    assert all(pair.token_count_match for pair in pairs)


def test_audit_rejects_missing_duplicate_token_and_metadata_mismatches(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    write_jsonl(
        holyc,
        [
            {"id": "dup", "token_count": 2, "total_nll": 1.0, "dataset": "arc", "split": "validation"},
            {"id": "dup", "token_count": 2, "total_nll": 1.1, "dataset": "arc", "split": "validation"},
            {"id": "only-holyc", "token_count": 1, "total_nll": 0.1},
        ],
    )
    write_jsonl(
        llama,
        [
            {"id": "dup", "token_count": 3, "total_nll": 1.2, "dataset": "arc", "split": "test"},
            {"id": "only-llama", "token_count": 1, "total_nll": 0.2},
        ],
    )

    pairs, findings = perplexity_pairing_audit.audit_pairing(
        perplexity_pairing_audit.load_records(holyc)[0],
        perplexity_pairing_audit.load_records(llama)[0],
        min_pairs=3,
    )
    kinds = {finding.kind for finding in findings}

    assert len(pairs) == 1
    assert {"duplicate_record_id", "missing_llama_record", "missing_holyc_record", "token_count_mismatch", "split_mismatch", "min_pairs"} <= kinds


def test_cli_writes_json_csv_markdown_and_junit(tmp_path: Path) -> None:
    holyc = tmp_path / "holyc.jsonl"
    llama = tmp_path / "llama.jsonl"
    out = tmp_path / "out"
    write_jsonl(holyc, [{"id": "a", "token_logprobs": [-0.1], "dataset": "arc", "split": "validation"}])
    write_jsonl(llama, [{"id": "a", "token_logprobs": [-0.2], "dataset": "arc", "split": "validation"}])

    status = perplexity_pairing_audit.main(["--holyc", str(holyc), "--llama", str(llama), "--output-dir", str(out), "--output-stem", "pairing"])

    assert status == 0
    payload = json.loads((out / "pairing.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["paired_rows"] == 1
    assert "No perplexity pairing findings." in (out / "pairing.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((out / "pairing.csv").open(encoding="utf-8")))
    assert rows[0]["record_id"] == "a"
    finding_rows = list(csv.DictReader((out / "pairing_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit = ET.parse(out / "pairing_junit.xml").getroot()
    assert junit.attrib["name"] == "holyc_perplexity_pairing_audit"
    assert junit.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_paired_ids_metadata_and_token_counts(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_rejects_missing_duplicate_token_and_metadata_mismatches(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_csv_markdown_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
