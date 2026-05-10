from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_significance_audit


def report(delta: float, holyc_only: int, llama_only: int, p_value: float) -> dict[str, object]:
    scope = {
        "record_count": 10,
        "holyc_accuracy": 0.5 + delta,
        "llama_accuracy": 0.5,
        "accuracy_delta_holyc_minus_llama": delta,
        "mcnemar_exact": {
            "holyc_only_correct": holyc_only,
            "llama_only_correct": llama_only,
            "discordant_count": holyc_only + llama_only,
            "p_value": p_value,
            "method": "exact_binomial_two_sided",
        },
    }
    return {
        "dataset": "smoke-eval",
        "split": "validation",
        "summary": {
            **scope,
            "dataset_breakdown": [{**scope, "dataset": "arc-smoke", "split": "validation"}],
        },
    }


def test_audit_extracts_overall_and_dataset_split_mcnemar_rows(tmp_path: Path) -> None:
    path = tmp_path / "eval_compare_smoke.json"
    path.write_text(json.dumps(report(0.0, 2, 2, 1.0)) + "\n", encoding="utf-8")

    payload = eval_significance_audit.audit([path], patterns=["*.json"], min_reports=1, max_holyc_loss_p=0.05)

    assert payload["status"] == "pass"
    assert payload["summary"]["scope_count"] == 2
    assert payload["scopes"][0]["scope"] == "overall"
    assert payload["scopes"][1]["dataset"] == "arc-smoke"
    assert payload["scopes"][0]["discordant_count"] == 4
    assert payload["findings"] == []


def test_audit_rejects_significant_paired_holyc_loss(tmp_path: Path) -> None:
    path = tmp_path / "eval_compare_loss.json"
    path.write_text(json.dumps(report(-0.3, 0, 6, 0.0625)) + "\n", encoding="utf-8")

    payload = eval_significance_audit.audit([path], patterns=["*.json"], min_reports=1, max_holyc_loss_p=0.1)

    assert payload["status"] == "fail"
    assert payload["summary"]["significant_holyc_loss_count"] == 2
    assert {finding["kind"] for finding in payload["findings"]} == {"significant_holyc_loss"}


def test_cli_writes_json_csv_markdown_findings_and_junit(tmp_path: Path) -> None:
    path = tmp_path / "eval_compare_smoke.json"
    out = tmp_path / "out"
    path.write_text(json.dumps(report(0.1, 3, 1, 0.625)) + "\n", encoding="utf-8")

    status = eval_significance_audit.main(
        [
            str(path),
            "--output-dir",
            str(out),
            "--output-stem",
            "significance",
            "--max-holyc-loss-p",
            "0.05",
        ]
    )

    assert status == 0
    payload = json.loads((out / "significance.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((out / "significance.csv").open(encoding="utf-8")))
    findings = list(csv.DictReader((out / "significance_findings.csv").open(encoding="utf-8")))
    junit = ET.parse(out / "significance_junit.xml").getroot()
    assert payload["status"] == "pass"
    assert rows[0]["p_value"] == "0.625"
    assert findings == []
    assert "No significance findings." in (out / "significance.md").read_text(encoding="utf-8")
    assert junit.attrib["name"] == "holyc_eval_significance_audit"
    assert junit.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_extracts_overall_and_dataset_split_mcnemar_rows(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_rejects_significant_paired_holyc_loss(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_csv_markdown_findings_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
