#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for host-side eval dataset tooling."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-ci-") as tmp:
        tmp_path = Path(tmp)
        datasets_dir = tmp_path / "datasets"
        curated_source = datasets_dir / "smoke_eval_with_duplicate.jsonl"
        curated = datasets_dir / "smoke_curated.jsonl"
        curated_manifest = datasets_dir / "smoke_curated.manifest.json"
        packed = datasets_dir / "smoke_curated.hceval"
        pack_manifest = datasets_dir / "smoke_curated.hceval.manifest.json"
        schema_json = datasets_dir / "dataset_schema_audit_smoke_latest.json"
        schema_md = datasets_dir / "dataset_schema_audit_smoke_latest.md"
        schema_csv = datasets_dir / "dataset_schema_audit_smoke_latest.csv"
        schema_junit = datasets_dir / "dataset_schema_audit_smoke_latest_junit.xml"
        inspect_json = datasets_dir / "smoke_curated.inspect.json"
        inspect_md = datasets_dir / "smoke_curated.inspect.md"
        inspect_junit = datasets_dir / "smoke_curated.inspect.junit.xml"
        leak_json = datasets_dir / "dataset_leak_audit_smoke_latest.json"
        leak_md = datasets_dir / "dataset_leak_audit_smoke_latest.md"
        leak_csv = datasets_dir / "dataset_leak_audit_smoke_latest.csv"
        leak_junit = datasets_dir / "dataset_leak_audit_smoke_latest_junit.xml"
        provenance_json = datasets_dir / "dataset_provenance_audit_latest.json"
        provenance_md = datasets_dir / "dataset_provenance_audit_latest.md"
        provenance_csv = datasets_dir / "dataset_provenance_audit_latest.csv"
        provenance_junit = datasets_dir / "dataset_provenance_audit_junit_latest.xml"

        sample_lines = SAMPLE.read_text(encoding="utf-8").splitlines()
        curated_source.parent.mkdir(parents=True, exist_ok=True)
        curated_source.write_text("\n".join(sample_lines + [sample_lines[0]]) + "\n", encoding="utf-8")

        schema_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_schema_audit.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(schema_json),
            "--markdown",
            str(schema_md),
            "--csv",
            str(schema_csv),
            "--junit",
            str(schema_junit),
            "--require-provenance",
            "--min-choices",
            "4",
            "--max-choices",
            "4",
            "--max-prompt-bytes",
            "4096",
            "--max-choice-bytes",
            "1024",
            "--max-record-payload-bytes",
            "8192",
            "--max-majority-answer-pct",
            "100",
            "--max-dataset-split-majority-answer-pct",
            "100",
            "--fail-on-duplicate-ids",
            "--fail-on-duplicate-payloads",
            "--fail-on-conflicting-payload-answers",
            "--fail-on-findings",
        ]
        completed = run_command(schema_command)
        if completed.returncode != 0:
            return completed.returncode

        schema_report = json.loads(schema_json.read_text(encoding="utf-8"))
        if rc := require(schema_report["status"] == "pass", "unexpected_schema_status"):
            return rc
        if rc := require(schema_report["normalized_record_count"] == 3, "unexpected_schema_record_count"):
            return rc
        if rc := require(schema_report["choice_count_histogram"] == {"4": 3}, "unexpected_schema_choice_histogram"):
            return rc
        if rc := require(schema_report["answer_histogram"] == {"0": 3}, "unexpected_schema_answer_histogram"):
            return rc
        if rc := require(
            schema_report["majority_answer"] == {"answer_index": "0", "pct": 100.0, "records": 3},
            "unexpected_schema_majority_answer",
        ):
            return rc
        if rc := require(
            schema_report["dataset_split_answer_histograms"] == {
                "arc-smoke": {"validation": {"0": 1}},
                "hellaswag-smoke": {"validation": {"0": 1}},
                "truthfulqa-smoke": {"validation": {"0": 1}},
            },
            "unexpected_schema_dataset_split_answer_histograms",
        ):
            return rc
        if rc := require(schema_report["duplicate_payloads"] == [], "unexpected_schema_duplicate_payloads"):
            return rc
        if rc := require(
            "Eval Dataset Schema Audit" in schema_md.read_text(encoding="utf-8"),
            "missing_schema_markdown",
        ):
            return rc
        if rc := require(
            "severity,kind,source,detail" in schema_csv.read_text(encoding="utf-8"),
            "missing_schema_csv_header",
        ):
            return rc
        schema_root = ET.parse(schema_junit).getroot()
        if rc := require(schema_root.attrib.get("name") == "holyc_dataset_schema_audit", "missing_schema_junit"):
            return rc
        if rc := require(schema_root.attrib.get("failures") == "0", "unexpected_schema_junit_failures"):
            return rc

        curate_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_curate.py"),
            "--input",
            str(curated_source),
            "--output",
            str(curated),
            "--manifest",
            str(curated_manifest),
            "--source-name",
            "smoke-eval",
            "--source-version",
            "synthetic",
            "--source-license",
            "synthetic-smoke",
            "--min-choices",
            "4",
            "--max-choices",
            "4",
            "--max-records",
            "3",
            "--max-records-per-dataset-split",
            "1",
            "--max-records-per-provenance",
            "1",
            "--dedupe-within-split-payloads",
            "--max-prompt-bytes",
            "4096",
            "--max-choice-bytes",
            "1024",
            "--max-record-payload-bytes",
            "8192",
            "--balance-answer-index",
            "--pack-output",
            str(packed),
            "--pack-manifest",
            str(pack_manifest),
            "--pack-dataset",
            "smoke-eval",
            "--pack-split",
            "validation",
        ]
        completed = run_command(curate_command)
        if completed.returncode != 0:
            return completed.returncode

        curated_report = json.loads(curated_manifest.read_text(encoding="utf-8"))
        if rc := require(curated_report["source"]["record_count"] == 4, "unexpected_curated_source_count"):
            return rc
        if rc := require(curated_report["total_after_deduplication"] == 3, "unexpected_deduped_record_count"):
            return rc
        if rc := require(curated_report["record_count"] == 3, "unexpected_curated_record_count"):
            return rc
        pack_report = json.loads(pack_manifest.read_text(encoding="utf-8"))
        if rc := require(len(pack_report["record_spans"]) == 3, "unexpected_pack_span_count"):
            return rc
        first_span = pack_report["record_spans"][0]
        if rc := require(first_span["record_index"] == 0, "unexpected_pack_span_index"):
            return rc
        if rc := require(first_span["offset"] >= 50, "unexpected_pack_span_offset"):
            return rc
        if rc := require(first_span["length"] > first_span["payload_bytes"], "unexpected_pack_span_length"):
            return rc
        if rc := require(first_span["choice_count"] == 4, "unexpected_pack_span_choice_count"):
            return rc
        if rc := require(len(first_span["choice_bytes"]) == 4, "unexpected_pack_span_choice_bytes"):
            return rc
        if rc := require(
            first_span["choice_bytes_total"] == sum(first_span["choice_bytes"]),
            "unexpected_pack_span_choice_total",
        ):
            return rc
        if rc := require(first_span["prompt_bytes"] > 0, "unexpected_pack_span_prompt_bytes"):
            return rc
        if rc := require(first_span["provenance_bytes"] > 0, "unexpected_pack_span_provenance_bytes"):
            return rc
        if rc := require(first_span["answer_index"] == 0, "unexpected_pack_span_answer_index"):
            return rc
        if rc := require(
            curated_report["answer_histogram"] == {"0": 3},
            "unexpected_curated_answer_histogram",
        ):
            return rc
        if rc := require(
            curated_report["dataset_answer_histograms"] == {
                "arc-smoke": {"0": 1},
                "hellaswag-smoke": {"0": 1},
                "truthfulqa-smoke": {"0": 1},
            },
            "unexpected_curated_dataset_answer_histograms",
        ):
            return rc
        if rc := require(
            curated_report["split_answer_histograms"] == {"validation": {"0": 3}},
            "unexpected_curated_split_answer_histograms",
        ):
            return rc
        if rc := require(
            curated_report["provenance_counts"] == {
                "synthetic ARC-shaped smoke row": 1,
                "synthetic HellaSwag-shaped smoke row": 1,
                "synthetic TruthfulQA-shaped smoke row": 1,
            },
            "unexpected_curated_provenance_counts",
        ):
            return rc
        if rc := require(curated_report["filters"]["balance_answer_index"] is True, "missing_balance_flag"):
            return rc
        if rc := require(
            curated_report["filters"]["dedupe_within_split_payloads"] is True,
            "missing_dedupe_payload_flag",
        ):
            return rc
        if rc := require(
            curated_report["filters"]["max_records_per_dataset_split"] == 1,
            "missing_dataset_split_cap",
        ):
            return rc
        if rc := require(
            curated_report["filters"]["max_records_per_provenance"] == 1,
            "missing_provenance_cap",
        ):
            return rc
        if rc := require(curated_report["filters"]["max_prompt_bytes"] == 4096, "missing_prompt_byte_filter"):
            return rc
        if rc := require(curated_report["filters"]["max_choice_bytes"] == 1024, "missing_choice_byte_filter"):
            return rc
        if rc := require(
            curated_report["filters"]["max_record_payload_bytes"] == 8192,
            "missing_record_payload_byte_filter",
        ):
            return rc
        if rc := require(
            all(
                split_count <= 1
                for split_counts in curated_report["dataset_split_counts"].values()
                for split_count in split_counts.values()
            ),
            "unexpected_dataset_split_cap_count",
        ):
            return rc

        inspect_command = [
            sys.executable,
            str(ROOT / "bench" / "hceval_inspect.py"),
            "--input",
            str(packed),
            "--manifest",
            str(pack_manifest),
            "--output",
            str(inspect_json),
            "--markdown",
            str(inspect_md),
            "--junit",
            str(inspect_junit),
            "--max-prompt-bytes",
            "4096",
            "--max-choice-bytes",
            "1024",
            "--max-record-payload-bytes",
            "8192",
        ]
        completed = run_command(inspect_command)
        if completed.returncode != 0:
            return completed.returncode

        inspect_report = json.loads(inspect_json.read_text(encoding="utf-8"))
        if rc := require(inspect_report["status"] == "pass", "unexpected_inspect_status"):
            return rc
        if rc := require(inspect_report["record_count"] == 3, "unexpected_inspect_record_count"):
            return rc
        if rc := require(
            inspect_report["record_spans"] == pack_report["record_spans"],
            "unexpected_inspect_record_spans",
        ):
            return rc
        if rc := require("HCEval Dataset Inspection" in inspect_md.read_text(encoding="utf-8"), "missing_inspect_md"):
            return rc
        inspect_root = ET.parse(inspect_junit).getroot()
        if rc := require(inspect_root.attrib.get("name") == "holyc_hceval_inspect", "missing_inspect_junit"):
            return rc
        if rc := require(inspect_root.attrib.get("failures") == "0", "unexpected_inspect_junit_failures"):
            return rc

        leak_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_leak_audit.py"),
            "--input",
            str(curated),
            "--output",
            str(leak_json),
            "--markdown",
            str(leak_md),
            "--csv",
            str(leak_csv),
            "--junit",
            str(leak_junit),
            "--fail-on-leaks",
        ]
        completed = run_command(leak_command)
        if completed.returncode != 0:
            return completed.returncode

        leak_report = json.loads(leak_json.read_text(encoding="utf-8"))
        if rc := require(leak_report["status"] == "pass", "unexpected_leak_status"):
            return rc
        if rc := require(leak_report["findings"] == [], "unexpected_leak_findings"):
            return rc
        if rc := require(
            "severity,kind,dataset,splits,key_sha256,record_ids,sources,detail"
            in leak_csv.read_text(encoding="utf-8"),
            "missing_leak_csv_header",
        ):
            return rc

        index_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_index.py"),
            "--input",
            str(datasets_dir),
            "--output-dir",
            str(datasets_dir),
            "--fail-on-findings",
        ]
        completed = run_command(index_command)
        if completed.returncode != 0:
            return completed.returncode

        index_report = json.loads((datasets_dir / "dataset_index_latest.json").read_text(encoding="utf-8"))
        if rc := require(index_report["status"] == "pass", "unexpected_index_status"):
            return rc
        if rc := require(len(index_report["artifacts"]) == 3, "unexpected_index_artifact_count"):
            return rc
        if rc := require(
            "Eval Dataset Artifact Index" in (datasets_dir / "dataset_index_latest.md").read_text(encoding="utf-8"),
            "missing_index_markdown",
        ):
            return rc
        index_root = ET.parse(datasets_dir / "dataset_index_junit_latest.xml").getroot()
        if rc := require(index_root.attrib.get("name") == "holyc_dataset_index", "missing_index_junit"):
            return rc
        if rc := require(index_root.attrib.get("failures") == "0", "unexpected_index_junit_failures"):
            return rc

        provenance_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(datasets_dir),
            "--output-dir",
            str(datasets_dir),
            "--max-majority-answer-pct",
            "100",
            "--max-provenance-pct",
            "100",
            "--max-dataset-majority-answer-pct",
            "100",
            "--max-split-majority-answer-pct",
            "100",
            "--max-dataset-split-majority-answer-pct",
            "100",
            "--allow-license",
            "synthetic-smoke",
            "--fail-on-findings",
        ]
        completed = run_command(provenance_command)
        if completed.returncode != 0:
            return completed.returncode

        provenance_report = json.loads(provenance_json.read_text(encoding="utf-8"))
        if rc := require(provenance_report["status"] == "pass", "unexpected_provenance_status"):
            return rc
        if rc := require(len(provenance_report["artifacts"]) == 1, "unexpected_provenance_artifact_count"):
            return rc
        if rc := require(
            "Eval Dataset Provenance Audit" in provenance_md.read_text(encoding="utf-8"),
            "missing_provenance_markdown",
        ):
            return rc
        if rc := require(
            "source,status,source_name,source_version,license,source_url,output"
            in provenance_csv.read_text(encoding="utf-8"),
            "missing_provenance_csv_header",
        ):
            return rc
        with provenance_csv.open(newline="", encoding="utf-8") as handle:
            provenance_rows = list(csv.DictReader(handle))
        if rc := require(
            json.loads(provenance_rows[0]["answer_histogram"]) == {"0": 3},
            "missing_provenance_answer_histogram",
        ):
            return rc
        if rc := require(
            json.loads(provenance_rows[0]["dataset_answer_histograms"]) == {
                "arc-smoke": {"0": 1},
                "hellaswag-smoke": {"0": 1},
                "truthfulqa-smoke": {"0": 1},
            },
            "missing_provenance_dataset_answer_histograms",
        ):
            return rc
        if rc := require(
            json.loads(provenance_rows[0]["split_answer_histograms"]) == {"validation": {"0": 3}},
            "missing_provenance_split_answer_histograms",
        ):
            return rc
        if rc := require(
            json.loads(provenance_rows[0]["dataset_split_answer_histograms"]) == {
                "arc-smoke": {"validation": {"0": 1}},
                "hellaswag-smoke": {"validation": {"0": 1}},
                "truthfulqa-smoke": {"validation": {"0": 1}},
            },
            "missing_provenance_dataset_split_answer_histograms",
        ):
            return rc
        if rc := require(
            json.loads(provenance_rows[0]["provenance_counts"]) == {
                "synthetic ARC-shaped smoke row": 1,
                "synthetic HellaSwag-shaped smoke row": 1,
                "synthetic TruthfulQA-shaped smoke row": 1,
            },
            "missing_provenance_counts",
        ):
            return rc
        provenance_root = ET.parse(provenance_junit).getroot()
        if rc := require(
            provenance_root.attrib.get("name") == "holyc_dataset_provenance_audit",
            "missing_provenance_junit",
        ):
            return rc
        if rc := require(
            provenance_root.attrib.get("failures") == "0",
            "unexpected_provenance_junit_failures",
        ):
            return rc

        denied_license_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(curated_manifest),
            "--output-dir",
            str(tmp_path / "denied-license-provenance"),
            "--deny-license",
            "synthetic-smoke",
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            denied_license_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("denied_license_not_rejected=true", file=sys.stderr)
            return 1
        denied_license_failure = json.loads(
            (tmp_path / "denied-license-provenance" / "dataset_provenance_audit_latest.json").read_text(
                encoding="utf-8"
            )
        )
        denied_license_kinds = {
            finding["kind"]
            for artifact in denied_license_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require("license_denied" in denied_license_kinds, "missing_denied_license_finding"):
            return rc

        unallowed_license_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(curated_manifest),
            "--output-dir",
            str(tmp_path / "unallowed-license-provenance"),
            "--allow-license",
            "other-local-license",
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            unallowed_license_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("unallowed_license_not_rejected=true", file=sys.stderr)
            return 1
        unallowed_license_failure = json.loads(
            (tmp_path / "unallowed-license-provenance" / "dataset_provenance_audit_latest.json").read_text(
                encoding="utf-8"
            )
        )
        unallowed_license_kinds = {
            finding["kind"]
            for artifact in unallowed_license_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require("license_not_allowed" in unallowed_license_kinds, "missing_unallowed_license_finding"):
            return rc

        stale_manifest = tmp_path / "stale_selected_ids.manifest.json"
        stale_report = dict(curated_report)
        stale_report["selected_record_ids"] = ["wrong-id"]
        stale_manifest.write_text(json.dumps(stale_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stale_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(stale_manifest),
            "--output-dir",
            str(tmp_path / "stale-provenance"),
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            stale_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("stale_provenance_not_rejected=true", file=sys.stderr)
            return 1
        stale_failure = json.loads(
            (tmp_path / "stale-provenance" / "dataset_provenance_audit_latest.json").read_text(encoding="utf-8")
        )
        stale_kinds = {
            finding["kind"]
            for artifact in stale_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require("selected_record_ids_mismatch" in stale_kinds, "missing_stale_selected_id_finding"):
            return rc

        skew_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(curated_manifest),
            "--output-dir",
            str(tmp_path / "skew-provenance"),
            "--max-majority-answer-pct",
            "90",
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            skew_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("answer_skew_not_rejected=true", file=sys.stderr)
            return 1
        skew_failure = json.loads(
            (tmp_path / "skew-provenance" / "dataset_provenance_audit_latest.json").read_text(encoding="utf-8")
        )
        skew_kinds = {
            finding["kind"]
            for artifact in skew_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require("majority_answer_skew" in skew_kinds, "missing_answer_skew_finding"):
            return rc

        provenance_skew_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(curated_manifest),
            "--output-dir",
            str(tmp_path / "provenance-skew"),
            "--max-provenance-pct",
            "30",
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            provenance_skew_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("provenance_skew_not_rejected=true", file=sys.stderr)
            return 1
        provenance_skew_failure = json.loads(
            (tmp_path / "provenance-skew" / "dataset_provenance_audit_latest.json").read_text(encoding="utf-8")
        )
        provenance_skew_kinds = {
            finding["kind"]
            for artifact in provenance_skew_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require("provenance_source_skew" in provenance_skew_kinds, "missing_provenance_skew_finding"):
            return rc

        stale_answer_manifest = tmp_path / "stale_answer_histogram.manifest.json"
        stale_answer_report = dict(curated_report)
        stale_answer_report["answer_histogram"] = {"1": 3}
        stale_answer_manifest.write_text(
            json.dumps(stale_answer_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stale_answer_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(stale_answer_manifest),
            "--output-dir",
            str(tmp_path / "stale-answer-provenance"),
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            stale_answer_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("stale_answer_histogram_not_rejected=true", file=sys.stderr)
            return 1
        stale_answer_failure = json.loads(
            (tmp_path / "stale-answer-provenance" / "dataset_provenance_audit_latest.json").read_text(
                encoding="utf-8"
            )
        )
        stale_answer_kinds = {
            finding["kind"]
            for artifact in stale_answer_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require("answer_histogram_mismatch" in stale_answer_kinds, "missing_answer_histogram_finding"):
            return rc

        stale_dataset_answer_manifest = tmp_path / "stale_dataset_answer_histogram.manifest.json"
        stale_dataset_answer_report = dict(curated_report)
        stale_dataset_answer_report["dataset_answer_histograms"] = {"arc-smoke": {"1": 1}}
        stale_dataset_answer_manifest.write_text(
            json.dumps(stale_dataset_answer_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stale_dataset_answer_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(stale_dataset_answer_manifest),
            "--output-dir",
            str(tmp_path / "stale-dataset-answer-provenance"),
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            stale_dataset_answer_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("stale_dataset_answer_histogram_not_rejected=true", file=sys.stderr)
            return 1
        stale_dataset_answer_failure = json.loads(
            (tmp_path / "stale-dataset-answer-provenance" / "dataset_provenance_audit_latest.json").read_text(
                encoding="utf-8"
            )
        )
        stale_dataset_answer_kinds = {
            finding["kind"]
            for artifact in stale_dataset_answer_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require(
            "dataset_answer_histograms_mismatch" in stale_dataset_answer_kinds,
            "missing_dataset_answer_histogram_finding",
        ):
            return rc

        stale_split_answer_manifest = tmp_path / "stale_split_answer_histogram.manifest.json"
        stale_split_answer_report = dict(curated_report)
        stale_split_answer_report["split_answer_histograms"] = {"validation": {"1": 3}}
        stale_split_answer_manifest.write_text(
            json.dumps(stale_split_answer_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stale_split_answer_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(stale_split_answer_manifest),
            "--output-dir",
            str(tmp_path / "stale-split-answer-provenance"),
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            stale_split_answer_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("stale_split_answer_histogram_not_rejected=true", file=sys.stderr)
            return 1
        stale_split_answer_failure = json.loads(
            (tmp_path / "stale-split-answer-provenance" / "dataset_provenance_audit_latest.json").read_text(
                encoding="utf-8"
            )
        )
        stale_split_answer_kinds = {
            finding["kind"]
            for artifact in stale_split_answer_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require(
            "split_answer_histograms_mismatch" in stale_split_answer_kinds,
            "missing_split_answer_histogram_finding",
        ):
            return rc

        stale_dataset_split_answer_manifest = tmp_path / "stale_dataset_split_answer_histogram.manifest.json"
        stale_dataset_split_answer_report = dict(curated_report)
        stale_dataset_split_answer_report["dataset_split_answer_histograms"] = {
            "arc-smoke": {"validation": {"1": 1}}
        }
        stale_dataset_split_answer_manifest.write_text(
            json.dumps(stale_dataset_split_answer_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stale_dataset_split_answer_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(stale_dataset_split_answer_manifest),
            "--output-dir",
            str(tmp_path / "stale-dataset-split-answer-provenance"),
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            stale_dataset_split_answer_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("stale_dataset_split_answer_histogram_not_rejected=true", file=sys.stderr)
            return 1
        stale_dataset_split_answer_failure = json.loads(
            (
                tmp_path
                / "stale-dataset-split-answer-provenance"
                / "dataset_provenance_audit_latest.json"
            ).read_text(encoding="utf-8")
        )
        stale_dataset_split_answer_kinds = {
            finding["kind"]
            for artifact in stale_dataset_split_answer_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require(
            "dataset_split_answer_histograms_mismatch" in stale_dataset_split_answer_kinds,
            "missing_dataset_split_answer_histogram_finding",
        ):
            return rc

        dataset_split_skew_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(curated_manifest),
            "--output-dir",
            str(tmp_path / "dataset-split-skew-provenance"),
            "--max-dataset-split-majority-answer-pct",
            "90",
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            dataset_split_skew_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("dataset_split_answer_skew_not_rejected=true", file=sys.stderr)
            return 1
        dataset_split_skew_failure = json.loads(
            (tmp_path / "dataset-split-skew-provenance" / "dataset_provenance_audit_latest.json").read_text(
                encoding="utf-8"
            )
        )
        dataset_split_skew_kinds = {
            finding["kind"]
            for artifact in dataset_split_skew_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require(
            "dataset_split_majority_answer_skew" in dataset_split_skew_kinds,
            "missing_dataset_split_answer_skew_finding",
        ):
            return rc

        stale_provenance_manifest = tmp_path / "stale_provenance_counts.manifest.json"
        stale_provenance_report = dict(curated_report)
        stale_provenance_report["provenance_counts"] = {"synthetic stale source": 3}
        stale_provenance_manifest.write_text(
            json.dumps(stale_provenance_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stale_provenance_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_provenance_audit.py"),
            "--input",
            str(stale_provenance_manifest),
            "--output-dir",
            str(tmp_path / "stale-provenance-counts"),
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            stale_provenance_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("stale_provenance_counts_not_rejected=true", file=sys.stderr)
            return 1
        stale_provenance_failure = json.loads(
            (tmp_path / "stale-provenance-counts" / "dataset_provenance_audit_latest.json").read_text(
                encoding="utf-8"
            )
        )
        stale_provenance_kinds = {
            finding["kind"]
            for artifact in stale_provenance_failure["artifacts"]
            for finding in artifact["findings"]
        }
        if rc := require(
            "provenance_counts_mismatch" in stale_provenance_kinds,
            "missing_provenance_counts_finding",
        ):
            return rc

        missing_provenance_fixture = tmp_path / "missing_provenance.jsonl"
        missing_provenance_fixture.write_text(
            json.dumps(
                {
                    "id": "missing-provenance",
                    "dataset": "schema-smoke",
                    "split": "validation",
                    "prompt": "Which answer is correct?",
                    "choices": ["A", "B", "C", "D"],
                    "answer_index": 0,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        missing_provenance_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_schema_audit.py"),
            "--input",
            str(missing_provenance_fixture),
            "--output",
            str(tmp_path / "missing_provenance_schema.json"),
            "--require-provenance",
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            missing_provenance_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("missing_provenance_not_rejected=true", file=sys.stderr)
            return 1
        missing_provenance_failure = json.loads(
            (tmp_path / "missing_provenance_schema.json").read_text(encoding="utf-8")
        )
        schema_kinds = {finding["kind"] for finding in missing_provenance_failure["findings"]}
        if rc := require("missing_provenance" in schema_kinds, "missing_schema_provenance_finding"):
            return rc

        skewed_schema_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_schema_audit.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(tmp_path / "skewed_schema.json"),
            "--max-majority-answer-pct",
            "90",
            "--max-dataset-split-majority-answer-pct",
            "90",
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            skewed_schema_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("schema_answer_skew_not_rejected=true", file=sys.stderr)
            return 1
        skewed_schema_failure = json.loads((tmp_path / "skewed_schema.json").read_text(encoding="utf-8"))
        skewed_schema_kinds = {finding["kind"] for finding in skewed_schema_failure["findings"]}
        if rc := require(
            {"majority_answer_skew", "dataset_split_majority_answer_skew"}.issubset(skewed_schema_kinds),
            "missing_schema_answer_skew_finding",
        ):
            return rc

        conflict_fixture = tmp_path / "conflicting_duplicate_payloads.jsonl"
        conflict_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "conflict-a",
                            "dataset": "conflict-smoke",
                            "split": "validation",
                            "prompt": "Same prompt?",
                            "choices": ["yes", "no"],
                            "answer_index": 0,
                            "provenance": "synthetic conflict smoke row",
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "id": "conflict-b",
                            "dataset": "conflict-smoke",
                            "split": "validation",
                            "prompt": " Same   prompt? ",
                            "choices": ["yes", "no"],
                            "answer_index": 1,
                            "provenance": "synthetic conflict smoke row",
                        },
                        sort_keys=True,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        conflict_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_curate.py"),
            "--input",
            str(conflict_fixture),
            "--output",
            str(tmp_path / "conflict_curated.jsonl"),
            "--manifest",
            str(tmp_path / "conflict_curated.manifest.json"),
            "--source-name",
            "conflict-smoke",
            "--dedupe-within-split-payloads",
        ]
        completed = subprocess.run(
            conflict_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("conflicting_payloads_not_rejected=true", file=sys.stderr)
            return 1
        if rc := require("conflicting answers" in completed.stderr, "missing_conflicting_payload_error"):
            return rc

        leak_fixture = tmp_path / "leaky.jsonl"
        leak_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "leak-train",
                            "dataset": "leak-smoke",
                            "split": "train",
                            "prompt": "Shared prompt?",
                            "choices": ["yes", "no"],
                            "answer_index": 0,
                            "provenance": "synthetic leakage smoke row",
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "id": "leak-validation",
                            "dataset": "leak-smoke",
                            "split": "validation",
                            "prompt": " Shared   prompt? ",
                            "choices": ["yes", "no"],
                            "answer_index": 0,
                            "provenance": "synthetic leakage smoke row",
                        },
                        sort_keys=True,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        unsafe_leak_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_leak_audit.py"),
            "--input",
            str(leak_fixture),
            "--output",
            str(tmp_path / "leaky_audit.json"),
            "--fail-on-leaks",
        ]
        completed = subprocess.run(
            unsafe_leak_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("leaky_dataset_not_rejected=true", file=sys.stderr)
            return 1
        leak_failure = json.loads((tmp_path / "leaky_audit.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in leak_failure["findings"]}
        if rc := require({"prompt_split_leak", "payload_split_leak"}.issubset(kinds), "missing_leak_findings"):
            return rc

    print("dataset_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
