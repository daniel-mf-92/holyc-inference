#!/usr/bin/env python3
"""Audit offline eval dataset curation provenance.

The audit scans curated JSONL manifests produced by dataset_curate.py and
verifies that source/license metadata, hashes, selected IDs, and per-row
provenance remain coherent. It is host-side only and never fetches datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_index
import dataset_pack


PLACEHOLDER_VALUES = {"", "unknown", "todo", "tbd", "n/a", "na", "none", "unspecified"}


@dataclass(frozen=True)
class ProvenanceFinding:
    source: str
    severity: str
    kind: str
    detail: str


@dataclass(frozen=True)
class ProvenanceArtifact:
    source: str
    status: str
    source_name: str
    source_version: str
    license: str
    source_url: str
    output: str
    record_count: int
    source_records: int
    selected_records: int
    answer_histogram: dict[str, int]
    dataset_answer_histograms: dict[str, dict[str, int]]
    split_answer_histograms: dict[str, dict[str, int]]
    majority_answer_index: str
    majority_answer_pct: float | None
    dataset_majority_answers: dict[str, dict[str, Any]]
    split_majority_answers: dict[str, dict[str, Any]]
    findings: list[ProvenanceFinding]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(records: list[dataset_pack.EvalRecord]) -> str:
    return hashlib.sha256(dataset_pack.canonical_rows(records)).hexdigest()


def clean(value: Any) -> str:
    return dataset_pack.clean_text(value)


def is_placeholder(value: Any) -> bool:
    return clean(value).casefold() in PLACEHOLDER_VALUES


def is_synthetic_manifest(manifest: dict[str, Any]) -> bool:
    text = " ".join(
        clean(manifest.get(name))
        for name in ("source_name", "source_version", "license", "source_url")
    ).casefold()
    return "synthetic" in text or "smoke" in text


def finding(path: Path, severity: str, kind: str, detail: str) -> ProvenanceFinding:
    return ProvenanceFinding(source=str(path), severity=severity, kind=kind, detail=detail)


def parse_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def count_by(records: list[dataset_pack.EvalRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(getattr(record, field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def count_by_dataset_split(records: list[dataset_pack.EvalRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        split_counts = counts.setdefault(record.dataset, {})
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
    return {dataset: dict(sorted(split_counts.items())) for dataset, split_counts in sorted(counts.items())}


def answer_histogram(records: list[dataset_pack.EvalRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.answer_index)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def answer_histograms_by_dataset(records: list[dataset_pack.EvalRecord]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for record in records:
        histogram = grouped.setdefault(record.dataset, {})
        key = str(record.answer_index)
        histogram[key] = histogram.get(key, 0) + 1
    return {
        dataset: dict(sorted(histogram.items(), key=lambda item: int(item[0])))
        for dataset, histogram in sorted(grouped.items())
    }


def answer_histograms_by_split(records: list[dataset_pack.EvalRecord]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for record in records:
        histogram = grouped.setdefault(record.split, {})
        key = str(record.answer_index)
        histogram[key] = histogram.get(key, 0) + 1
    return {
        split: dict(sorted(histogram.items(), key=lambda item: int(item[0])))
        for split, histogram in sorted(grouped.items())
    }


def majority_answer(histogram: dict[str, int]) -> tuple[str, float | None]:
    if not histogram:
        return "", None
    label, count = max(histogram.items(), key=lambda item: (item[1], item[0]))
    total = sum(histogram.values())
    pct = (count / total * 100.0) if total else None
    return label, pct


def majority_answers_by_dataset(
    histograms: dict[str, dict[str, int]]
) -> dict[str, dict[str, Any]]:
    majorities: dict[str, dict[str, Any]] = {}
    for dataset, histogram in sorted(histograms.items()):
        answer_index, pct = majority_answer(histogram)
        majorities[dataset] = {
            "answer_index": answer_index,
            "pct": pct,
            "records": sum(histogram.values()),
        }
    return majorities


def majority_answers_by_split(
    histograms: dict[str, dict[str, int]]
) -> dict[str, dict[str, Any]]:
    majorities: dict[str, dict[str, Any]] = {}
    for split, histogram in sorted(histograms.items()):
        answer_index, pct = majority_answer(histogram)
        majorities[split] = {
            "answer_index": answer_index,
            "pct": pct,
            "records": sum(histogram.values()),
        }
    return majorities


def load_records(path: Path) -> list[dataset_pack.EvalRecord]:
    records: list[dataset_pack.EvalRecord] = []
    for index, row in enumerate(dataset_pack.read_jsonl(path)):
        records.append(dataset_index.eval_record_from_mapping(row, index))
    return records


def looks_like_curated_manifest(path: Path) -> bool:
    return path.name.endswith(".manifest.json")


def iter_manifest_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(child for child in path.rglob("*.manifest.json") if looks_like_curated_manifest(child))
        elif path.is_file() and looks_like_curated_manifest(path):
            yield path


def source_path(manifest: dict[str, Any], manifest_path: Path) -> Path | None:
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    path_text = clean(source.get("path"))
    if not path_text:
        return None
    return dataset_index.resolve_relative(path_text, Path.cwd(), manifest_path.parent)


def output_path(manifest: dict[str, Any], manifest_path: Path) -> Path | None:
    path_text = clean(manifest.get("output"))
    if not path_text:
        return None
    return dataset_index.resolve_relative(path_text, Path.cwd(), manifest_path.parent)


def audit_manifest(
    path: Path,
    require_source_url: bool,
    max_majority_answer_pct: float | None,
    max_dataset_majority_answer_pct: float | None = None,
    max_split_majority_answer_pct: float | None = None,
) -> ProvenanceArtifact | None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != "hceval-curated-jsonl":
        return None

    findings: list[ProvenanceFinding] = []
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    synthetic = is_synthetic_manifest(manifest)

    for key, kind in (
        ("source_name", "missing_source_name"),
        ("source_version", "missing_source_version"),
        ("license", "missing_license"),
    ):
        if is_placeholder(manifest.get(key)):
            findings.append(finding(path, "error", kind, f"{key} is missing or placeholder"))

    if not synthetic and (require_source_url or is_placeholder(manifest.get("source_url"))):
        findings.append(
            finding(
                path,
                "error" if require_source_url else "warning",
                "missing_source_url",
                "non-synthetic dataset manifest should record a source_url",
            )
        )

    source_file = source_path(manifest, path)
    if source_file is None:
        findings.append(finding(path, "error", "missing_source_path", "source.path is missing"))
    elif not source_file.exists():
        findings.append(finding(path, "error", "missing_source_file", f"source file does not exist: {source_file}"))
    else:
        expected_source_sha = clean(source.get("sha256"))
        if is_placeholder(expected_source_sha):
            findings.append(finding(path, "error", "missing_source_sha256", "source.sha256 is missing"))
        elif file_sha256(source_file) != expected_source_sha:
            findings.append(finding(path, "error", "source_sha256_mismatch", "source.sha256 does not match file"))
        try:
            source_record_count = len(dataset_pack.read_jsonl(source_file))
            if parse_int(source.get("record_count")) != source_record_count:
                findings.append(
                    finding(
                        path,
                        "error",
                        "source_record_count_mismatch",
                        f"source.record_count={source.get('record_count')!r}, actual={source_record_count}",
                    )
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(finding(path, "error", "cannot_read_source", str(exc)))

    records: list[dataset_pack.EvalRecord] = []
    observed_answer_histogram: dict[str, int] = {}
    observed_dataset_answer_histograms: dict[str, dict[str, int]] = {}
    observed_split_answer_histograms: dict[str, dict[str, int]] = {}
    majority_answer_index = ""
    majority_answer_pct: float | None = None
    observed_dataset_majority_answers: dict[str, dict[str, Any]] = {}
    observed_split_majority_answers: dict[str, dict[str, Any]] = {}
    curated_output = output_path(manifest, path)
    if curated_output is None:
        findings.append(finding(path, "error", "missing_output_path", "output is missing"))
    elif not curated_output.exists():
        findings.append(finding(path, "error", "missing_output_file", f"output file does not exist: {curated_output}"))
    else:
        try:
            records = load_records(curated_output)
            expected_digest = clean(manifest.get("normalized_sha256"))
            actual_digest = canonical_sha256(records)
            if is_placeholder(expected_digest):
                findings.append(finding(path, "error", "missing_normalized_sha256", "normalized_sha256 is missing"))
            elif expected_digest != actual_digest:
                findings.append(
                    finding(path, "error", "normalized_sha256_mismatch", "normalized_sha256 does not match output")
                )

            selected_ids = manifest.get("selected_record_ids")
            actual_ids = [record.record_id for record in records]
            if not isinstance(selected_ids, list):
                findings.append(finding(path, "error", "missing_selected_record_ids", "selected_record_ids is missing"))
            elif [clean(value) for value in selected_ids] != actual_ids:
                findings.append(
                    finding(path, "error", "selected_record_ids_mismatch", "selected_record_ids do not match output")
                )

            if parse_int(manifest.get("record_count")) != len(records):
                findings.append(
                    finding(
                        path,
                        "error",
                        "record_count_mismatch",
                        f"record_count={manifest.get('record_count')!r}, actual={len(records)}",
                    )
                )
            if count_by(records, "dataset") != manifest.get("dataset_counts"):
                findings.append(finding(path, "error", "dataset_counts_mismatch", "dataset_counts do not match output"))
            if count_by(records, "split") != manifest.get("split_counts"):
                findings.append(finding(path, "error", "split_counts_mismatch", "split_counts do not match output"))
            if count_by_dataset_split(records) != manifest.get("dataset_split_counts"):
                findings.append(
                    finding(path, "error", "dataset_split_counts_mismatch", "dataset_split_counts do not match output")
                )
            observed_answer_histogram = answer_histogram(records)
            observed_dataset_answer_histograms = answer_histograms_by_dataset(records)
            observed_split_answer_histograms = answer_histograms_by_split(records)
            majority_answer_index, majority_answer_pct = majority_answer(observed_answer_histogram)
            observed_dataset_majority_answers = majority_answers_by_dataset(observed_dataset_answer_histograms)
            observed_split_majority_answers = majority_answers_by_split(observed_split_answer_histograms)
            if observed_answer_histogram != manifest.get("answer_histogram"):
                findings.append(
                    finding(path, "error", "answer_histogram_mismatch", "answer_histogram does not match output")
                )
            manifest_dataset_answer_histograms = manifest.get("dataset_answer_histograms")
            if (
                manifest_dataset_answer_histograms is not None
                and observed_dataset_answer_histograms != manifest_dataset_answer_histograms
            ):
                findings.append(
                    finding(
                        path,
                        "error",
                        "dataset_answer_histograms_mismatch",
                        "dataset_answer_histograms does not match output",
                    )
                )
            manifest_split_answer_histograms = manifest.get("split_answer_histograms")
            if (
                manifest_split_answer_histograms is not None
                and observed_split_answer_histograms != manifest_split_answer_histograms
            ):
                findings.append(
                    finding(
                        path,
                        "error",
                        "split_answer_histograms_mismatch",
                        "split_answer_histograms does not match output",
                    )
                )
            if (
                max_majority_answer_pct is not None
                and majority_answer_pct is not None
                and majority_answer_pct > max_majority_answer_pct
            ):
                findings.append(
                    finding(
                        path,
                        "error",
                        "majority_answer_skew",
                        (
                            f"answer index {majority_answer_index} covers {majority_answer_pct:.2f}% of records, "
                            f"above {max_majority_answer_pct:.2f}% gate"
                        ),
                    )
                )
            if max_dataset_majority_answer_pct is not None:
                for dataset, majority in observed_dataset_majority_answers.items():
                    pct = majority.get("pct")
                    if pct is not None and pct > max_dataset_majority_answer_pct:
                        findings.append(
                            finding(
                                path,
                                "error",
                                "dataset_majority_answer_skew",
                                (
                                    f"{dataset} answer index {majority['answer_index']} covers {pct:.2f}% "
                                    f"of records, above {max_dataset_majority_answer_pct:.2f}% gate"
                                ),
                            )
                        )
            if max_split_majority_answer_pct is not None:
                for split, majority in observed_split_majority_answers.items():
                    pct = majority.get("pct")
                    if pct is not None and pct > max_split_majority_answer_pct:
                        findings.append(
                            finding(
                                path,
                                "error",
                                "split_majority_answer_skew",
                                (
                                    f"{split} answer index {majority['answer_index']} covers {pct:.2f}% "
                                    f"of records, above {max_split_majority_answer_pct:.2f}% gate"
                                ),
                            )
                        )
            empty_provenance = [record.record_id for record in records if not record.provenance]
            if empty_provenance:
                findings.append(
                    finding(
                        path,
                        "error",
                        "empty_record_provenance",
                        "records without provenance: " + ",".join(empty_provenance[:10]),
                    )
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(finding(path, "error", "cannot_read_output", str(exc)))

    pack_output = clean(manifest.get("pack_output"))
    if pack_output:
        pack_path = dataset_index.resolve_relative(pack_output, Path.cwd(), path.parent)
        if not pack_path.exists():
            findings.append(finding(path, "error", "missing_pack_output", f"pack_output does not exist: {pack_output}"))

    status = "fail" if any(item.severity == "error" for item in findings) else "warn" if findings else "pass"
    return ProvenanceArtifact(
        source=str(path),
        status=status,
        source_name=clean(manifest.get("source_name")),
        source_version=clean(manifest.get("source_version")),
        license=clean(manifest.get("license")),
        source_url=clean(manifest.get("source_url")),
        output=clean(manifest.get("output")),
        record_count=parse_int(manifest.get("record_count")),
        source_records=parse_int(source.get("record_count")),
        selected_records=len(records),
        answer_histogram=observed_answer_histogram,
        dataset_answer_histograms=observed_dataset_answer_histograms,
        split_answer_histograms=observed_split_answer_histograms,
        majority_answer_index=majority_answer_index,
        majority_answer_pct=majority_answer_pct,
        dataset_majority_answers=observed_dataset_majority_answers,
        split_majority_answers=observed_split_majority_answers,
        findings=findings,
    )


def load_artifacts(
    paths: Iterable[Path],
    require_source_url: bool,
    max_majority_answer_pct: float | None,
    max_dataset_majority_answer_pct: float | None = None,
    max_split_majority_answer_pct: float | None = None,
) -> list[ProvenanceArtifact]:
    artifacts: list[ProvenanceArtifact] = []
    for path in sorted(set(iter_manifest_files(paths))):
        artifact = audit_manifest(
            path,
            require_source_url,
            max_majority_answer_pct,
            max_dataset_majority_answer_pct,
            max_split_majority_answer_pct,
        )
        if artifact is not None:
            artifacts.append(artifact)
    return sorted(artifacts, key=lambda item: item.source)


def report_status(artifacts: list[ProvenanceArtifact]) -> str:
    if any(artifact.status == "fail" for artifact in artifacts):
        return "fail"
    if any(artifact.status == "warn" for artifact in artifacts):
        return "warn"
    return "pass"


def artifact_dict(artifact: ProvenanceArtifact) -> dict[str, Any]:
    row = asdict(artifact)
    row["findings"] = [asdict(item) for item in artifact.findings]
    return row


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Eval Dataset Provenance Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Artifacts: {len(report['artifacts'])}",
        "",
    ]
    if report["artifacts"]:
        lines.extend(
            [
                "| Status | Source | Source Name | Version | License | Records | Majority Answer | Findings |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for artifact in report["artifacts"]:
            majority = "-"
            if artifact["majority_answer_pct"] is not None:
                majority = f"{artifact['majority_answer_index']} ({artifact['majority_answer_pct']:.2f}%)"
            lines.append(
                "| {status} | {source} | {source_name} | {source_version} | {license} | {record_count} | {majority} | {findings} |".format(
                    status=artifact["status"],
                    source=artifact["source"],
                    source_name=artifact["source_name"] or "-",
                    source_version=artifact["source_version"] or "-",
                    license=artifact["license"] or "-",
                    record_count=artifact["record_count"],
                    majority=majority,
                    findings=len(artifact["findings"]),
                )
            )
    else:
        lines.append("No curated dataset manifests found.")

    findings = [item for artifact in report["artifacts"] for item in artifact["findings"]]
    if findings:
        lines.extend(["", "## Findings", ""])
        for item in findings:
            lines.append(f"- {item['severity']} {item['kind']} {item['source']}: {item['detail']}")
    else:
        lines.extend(["", "Findings: none."])
    return "\n".join(lines).rstrip() + "\n"


def write_csv(artifacts: list[ProvenanceArtifact], path: Path) -> None:
    fields = [
        "source",
        "status",
        "source_name",
        "source_version",
        "license",
        "source_url",
        "output",
        "record_count",
        "source_records",
        "selected_records",
        "answer_histogram",
        "dataset_answer_histograms",
        "split_answer_histograms",
        "majority_answer_index",
        "majority_answer_pct",
        "dataset_majority_answers",
        "split_majority_answers",
        "findings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for artifact in artifacts:
            row = asdict(artifact)
            row["answer_histogram"] = json.dumps(artifact.answer_histogram, separators=(",", ":"))
            row["dataset_answer_histograms"] = json.dumps(
                artifact.dataset_answer_histograms, separators=(",", ":")
            )
            row["split_answer_histograms"] = json.dumps(
                artifact.split_answer_histograms, separators=(",", ":")
            )
            row["dataset_majority_answers"] = json.dumps(
                artifact.dataset_majority_answers, separators=(",", ":")
            )
            row["split_majority_answers"] = json.dumps(
                artifact.split_majority_answers, separators=(",", ":")
            )
            row["findings"] = json.dumps([asdict(item) for item in artifact.findings], separators=(",", ":"))
            writer.writerow({field: row[field] for field in fields})


def junit_report(report: dict[str, Any]) -> str:
    artifacts = [row for row in report["artifacts"] if isinstance(row, dict)]
    failing = [row for row in artifacts if row.get("status") == "fail"]
    findings = [item for row in artifacts for item in row.get("findings", [])]
    failures = int(bool(failing)) + int(bool(findings))
    suite = ET.Element("testsuite", {"name": "holyc_dataset_provenance_audit", "tests": "2", "failures": str(failures)})

    status_case = ET.SubElement(suite, "testcase", {"name": "provenance_status"})
    if failing:
        failure = ET.SubElement(
            status_case,
            "failure",
            {"type": "dataset_provenance_failure", "message": f"{len(failing)} manifest(s) failed"},
        )
        failure.text = "\n".join(str(row.get("source", "")) for row in failing)

    findings_case = ET.SubElement(suite, "testcase", {"name": "provenance_findings"})
    if findings:
        failure = ET.SubElement(
            findings_case,
            "failure",
            {"type": "dataset_provenance_findings", "message": f"{len(findings)} provenance finding(s)"},
        )
        failure.text = "\n".join(
            f"{item.get('severity')} {item.get('kind')} {item.get('source')}: {item.get('detail')}"
            for item in findings
        )
    return ET.tostring(suite, encoding="unicode") + "\n"


def write_report(artifacts: list[ProvenanceArtifact], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": iso_now(),
        "status": report_status(artifacts),
        "artifacts": [artifact_dict(artifact) for artifact in artifacts],
    }
    json_path = output_dir / "dataset_provenance_audit_latest.json"
    md_path = output_dir / "dataset_provenance_audit_latest.md"
    csv_path = output_dir / "dataset_provenance_audit_latest.csv"
    junit_path = output_dir / "dataset_provenance_audit_junit_latest.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(artifacts, csv_path)
    junit_path.write_text(junit_report(report), encoding="utf-8")
    return json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Curated manifest file or directory; defaults to bench/results/datasets",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results/datasets"))
    parser.add_argument(
        "--require-source-url",
        action="store_true",
        help="Treat missing source_url as an error for non-synthetic manifests",
    )
    parser.add_argument(
        "--max-majority-answer-pct",
        type=float,
        help="Fail when one answer index covers more than this percentage of curated records",
    )
    parser.add_argument(
        "--max-dataset-majority-answer-pct",
        type=float,
        help="Fail when one answer index covers more than this percentage within any single dataset",
    )
    parser.add_argument(
        "--max-split-majority-answer-pct",
        type=float,
        help="Fail when one answer index covers more than this percentage within any single split",
    )
    parser.add_argument("--fail-on-findings", action="store_true", help="Return non-zero if any finding is emitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_majority_answer_pct is not None and not 0.0 <= args.max_majority_answer_pct <= 100.0:
        print("error: --max-majority-answer-pct must be between 0 and 100", file=sys.stderr)
        return 2
    if (
        args.max_dataset_majority_answer_pct is not None
        and not 0.0 <= args.max_dataset_majority_answer_pct <= 100.0
    ):
        print("error: --max-dataset-majority-answer-pct must be between 0 and 100", file=sys.stderr)
        return 2
    if (
        args.max_split_majority_answer_pct is not None
        and not 0.0 <= args.max_split_majority_answer_pct <= 100.0
    ):
        print("error: --max-split-majority-answer-pct must be between 0 and 100", file=sys.stderr)
        return 2
    inputs = args.input or [Path("bench/results/datasets")]
    try:
        artifacts = load_artifacts(
            inputs,
            args.require_source_url,
            args.max_majority_answer_pct,
            args.max_dataset_majority_answer_pct,
            args.max_split_majority_answer_pct,
        )
        output = write_report(artifacts, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    findings = sum(len(artifact.findings) for artifact in artifacts)
    status = report_status(artifacts)
    print(f"wrote_json={output}")
    print(f"status={status}")
    print(f"artifacts={len(artifacts)}")
    print(f"findings={findings}")
    if args.fail_on_findings and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
