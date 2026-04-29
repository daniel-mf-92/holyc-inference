#!/usr/bin/env python3
"""Index offline eval dataset artifacts and provenance.

The indexer scans curated JSONL manifests, packed `.hceval` manifests, and
inspection reports. It verifies local hashes where files are present and writes a
compact JSON/Markdown/CSV rollup for CI artifacts or release notes. It is
host-side only and never launches QEMU or fetches remote datasets.
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

import dataset_pack
import hceval_inspect


@dataclass(frozen=True)
class DatasetArtifact:
    source: str
    artifact_type: str
    status: str
    dataset: str
    split: str
    record_count: int
    license: str
    source_name: str
    source_version: str
    output: str
    binary_sha256: str
    source_sha256: str
    answer_histogram: dict[str, int]
    findings: list[str]


@dataclass(frozen=True)
class ArtifactTypeCoverageViolation:
    artifact_type: str
    present_count: int
    sources: list[str]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_relative(path_text: str, base: Path, fallback_base: Path | None = None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    resolved = (base / path).resolve()
    if resolved.exists() or fallback_base is None:
        return resolved
    return (fallback_base / path).resolve()


def parse_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: JSON artifact must be an object")
    return data


def read_curated_records(path: Path) -> list[dataset_pack.EvalRecord]:
    return [eval_record_from_mapping(row, index) for index, row in enumerate(dataset_pack.read_jsonl(path))]


def eval_record_from_mapping(row: dict[str, Any], index: int) -> dataset_pack.EvalRecord:
    record_id = dataset_pack.clean_text(row.get("record_id") or row.get("id"))
    dataset = dataset_pack.clean_text(row.get("dataset"))
    split = dataset_pack.clean_text(row.get("split"))
    prompt = dataset_pack.clean_text(row.get("prompt"))
    choices = row.get("choices")
    answer_index = row.get("answer_index")
    provenance = dataset_pack.clean_text(row.get("provenance"))

    if not isinstance(choices, list):
        raise ValueError(f"row {index + 1}: choices must be a list")
    cleaned_choices = [dataset_pack.clean_text(choice) for choice in choices]
    try:
        normalized_answer = int(answer_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {index + 1}: invalid answer_index {answer_index!r}") from exc

    dataset_pack.validate_record(record_id, prompt, cleaned_choices, normalized_answer, index)
    return dataset_pack.EvalRecord(
        record_id=record_id,
        dataset=dataset,
        split=split,
        prompt=prompt,
        choices=cleaned_choices,
        answer_index=normalized_answer,
        provenance=provenance,
    )


def summarize_curated_manifest(path: Path, manifest: dict[str, Any]) -> DatasetArtifact:
    findings: list[str] = []
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    output = str(manifest.get("output", ""))
    output_path = resolve_relative(output, Path.cwd(), path.parent) if output else None
    source_path_text = str(source.get("path", ""))
    source_path = resolve_relative(source_path_text, Path.cwd(), path.parent) if source_path_text else None
    source_sha256 = str(source.get("sha256", ""))
    normalized_sha256 = str(manifest.get("normalized_sha256", ""))

    if not manifest.get("source_name"):
        findings.append("missing source_name")
    if not manifest.get("source_version"):
        findings.append("missing source_version")
    if not manifest.get("license"):
        findings.append("missing license")
    if not source_path_text:
        findings.append("missing source path")
    elif source_path is None or not source_path.exists():
        findings.append(f"source path does not exist: {source_path_text}")
    elif source_sha256 and file_sha256(source_path) != source_sha256:
        findings.append("source sha256 does not match manifest")

    if not output:
        findings.append("missing curated output path")
    elif output_path is None or not output_path.exists():
        findings.append(f"curated output does not exist: {output}")
    else:
        try:
            records = read_curated_records(output_path)
            actual = hashlib.sha256(dataset_pack.canonical_rows(records)).hexdigest()
            if normalized_sha256 and actual != normalized_sha256:
                findings.append("normalized_sha256 does not match curated output")
            if any(not record.provenance for record in records):
                findings.append("one or more curated records have empty provenance")
        except ValueError as exc:
            findings.append(f"cannot normalize curated output: {exc}")

    pack_output = str(manifest.get("pack_output") or "")
    if pack_output and not resolve_relative(pack_output, Path.cwd(), path.parent).exists():
        findings.append(f"pack_output does not exist: {pack_output}")

    dataset_counts = manifest.get("dataset_counts") if isinstance(manifest.get("dataset_counts"), dict) else {}
    split_counts = manifest.get("split_counts") if isinstance(manifest.get("split_counts"), dict) else {}
    dataset = ",".join(sorted(str(key) for key in dataset_counts)) if dataset_counts else str(manifest.get("source_name", ""))
    split = ",".join(sorted(str(key) for key in split_counts)) if split_counts else ""

    return DatasetArtifact(
        source=str(path),
        artifact_type="curated_manifest",
        status="fail" if findings else "pass",
        dataset=dataset,
        split=split,
        record_count=parse_int(manifest.get("record_count")),
        license=str(manifest.get("license", "")),
        source_name=str(manifest.get("source_name", "")),
        source_version=str(manifest.get("source_version", "")),
        output=output,
        binary_sha256="",
        source_sha256=normalized_sha256,
        answer_histogram=dict(manifest.get("answer_histogram", {})),
        findings=findings,
    )


def summarize_pack_manifest(path: Path, manifest: dict[str, Any]) -> DatasetArtifact:
    findings: list[str] = []
    output = str(manifest.get("output", ""))
    output_path = resolve_relative(output, Path.cwd(), path.parent) if output else None
    binary_sha256 = str(manifest.get("binary_sha256", ""))
    source_sha256 = str(manifest.get("source_sha256", ""))
    records = manifest.get("records") if isinstance(manifest.get("records"), list) else []

    if not output:
        findings.append("missing binary output path")
    elif output_path is None or not output_path.exists():
        findings.append(f"binary output does not exist: {output}")
    else:
        actual_binary_sha256 = file_sha256(output_path)
        if binary_sha256 and actual_binary_sha256 != binary_sha256:
            findings.append("binary_sha256 does not match output file")
        try:
            dataset = hceval_inspect.parse_hceval(output_path)
            findings.extend(hceval_inspect.validate_dataset(dataset, path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(f"cannot inspect binary output: {exc}")

    if records:
        try:
            normalized = [eval_record_from_mapping(row, index) for index, row in enumerate(records)]
            actual_source_sha256 = hashlib.sha256(dataset_pack.canonical_rows(normalized)).hexdigest()
            if source_sha256 and actual_source_sha256 != source_sha256:
                findings.append("source_sha256 does not match manifest records")
            if any(not record.provenance for record in normalized):
                findings.append("one or more manifest records have empty provenance")
        except ValueError as exc:
            findings.append(f"cannot normalize manifest records: {exc}")
    else:
        findings.append("manifest records are missing")

    return DatasetArtifact(
        source=str(path),
        artifact_type="pack_manifest",
        status="fail" if findings else "pass",
        dataset=str(manifest.get("dataset", "")),
        split=str(manifest.get("split", "")),
        record_count=parse_int(manifest.get("record_count")),
        license="",
        source_name=str(manifest.get("dataset", "")),
        source_version="",
        output=output,
        binary_sha256=binary_sha256,
        source_sha256=source_sha256,
        answer_histogram=dict(manifest.get("answer_histogram", {})),
        findings=findings,
    )


def summarize_inspect_report(path: Path, report: dict[str, Any]) -> DatasetArtifact:
    findings = [str(finding) for finding in report.get("findings", [])]
    status = str(report.get("status", "fail" if findings else "pass"))
    input_path_text = str(report.get("input", ""))
    input_path = resolve_relative(input_path_text, Path.cwd(), path.parent) if input_path_text else None
    payload_sha256 = str(report.get("payload_sha256", ""))
    if input_path_text and input_path is not None and input_path.exists() and payload_sha256:
        if file_sha256(input_path) != payload_sha256:
            findings.append("payload_sha256 does not match input file")
            status = "fail"
    elif input_path_text:
        findings.append(f"input file does not exist: {input_path_text}")
        status = "fail"

    return DatasetArtifact(
        source=str(path),
        artifact_type="inspect_report",
        status=status,
        dataset=str(report.get("dataset", "")),
        split=str(report.get("split", "")),
        record_count=parse_int(report.get("record_count")),
        license="",
        source_name=str(report.get("dataset", "")),
        source_version="",
        output=input_path_text,
        binary_sha256=payload_sha256,
        source_sha256=str(report.get("source_sha256", "")),
        answer_histogram=dict(report.get("answer_histogram", {})),
        findings=findings,
    )


def looks_like_artifact(path: Path) -> bool:
    name = path.name
    return name.endswith(".manifest.json") or name.endswith(".inspect.json")


def iter_artifact_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(child for child in path.rglob("*.json") if looks_like_artifact(child))
        elif path.is_file() and looks_like_artifact(path):
            yield path


def summarize_artifact(path: Path) -> DatasetArtifact | None:
    data = read_manifest(path)
    if "payload_sha256" in data and "source_sha256" in data and "record_count" in data:
        return summarize_inspect_report(path, data)
    if data.get("format") == "hceval-curated-jsonl":
        return summarize_curated_manifest(path, data)
    if data.get("format") == "hceval-mc":
        return summarize_pack_manifest(path, data)
    return None


def load_artifacts(paths: Iterable[Path]) -> list[DatasetArtifact]:
    artifacts: list[DatasetArtifact] = []
    for path in sorted(set(iter_artifact_files(paths))):
        artifact = summarize_artifact(path)
        if artifact is not None:
            artifacts.append(artifact)
    return sorted(artifacts, key=lambda item: (item.artifact_type, item.dataset, item.split, item.source))


def index_status(artifacts: list[DatasetArtifact]) -> str:
    return "fail" if any(artifact.status == "fail" for artifact in artifacts) else "pass"


def artifact_type_coverage_violations(
    artifacts: list[DatasetArtifact],
    required_types: Iterable[str],
) -> list[ArtifactTypeCoverageViolation]:
    by_type: dict[str, list[DatasetArtifact]] = {}
    for artifact in artifacts:
        by_type.setdefault(artifact.artifact_type, []).append(artifact)

    violations: list[ArtifactTypeCoverageViolation] = []
    for artifact_type in sorted(set(required_types)):
        present = by_type.get(artifact_type, [])
        if present:
            continue
        violations.append(
            ArtifactTypeCoverageViolation(
                artifact_type=artifact_type,
                present_count=0,
                sources=[],
            )
        )
    return violations


def markdown_report(report: dict[str, Any]) -> str:
    coverage_violations = report.get("artifact_type_coverage_violations", [])
    lines = [
        "# Eval Dataset Artifact Index",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Artifacts: {len(report['artifacts'])}",
        f"Required artifact types: {', '.join(report.get('required_artifact_types', [])) or '-'}",
        f"Artifact type coverage violations: {len(coverage_violations)}",
        "",
    ]
    if report["artifacts"]:
        lines.extend(
            [
                "| Type | Status | Dataset | Split | Records | License | Findings | Source |",
                "| --- | --- | --- | --- | ---: | --- | ---: | --- |",
            ]
        )
        for artifact in report["artifacts"]:
            lines.append(
                "| {artifact_type} | {status} | {dataset} | {split} | {record_count} | "
                "{license} | {findings} | {source} |".format(
                    artifact_type=artifact["artifact_type"],
                    status=artifact["status"],
                    dataset=artifact["dataset"] or "-",
                    split=artifact["split"] or "-",
                    record_count=artifact["record_count"],
                    license=artifact["license"] or "-",
                    findings=len(artifact["findings"]),
                    source=artifact["source"],
                )
            )
    else:
        lines.append("No dataset artifacts found.")

    if coverage_violations:
        lines.extend(
            [
                "",
                "## Artifact Type Coverage Violations",
                "",
                "| Artifact type | Present |",
                "| --- | ---: |",
            ]
        )
        for violation in coverage_violations:
            lines.append(f"| {violation['artifact_type']} | {violation['present_count']} |")
    elif report.get("required_artifact_types"):
        lines.extend(["", "Artifact type coverage requirements satisfied."])

    failing = [artifact for artifact in report["artifacts"] if artifact["findings"]]
    if failing:
        lines.extend(["", "## Findings", ""])
        for artifact in failing:
            lines.append(f"### {artifact['source']}")
            lines.extend(f"- {finding}" for finding in artifact["findings"])
            lines.append("")
    else:
        lines.extend(["", "Findings: none."])
    return "\n".join(lines).rstrip() + "\n"


def junit_report(report: dict[str, Any]) -> str:
    artifacts = [row for row in report["artifacts"] if isinstance(row, dict)]
    failed_artifacts = [row for row in artifacts if row.get("status") == "fail"]
    finding_artifacts = [row for row in artifacts if row.get("findings")]
    coverage_violations = [
        row for row in report.get("artifact_type_coverage_violations", []) if isinstance(row, dict)
    ]
    failures = int(bool(failed_artifacts)) + int(bool(finding_artifacts)) + int(bool(coverage_violations))

    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_index",
            "tests": "3",
            "failures": str(failures),
        },
    )

    status_case = ET.SubElement(suite, "testcase", {"name": "artifact_status"})
    if failed_artifacts:
        failure = ET.SubElement(
            status_case,
            "failure",
            {
                "type": "dataset_artifact_failure",
                "message": f"{len(failed_artifacts)} dataset artifact(s) failed",
            },
        )
        failure.text = "\n".join(str(row.get("source", "")) for row in failed_artifacts)

    findings_case = ET.SubElement(suite, "testcase", {"name": "artifact_findings"})
    if finding_artifacts:
        failure = ET.SubElement(
            findings_case,
            "failure",
            {
                "type": "dataset_artifact_findings",
                "message": f"{len(finding_artifacts)} dataset artifact(s) have findings",
            },
        )
        failure.text = "\n".join(
            f"{row.get('source', '')}: {json.dumps(row.get('findings', []), separators=(',', ':'))}"
            for row in finding_artifacts
        )

    coverage_case = ET.SubElement(suite, "testcase", {"name": "artifact_type_coverage"})
    if coverage_violations:
        failure = ET.SubElement(
            coverage_case,
            "failure",
            {
                "type": "dataset_artifact_type_missing",
                "message": f"{len(coverage_violations)} required artifact type(s) are missing",
            },
        )
        failure.text = "\n".join(str(row.get("artifact_type", "")) for row in coverage_violations)

    return ET.tostring(suite, encoding="unicode") + "\n"


def write_csv(artifacts: list[DatasetArtifact], path: Path) -> None:
    fields = [
        "source",
        "artifact_type",
        "status",
        "dataset",
        "split",
        "record_count",
        "license",
        "source_name",
        "source_version",
        "output",
        "binary_sha256",
        "source_sha256",
        "answer_histogram",
        "findings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for artifact in artifacts:
            row = asdict(artifact)
            row["answer_histogram"] = json.dumps(artifact.answer_histogram, sort_keys=True, separators=(",", ":"))
            row["findings"] = json.dumps(artifact.findings, separators=(",", ":"))
            writer.writerow({field: row[field] for field in fields})


def write_coverage_csv(violations: list[ArtifactTypeCoverageViolation], path: Path) -> None:
    fields = ["artifact_type", "present_count", "sources"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for violation in violations:
            writer.writerow(
                {
                    "artifact_type": violation.artifact_type,
                    "present_count": violation.present_count,
                    "sources": json.dumps(violation.sources, separators=(",", ":")),
                }
            )


def write_report(
    artifacts: list[DatasetArtifact],
    output_dir: Path,
    required_artifact_types: list[str] | None = None,
) -> tuple[Path, list[ArtifactTypeCoverageViolation]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    required_artifact_types = required_artifact_types or []
    coverage_violations = artifact_type_coverage_violations(artifacts, required_artifact_types)
    status = "fail" if coverage_violations else index_status(artifacts)
    report = {
        "generated_at": iso_now(),
        "status": status,
        "required_artifact_types": sorted(set(required_artifact_types)),
        "artifact_type_coverage_violations": [
            asdict(violation) for violation in coverage_violations
        ],
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }
    json_path = output_dir / "dataset_index_latest.json"
    md_path = output_dir / "dataset_index_latest.md"
    csv_path = output_dir / "dataset_index_latest.csv"
    coverage_csv_path = output_dir / "dataset_index_artifact_type_coverage_latest.csv"
    junit_path = output_dir / "dataset_index_junit_latest.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    junit_path.write_text(junit_report(report), encoding="utf-8")
    write_csv(artifacts, csv_path)
    write_coverage_csv(coverage_violations, coverage_csv_path)
    return json_path, coverage_violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Dataset artifact file or directory; defaults to bench/results/datasets",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results/datasets"))
    parser.add_argument("--fail-on-findings", action="store_true", help="Return non-zero if any artifact fails")
    parser.add_argument(
        "--require-artifact-type",
        action="append",
        choices=["curated_manifest", "pack_manifest", "inspect_report"],
        default=[],
        help="Require at least one artifact of this type in the index; may be repeated",
    )
    parser.add_argument(
        "--fail-on-coverage",
        action="store_true",
        help="Return non-zero if any --require-artifact-type gate is missing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or [Path("bench/results/datasets")]
    try:
        artifacts = load_artifacts(inputs)
        output, coverage_violations = write_report(artifacts, args.output_dir, args.require_artifact_type)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    status = "fail" if coverage_violations else index_status(artifacts)
    print(f"wrote_json={output}")
    print(f"status={status}")
    print(f"artifacts={len(artifacts)}")
    print(f"artifact_type_coverage_violations={len(coverage_violations)}")
    if args.fail_on_coverage and coverage_violations:
        return 1
    if args.fail_on_findings and index_status(artifacts) == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
