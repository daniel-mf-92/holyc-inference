#!/usr/bin/env python3
"""Index HolyC-vs-llama eval result artifacts for CI dashboards.

This host-side tool reads existing eval comparison and eval suite summary JSON
artifacts, extracts comparable quality metadata, and writes compact
JSON/CSV/Markdown/JUnit rollups. It never launches QEMU and does not touch the
TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalArtifact:
    source: str
    artifact_type: str
    status: str
    generated_at: str
    dataset: str
    split: str
    model: str
    quantization: str
    record_count: int
    holyc_accuracy: float | None
    llama_accuracy: float | None
    accuracy_delta_holyc_minus_llama: float | None
    agreement: float | None
    regressions: int
    findings: int
    gold_sha256: str
    holyc_predictions_sha256: str
    llama_predictions_sha256: str
    suite_reports: int
    error: str = ""


@dataclass(frozen=True)
class IndexFinding:
    gate: str
    source: str
    value: float | int | str | None
    threshold: float | int | str | None
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_json_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(child for child in path.rglob("*.json") if child.is_file())
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    return sorted(files)


def classify_artifact(payload: dict[str, Any]) -> str:
    summary = payload.get("summary")
    if isinstance(summary, dict) and "reports" in payload and "weighted_holyc_accuracy" in summary:
        return "eval_suite_summary"
    if isinstance(summary, dict) and (
        "holyc_accuracy" in summary or "llama_accuracy" in summary or "agreement" in summary
    ):
        return "eval_compare"
    return "unknown"


def load_artifact(path: Path) -> EvalArtifact | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return EvalArtifact(str(path), "missing", "missing", "", "", "", "", "", 0, None, None, None, None, 0, 1, "", "", "", 0, str(exc))
    except json.JSONDecodeError as exc:
        return EvalArtifact(
            str(path),
            "invalid",
            "invalid",
            "",
            "",
            "",
            "",
            "",
            0,
            None,
            None,
            None,
            None,
            0,
            1,
            "",
            "",
            "",
            0,
            f"invalid json: {exc}",
        )
    if not isinstance(payload, dict):
        return EvalArtifact(str(path), "invalid", "invalid", "", "", "", "", "", 0, None, None, None, None, 0, 1, "", "", "", 0, "root must be an object")

    artifact_type = classify_artifact(payload)
    if artifact_type == "unknown":
        return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    findings = payload.get("findings")
    regressions = payload.get("regressions")
    if artifact_type == "eval_suite_summary":
        return EvalArtifact(
            source=str(path),
            artifact_type=artifact_type,
            status=str(payload.get("status") or "pass").lower(),
            generated_at=str(payload.get("generated_at") or ""),
            dataset="",
            split="",
            model="",
            quantization="",
            record_count=as_int(summary.get("records")),
            holyc_accuracy=as_float(summary.get("weighted_holyc_accuracy")),
            llama_accuracy=None,
            accuracy_delta_holyc_minus_llama=None,
            agreement=as_float(summary.get("weighted_agreement")),
            regressions=as_int(summary.get("regressions")),
            findings=as_int(summary.get("findings")),
            gold_sha256="",
            holyc_predictions_sha256="",
            llama_predictions_sha256="",
            suite_reports=as_int(summary.get("reports")),
        )
    return EvalArtifact(
        source=str(path),
        artifact_type=artifact_type,
        status=str(payload.get("status") or "pass").lower(),
        generated_at=str(payload.get("generated_at") or ""),
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        record_count=as_int(summary.get("record_count")),
        holyc_accuracy=as_float(summary.get("holyc_accuracy")),
        llama_accuracy=as_float(summary.get("llama_accuracy")),
        accuracy_delta_holyc_minus_llama=as_float(summary.get("accuracy_delta_holyc_minus_llama")),
        agreement=as_float(summary.get("agreement")),
        regressions=len(regressions) if isinstance(regressions, list) else as_int(regressions),
        findings=len(findings) if isinstance(findings, list) else as_int(findings),
        gold_sha256=str(payload.get("gold_sha256") or ""),
        holyc_predictions_sha256=str(payload.get("holyc_predictions_sha256") or ""),
        llama_predictions_sha256=str(payload.get("llama_predictions_sha256") or ""),
        suite_reports=0,
    )


def load_artifacts(paths: list[Path]) -> list[EvalArtifact]:
    artifacts: list[EvalArtifact] = []
    for path in iter_json_files(paths):
        artifact = load_artifact(path)
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def evaluate(artifacts: list[EvalArtifact], args: argparse.Namespace) -> list[IndexFinding]:
    findings: list[IndexFinding] = []
    if len(artifacts) < args.min_artifacts:
        findings.append(
            IndexFinding(
                "min_artifacts",
                "",
                len(artifacts),
                args.min_artifacts,
                f"found {len(artifacts)} eval artifact(s), below minimum {args.min_artifacts}",
            )
        )
    total_records = sum(artifact.record_count for artifact in artifacts)
    if total_records < args.min_records:
        findings.append(
            IndexFinding(
                "min_records",
                "",
                total_records,
                args.min_records,
                f"found {total_records} eval row(s), below minimum {args.min_records}",
            )
        )
    present_datasets = {artifact.dataset for artifact in artifacts if artifact.dataset}
    present_quantizations = {artifact.quantization for artifact in artifacts if artifact.quantization}
    for dataset in args.require_dataset:
        if dataset not in present_datasets:
            findings.append(
                IndexFinding(
                    "required_dataset",
                    "",
                    ",".join(sorted(present_datasets)),
                    dataset,
                    f"required dataset {dataset!r} is missing",
                )
            )
    for quantization in args.require_quantization:
        if quantization not in present_quantizations:
            findings.append(
                IndexFinding(
                    "required_quantization",
                    "",
                    ",".join(sorted(present_quantizations)),
                    quantization,
                    f"required quantization {quantization!r} is missing",
                )
            )
    for artifact in artifacts:
        if artifact.artifact_type in {"invalid", "missing"}:
            findings.append(IndexFinding(artifact.artifact_type, artifact.source, artifact.error, "valid", artifact.error))
            continue
        if args.fail_on_failed and artifact.status != "pass":
            findings.append(
                IndexFinding("status", artifact.source, artifact.status, "pass", f"{artifact.source} status is {artifact.status}")
            )
        if args.fail_on_regressions and artifact.regressions:
            findings.append(
                IndexFinding(
                    "regressions",
                    artifact.source,
                    artifact.regressions,
                    0,
                    f"{artifact.source} has {artifact.regressions} regression(s)",
                )
            )
        if args.min_holyc_accuracy is not None and (
            artifact.holyc_accuracy is None or artifact.holyc_accuracy < args.min_holyc_accuracy
        ):
            findings.append(
                IndexFinding(
                    "min_holyc_accuracy",
                    artifact.source,
                    artifact.holyc_accuracy,
                    args.min_holyc_accuracy,
                    f"{artifact.source} HolyC accuracy is below threshold",
                )
            )
        if args.min_agreement is not None and (artifact.agreement is None or artifact.agreement < args.min_agreement):
            findings.append(
                IndexFinding(
                    "min_agreement",
                    artifact.source,
                    artifact.agreement,
                    args.min_agreement,
                    f"{artifact.source} HolyC-vs-llama agreement is below threshold",
                )
            )
    return findings


def weighted(values: list[tuple[float | None, int]]) -> float | None:
    total = sum(count for value, count in values if value is not None and count > 0)
    if not total:
        return None
    return sum(float(value) * count for value, count in values if value is not None and count > 0) / total


def build_report(artifacts: list[EvalArtifact], findings: list[IndexFinding]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for artifact in artifacts:
        status_counts[artifact.status] = status_counts.get(artifact.status, 0) + 1
        type_counts[artifact.artifact_type] = type_counts.get(artifact.artifact_type, 0) + 1
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "records": sum(artifact.record_count for artifact in artifacts),
            "findings": len(findings),
            "regressions": sum(artifact.regressions for artifact in artifacts),
            "weighted_holyc_accuracy": weighted([(artifact.holyc_accuracy, artifact.record_count) for artifact in artifacts]),
            "weighted_agreement": weighted([(artifact.agreement, artifact.record_count) for artifact in artifacts]),
            "status_counts": dict(sorted(status_counts.items())),
            "type_counts": dict(sorted(type_counts.items())),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, artifacts: list[EvalArtifact]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvalArtifact.__dataclass_fields__))
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(asdict(artifact))


def write_findings_csv(path: Path, findings: list[IndexFinding]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(IndexFinding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Eval Result Index",
        "",
        f"- Status: {payload['status']}",
        f"- Artifacts: {summary['artifacts']}",
        f"- Records: {summary['records']}",
        f"- Findings: {summary['findings']}",
        f"- Weighted HolyC accuracy: {summary['weighted_holyc_accuracy']}",
        f"- Weighted agreement: {summary['weighted_agreement']}",
        "",
        "## Artifacts",
        "",
        "| Source | Type | Status | Dataset | Split | Model | Quant | Records | HolyC acc | Agreement | Regressions |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for artifact in payload["artifacts"]:
        lines.append(
            "| {source} | {artifact_type} | {status} | {dataset} | {split} | {model} | {quantization} | "
            "{record_count} | {holyc_accuracy} | {agreement} | {regressions} |".format(**artifact)
        )
    lines.extend(["", "## Findings", ""])
    if payload["findings"]:
        for finding in payload["findings"]:
            lines.append(f"- {finding['gate']}: {finding['message']}")
    else:
        lines.append("No eval index gate findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[IndexFinding]) -> None:
    root = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_result_index",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    testcase = ET.SubElement(root, "testcase", {"name": "eval_result_index_gates"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"type": "eval_result_index_failure"})
        failure.text = "\n".join(finding.message for finding in findings)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="eval JSON files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_result_index_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-holyc-accuracy", type=float)
    parser.add_argument("--min-agreement", type=float)
    parser.add_argument("--require-dataset", action="append", default=[])
    parser.add_argument("--require-quantization", action="append", default=[])
    parser.add_argument("--fail-on-failed", action="store_true")
    parser.add_argument("--fail-on-regressions", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    artifacts = load_artifacts(args.inputs)
    findings = evaluate(artifacts, args)
    payload = build_report(artifacts, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", payload)
    write_csv(args.output_dir / f"{stem}.csv", artifacts)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", payload)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
