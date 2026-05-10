#!/usr/bin/env python3
"""Audit eval quality drift between quantization variants.

This host-side tool consumes existing eval JSON artifacts only. It does not
launch QEMU, alter TempleOS images, or require any network access.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvalReport:
    source: str
    status: str
    dataset: str
    split: str
    model: str
    quantization: str
    records: int
    holyc_accuracy: float | None
    llama_accuracy: float | None
    agreement: float | None
    regressions: int
    error: str = ""


@dataclass(frozen=True)
class Comparison:
    dataset: str
    split: str
    model: str
    candidate_quantization: str
    reference_quantization: str
    candidate_source: str
    reference_source: str
    candidate_status: str
    reference_status: str
    candidate_records: int
    reference_records: int
    holyc_accuracy_drop: float | None
    llama_accuracy_drop: float | None
    agreement_delta: float | None
    candidate_regressions: int
    reference_regressions: int


@dataclass(frozen=True)
class Finding:
    gate: str
    dataset: str
    split: str
    model: str
    candidate_quantization: str
    reference_quantization: str
    value: float | int | str | None
    threshold: float | int | str | None
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int:
    number = as_float(value)
    return int(number) if number is not None else 0


def load_json(path: Path) -> tuple[Any | None, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"


def iter_json_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*.json") if child.is_file()))
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    return sorted(files)


def load_eval_report(path: Path) -> EvalReport:
    payload, error = load_json(path)
    if not isinstance(payload, dict):
        return EvalReport(str(path), "invalid", "", "", "", "", 0, None, None, None, 0, error or "root must be object")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return EvalReport(
            str(path),
            "invalid",
            str(payload.get("dataset") or ""),
            str(payload.get("split") or ""),
            str(payload.get("model") or ""),
            str(payload.get("quantization") or ""),
            0,
            None,
            None,
            None,
            len(payload.get("regressions") or []),
            "missing summary object",
        )
    return EvalReport(
        source=str(path),
        status=str(payload.get("status") or "pass").lower(),
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        records=as_int(summary.get("record_count")),
        holyc_accuracy=as_float(summary.get("holyc_accuracy")),
        llama_accuracy=as_float(summary.get("llama_accuracy")),
        agreement=as_float(summary.get("agreement")),
        regressions=len(payload.get("regressions") or []),
    )


def parse_pair(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("quantization pairs must use candidate:reference syntax")
    candidate, reference = (part.strip() for part in value.split(":", 1))
    if not candidate or not reference:
        raise argparse.ArgumentTypeError("candidate and reference quantization names are required")
    if candidate == reference:
        raise argparse.ArgumentTypeError("candidate and reference quantizations must differ")
    return candidate, reference


def indexed_reports(reports: Iterable[EvalReport]) -> dict[tuple[str, str, str, str], EvalReport]:
    index: dict[tuple[str, str, str, str], EvalReport] = {}
    for report in reports:
        key = (report.dataset, report.split, report.model, report.quantization)
        if all(key) and key not in index:
            index[key] = report
    return index


def build_comparisons(reports: list[EvalReport], pairs: list[tuple[str, str]]) -> tuple[list[Comparison], list[Finding]]:
    index = indexed_reports(reports)
    groups = sorted({(report.dataset, report.split, report.model) for report in reports if report.dataset and report.split and report.model})
    comparisons: list[Comparison] = []
    findings: list[Finding] = []
    for dataset, split, model in groups:
        for candidate_quantization, reference_quantization in pairs:
            candidate = index.get((dataset, split, model, candidate_quantization))
            reference = index.get((dataset, split, model, reference_quantization))
            if candidate is None or reference is None:
                missing = candidate_quantization if candidate is None else reference_quantization
                findings.append(
                    Finding(
                        "missing_pair",
                        dataset,
                        split,
                        model,
                        candidate_quantization,
                        reference_quantization,
                        missing,
                        f"{candidate_quantization}:{reference_quantization}",
                        f"missing {missing} eval report for {dataset}/{split}/{model}",
                    )
                )
                continue
            comparisons.append(
                Comparison(
                    dataset=dataset,
                    split=split,
                    model=model,
                    candidate_quantization=candidate_quantization,
                    reference_quantization=reference_quantization,
                    candidate_source=candidate.source,
                    reference_source=reference.source,
                    candidate_status=candidate.status,
                    reference_status=reference.status,
                    candidate_records=candidate.records,
                    reference_records=reference.records,
                    holyc_accuracy_drop=metric_drop(candidate.holyc_accuracy, reference.holyc_accuracy),
                    llama_accuracy_drop=metric_drop(candidate.llama_accuracy, reference.llama_accuracy),
                    agreement_delta=metric_abs_delta(candidate.agreement, reference.agreement),
                    candidate_regressions=candidate.regressions,
                    reference_regressions=reference.regressions,
                )
            )
    return comparisons, findings


def metric_drop(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    return reference - candidate


def metric_abs_delta(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    return abs(reference - candidate)


def evaluate(comparisons: list[Comparison], reports: list[EvalReport], pair_findings: list[Finding], args: argparse.Namespace) -> list[Finding]:
    findings = list(pair_findings)
    for report in reports:
        if report.status == "invalid":
            findings.append(
                Finding("invalid_report", report.dataset, report.split, report.model, report.quantization, "", report.error, "valid json eval report", report.error)
            )
    for row in comparisons:
        fields = {
            "candidate": (row.candidate_status, row.candidate_records, row.candidate_regressions, row.candidate_quantization),
            "reference": (row.reference_status, row.reference_records, row.reference_regressions, row.reference_quantization),
        }
        for role, (status, records, regressions, quantization) in fields.items():
            if args.fail_on_failed_eval and status != "pass":
                findings.append(Finding(f"{role}_status", row.dataset, row.split, row.model, row.candidate_quantization, row.reference_quantization, status, "pass", f"{role} eval status is {status}"))
            if records < args.min_records:
                findings.append(Finding(f"{role}_min_records", row.dataset, row.split, row.model, row.candidate_quantization, row.reference_quantization, records, args.min_records, f"{quantization} has too few records"))
            if args.fail_on_regressions and regressions > 0:
                findings.append(Finding(f"{role}_regressions", row.dataset, row.split, row.model, row.candidate_quantization, row.reference_quantization, regressions, 0, f"{quantization} report has regressions"))
        if row.holyc_accuracy_drop is None:
            findings.append(Finding("missing_holyc_accuracy", row.dataset, row.split, row.model, row.candidate_quantization, row.reference_quantization, None, "numeric", "missing HolyC accuracy metric"))
        elif row.holyc_accuracy_drop > args.max_holyc_accuracy_drop:
            findings.append(Finding("max_holyc_accuracy_drop", row.dataset, row.split, row.model, row.candidate_quantization, row.reference_quantization, row.holyc_accuracy_drop, args.max_holyc_accuracy_drop, "HolyC accuracy drop exceeds threshold"))
        if row.llama_accuracy_drop is None:
            findings.append(Finding("missing_llama_accuracy", row.dataset, row.split, row.model, row.candidate_quantization, row.reference_quantization, None, "numeric", "missing llama accuracy metric"))
        elif row.llama_accuracy_drop > args.max_llama_accuracy_drop:
            findings.append(Finding("max_llama_accuracy_drop", row.dataset, row.split, row.model, row.candidate_quantization, row.reference_quantization, row.llama_accuracy_drop, args.max_llama_accuracy_drop, "llama accuracy drop exceeds threshold"))
        if row.agreement_delta is None:
            findings.append(Finding("missing_agreement", row.dataset, row.split, row.model, row.candidate_quantization, row.reference_quantization, None, "numeric", "missing agreement metric"))
        elif row.agreement_delta > args.max_agreement_delta:
            findings.append(Finding("max_agreement_delta", row.dataset, row.split, row.model, row.candidate_quantization, row.reference_quantization, row.agreement_delta, args.max_agreement_delta, "agreement delta exceeds threshold"))
    if len(comparisons) < args.min_comparisons:
        findings.append(Finding("min_comparisons", "", "", "", "", "", len(comparisons), args.min_comparisons, "too few quantization comparisons"))
    return findings


def write_csv(path: Path, rows: Iterable[Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Eval Quant Delta Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Reports: {summary['reports']}",
        f"- Comparisons: {summary['comparisons']}",
        f"- Findings: {summary['findings']}",
        "",
        "| dataset | split | model | pair | HolyC drop | llama drop | agreement delta |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["comparisons"]:
        pair = f"{row['candidate_quantization']}:{row['reference_quantization']}"
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['model']} | {pair} | "
            f"{format_metric(row['holyc_accuracy_drop'])} | {format_metric(row['llama_accuracy_drop'])} | {format_metric(row['agreement_delta'])} |"
        )
    if payload["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in payload["findings"]:
            lines.append(f"- {finding['gate']}: {finding['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_junit(path: Path, findings: list[Finding]) -> None:
    root = ET.Element("testsuite", name="eval_quant_delta_audit", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(root, "testcase", name="quant_delta_gates")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} finding(s)")
        failure.text = "\n".join(f"{finding.gate}: {finding.message}" for finding in findings)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Eval JSON files or directories")
    parser.add_argument("--pair", action="append", type=parse_pair, default=[], help="candidate:reference quantization pair, default Q4_0:Q8_0")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_quant_delta_audit_latest")
    parser.add_argument("--min-comparisons", type=int, default=1)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--max-holyc-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--max-llama-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--max-agreement-delta", type=float, default=0.05)
    parser.add_argument("--fail-on-failed-eval", action="store_true")
    parser.add_argument("--fail-on-regressions", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    pairs = args.pair or [("Q4_0", "Q8_0")]
    reports = [load_eval_report(path) for path in iter_json_files(args.inputs)]
    comparisons, pair_findings = build_comparisons(reports, pairs)
    findings = evaluate(comparisons, reports, pair_findings, args)
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "pairs": [f"{candidate}:{reference}" for candidate, reference in pairs],
        "summary": {
            "reports": len(reports),
            "comparisons": len(comparisons),
            "findings": len(findings),
        },
        "reports": [asdict(report) for report in reports],
        "comparisons": [asdict(row) for row in comparisons],
        "findings": [asdict(finding) for finding in findings],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{args.output_stem}.csv", comparisons, list(Comparison.__dataclass_fields__))
    write_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings, list(Finding.__dataclass_fields__))
    write_markdown(args.output_dir / f"{args.output_stem}.md", payload)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", findings)
    print(f"{payload['status']}: wrote {json_path}")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
