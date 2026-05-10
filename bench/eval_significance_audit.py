#!/usr/bin/env python3
"""Audit paired HolyC-vs-llama eval significance from eval_compare reports.

This host-side tool consumes existing local eval_compare JSON artifacts only.
It extracts overall and dataset/split McNemar exact-test rows, emits compact
dashboard sidecars, and can gate statistically significant HolyC losses.
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


DEFAULT_PATTERNS = ("eval_compare*.json",)


@dataclass(frozen=True)
class ScopeRow:
    source: str
    scope: str
    dataset: str
    split: str
    record_count: int
    holyc_accuracy: float | None
    llama_accuracy: float | None
    accuracy_delta_holyc_minus_llama: float | None
    holyc_only_correct: int
    llama_only_correct: int
    discordant_count: int
    p_value: float | None
    method: str
    significant_holyc_loss: bool


@dataclass(frozen=True)
class Finding:
    source: str
    scope: str
    dataset: str
    split: str
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_input_files(paths: Iterable[Path], patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            for pattern in patterns:
                for child in sorted(path.rglob(pattern)):
                    if child.is_file() and child not in seen:
                        seen.add(child)
                        files.append(child)
        elif path.is_file() and path not in seen:
            seen.add(path)
            files.append(path)
    return sorted(files)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_int(value: Any) -> int:
    if isinstance(value, bool) or value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def mcnemar_counts(payload: dict[str, Any]) -> tuple[int, int, int, float | None, str]:
    mcnemar = payload.get("mcnemar_exact")
    if not isinstance(mcnemar, dict):
        return 0, 0, 0, None, ""
    holyc_only = as_int(mcnemar.get("holyc_only_correct"))
    llama_only = as_int(mcnemar.get("llama_only_correct"))
    discordant = as_int(mcnemar.get("discordant_count"))
    if discordant == 0:
        discordant = holyc_only + llama_only
    return holyc_only, llama_only, discordant, as_float(mcnemar.get("p_value")), str(mcnemar.get("method") or "")


def scope_row(
    path: Path,
    report: dict[str, Any],
    payload: dict[str, Any],
    *,
    scope: str,
    default_dataset: str,
    default_split: str,
    max_holyc_loss_p: float | None,
) -> ScopeRow:
    holyc_only, llama_only, discordant, p_value, method = mcnemar_counts(payload)
    dataset = str(payload.get("dataset") or default_dataset)
    split = str(payload.get("split") or default_split)
    delta = as_float(payload.get("accuracy_delta_holyc_minus_llama"))
    if delta is None:
        holyc_accuracy = as_float(payload.get("holyc_accuracy"))
        llama_accuracy = as_float(payload.get("llama_accuracy"))
        delta = None if holyc_accuracy is None or llama_accuracy is None else holyc_accuracy - llama_accuracy
    significant_loss = (
        max_holyc_loss_p is not None
        and p_value is not None
        and p_value <= max_holyc_loss_p
        and (delta or 0.0) < 0.0
        and llama_only > holyc_only
    )
    return ScopeRow(
        source=str(path),
        scope=scope,
        dataset=dataset,
        split=split,
        record_count=as_int(payload.get("record_count")),
        holyc_accuracy=as_float(payload.get("holyc_accuracy")),
        llama_accuracy=as_float(payload.get("llama_accuracy")),
        accuracy_delta_holyc_minus_llama=delta,
        holyc_only_correct=holyc_only,
        llama_only_correct=llama_only,
        discordant_count=discordant,
        p_value=p_value,
        method=method,
        significant_holyc_loss=significant_loss,
    )


def rows_from_report(path: Path, report: dict[str, Any], max_holyc_loss_p: float | None) -> tuple[list[ScopeRow], list[Finding]]:
    findings: list[Finding] = []
    summary = report.get("summary")
    if not isinstance(summary, dict):
        return [], [Finding(str(path), "file", "", "", "error", "missing_summary", "summary object is absent")]

    default_dataset = str(report.get("dataset") or "")
    default_split = str(report.get("split") or "")
    rows = [
        scope_row(
            path,
            report,
            summary,
            scope="overall",
            default_dataset=default_dataset,
            default_split=default_split,
            max_holyc_loss_p=max_holyc_loss_p,
        )
    ]
    breakdown = summary.get("dataset_breakdown")
    if isinstance(breakdown, list):
        for item in breakdown:
            if isinstance(item, dict):
                rows.append(
                    scope_row(
                        path,
                        report,
                        item,
                        scope="dataset_split",
                        default_dataset=default_dataset,
                        default_split=default_split,
                        max_holyc_loss_p=max_holyc_loss_p,
                    )
                )

    for row in rows:
        if row.discordant_count == 0:
            findings.append(
                Finding(
                    row.source,
                    row.scope,
                    row.dataset,
                    row.split,
                    "warning",
                    "no_discordant_pairs",
                    "McNemar evidence is uninformative because both engines have identical correctness on this scope",
                )
            )
        if row.significant_holyc_loss:
            findings.append(
                Finding(
                    row.source,
                    row.scope,
                    row.dataset,
                    row.split,
                    "error",
                    "significant_holyc_loss",
                    f"p_value={row.p_value:.6g} threshold={max_holyc_loss_p:.6g} delta={row.accuracy_delta_holyc_minus_llama:.6g}",
                )
            )
    return rows, findings


def audit(paths: list[Path], *, patterns: list[str], min_reports: int, max_holyc_loss_p: float | None) -> dict[str, Any]:
    files = iter_input_files(paths, patterns)
    rows: list[ScopeRow] = []
    findings: list[Finding] = []
    for path in files:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), "file", "", "", "error", "unreadable_report", str(exc)))
            continue
        if not isinstance(report, dict):
            findings.append(Finding(str(path), "file", "", "", "error", "invalid_report", "root must be a JSON object"))
            continue
        report_rows, report_findings = rows_from_report(path, report, max_holyc_loss_p)
        rows.extend(report_rows)
        findings.extend(report_findings)

    if len(files) < min_reports:
        findings.append(
            Finding(
                "",
                "collection",
                "",
                "",
                "error",
                "insufficient_reports",
                f"reports={len(files)} min_reports={min_reports}",
            )
        )

    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    significant_losses = sum(1 for row in rows if row.significant_holyc_loss)
    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "inputs": [str(path) for path in files],
        "min_reports": min_reports,
        "max_holyc_loss_p": max_holyc_loss_p,
        "summary": {
            "report_count": len(files),
            "scope_count": len(rows),
            "significant_holyc_loss_count": significant_losses,
            "error_count": error_count,
            "warning_count": warning_count,
        },
        "scopes": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Eval Significance Audit",
        "",
        f"- Status: {report['status']}",
        f"- Reports: {report['summary']['report_count']}",
        f"- Scopes: {report['summary']['scope_count']}",
        f"- Significant HolyC losses: {report['summary']['significant_holyc_loss_count']}",
        "",
        "## Scopes",
        "",
        "| Scope | Dataset | Split | Records | Delta | Discordant | p-value | Significant HolyC loss |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["scopes"]:
        p_value = "" if row["p_value"] is None else f"{row['p_value']:.6g}"
        delta = "" if row["accuracy_delta_holyc_minus_llama"] is None else f"{row['accuracy_delta_holyc_minus_llama']:.6g}"
        lines.append(
            f"| {row['scope']} | {row['dataset']} | {row['split']} | {row['record_count']} | {delta} | "
            f"{row['discordant_count']} | {p_value} | {row['significant_holyc_loss']} |"
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(
                f"- {finding['severity']} {finding['kind']} {finding['scope']} {finding['dataset']}/{finding['split']}: {finding['detail']}"
            )
    else:
        lines.append("No significance findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[dict[str, Any]]) -> None:
    errors = [finding for finding in findings if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_significance_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(errors)),
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"name": "eval_significance_audit"})
    for index, finding in enumerate(findings, 1):
        case = ET.SubElement(suite, "testcase", {"name": f"{finding['kind']}_{index}"})
        if finding["severity"] == "error":
            failure = ET.SubElement(case, "failure", {"message": finding["kind"]})
            failure.text = finding["detail"]
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="eval_compare JSON files or directories")
    parser.add_argument("--pattern", action="append", dest="patterns", help="glob pattern for directory inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_significance_audit_latest")
    parser.add_argument("--min-reports", type=int, default=1)
    parser.add_argument("--max-holyc-loss-p", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_reports < 0:
        raise SystemExit("--min-reports must be non-negative")
    if args.max_holyc_loss_p is not None and not (0.0 <= args.max_holyc_loss_p <= 1.0):
        raise SystemExit("--max-holyc-loss-p must be between 0 and 1")
    patterns = args.patterns or list(DEFAULT_PATTERNS)
    report = audit(args.inputs, patterns=patterns, min_reports=args.min_reports, max_holyc_loss_p=args.max_holyc_loss_p)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    scope_fields = list(ScopeRow.__dataclass_fields__)
    finding_fields = list(Finding.__dataclass_fields__)
    write_csv(args.output_dir / f"{stem}.csv", report["scopes"], scope_fields)
    write_csv(args.output_dir / f"{stem}_findings.csv", report["findings"], finding_fields)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_junit(args.output_dir / f"{stem}_junit.xml", report["findings"])
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
