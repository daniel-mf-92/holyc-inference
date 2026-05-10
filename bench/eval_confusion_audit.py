#!/usr/bin/env python3
"""Audit eval confusion matrices and per-answer quality gates.

This host-side tool consumes existing `eval_compare.py` JSON artifacts and
checks macro F1, per-answer support, precision, recall, and HolyC loss against
llama.cpp. It does not launch QEMU and does not touch the TempleOS guest.
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
class ClassSummary:
    source: str
    scope: str
    dataset: str
    split: str
    engine: str
    answer_index: int
    label: str
    support: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ScopeSummary:
    source: str
    scope: str
    dataset: str
    split: str
    engine: str
    record_count: int
    macro_f1: float
    accuracy: float


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    scope: str
    dataset: str
    split: str
    engine: str
    answer_index: int | str
    metric: str
    value: float | int | str
    limit: float | int | str
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: eval report must be a JSON object")
    if not isinstance(payload.get("summary"), dict):
        raise ValueError(f"{path}: missing summary object")
    return payload


def scope_identity(path: Path, payload: dict[str, Any], item: dict[str, Any] | None = None) -> tuple[str, str, str]:
    if item is None:
        return "overall", str(payload.get("dataset") or ""), str(payload.get("split") or "")
    return "dataset", str(item.get("dataset") or ""), str(item.get("split") or payload.get("split") or "")


def parse_scope_summary(
    source: Path,
    payload: dict[str, Any],
    item: dict[str, Any] | None,
    *,
    engine: str,
) -> ScopeSummary:
    scope, dataset, split = scope_identity(source, payload, item)
    container = payload["summary"] if item is None else item
    return ScopeSummary(
        source=str(source),
        scope=scope,
        dataset=dataset,
        split=split,
        engine=engine,
        record_count=as_int(container.get("record_count")),
        macro_f1=as_float(container.get(f"{engine}_macro_f1")),
        accuracy=as_float(container.get(f"{engine}_accuracy")),
    )


def parse_class_summaries(
    source: Path,
    payload: dict[str, Any],
    item: dict[str, Any] | None,
    *,
    engine: str,
) -> list[ClassSummary]:
    scope, dataset, split = scope_identity(source, payload, item)
    container = payload["summary"] if item is None else item
    key = f"{engine}_per_answer_index"
    rows = container.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"{source}: missing {scope} {dataset} {key}")

    summaries: list[ClassSummary] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        answer_index = as_int(row.get("answer_index"), index)
        summaries.append(
            ClassSummary(
                source=str(source),
                scope=scope,
                dataset=dataset,
                split=split,
                engine=engine,
                answer_index=answer_index,
                label=str(row.get("label") or answer_index),
                support=as_int(row.get("support")),
                true_positive=as_int(row.get("true_positive")),
                false_positive=as_int(row.get("false_positive")),
                false_negative=as_int(row.get("false_negative")),
                precision=as_float(row.get("precision")),
                recall=as_float(row.get("recall")),
                f1=as_float(row.get("f1")),
            )
        )
    return summaries


def collect_summaries(paths: list[Path]) -> tuple[list[ScopeSummary], list[ClassSummary], list[Finding]]:
    scopes: list[ScopeSummary] = []
    classes: list[ClassSummary] = []
    findings: list[Finding] = []
    for path in paths:
        try:
            payload = load_report(path)
            for engine in ("holyc", "llama"):
                scopes.append(parse_scope_summary(path, payload, None, engine=engine))
                classes.extend(parse_class_summaries(path, payload, None, engine=engine))
            for item in payload["summary"].get("dataset_breakdown") or []:
                if not isinstance(item, dict):
                    continue
                for engine in ("holyc", "llama"):
                    if f"{engine}_per_answer_index" in item:
                        scopes.append(parse_scope_summary(path, payload, item, engine=engine))
                        classes.extend(parse_class_summaries(path, payload, item, engine=engine))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(
                Finding("error", str(path), "file", "", "", "", "", "unreadable_report", "", "", str(exc))
            )
    return scopes, classes, findings


def add_min_finding(
    findings: list[Finding],
    *,
    source: str,
    scope: str,
    dataset: str,
    split: str,
    engine: str,
    answer_index: int | str,
    metric: str,
    value: float | int,
    limit: float | int | None,
) -> None:
    if limit is None or value >= limit:
        return
    findings.append(
        Finding(
            "error",
            source,
            scope,
            dataset,
            split,
            engine,
            answer_index,
            metric,
            value,
            limit,
            f"{engine} {scope} {metric} {value} is below limit {limit}",
        )
    )


def audit_summaries(
    scopes: list[ScopeSummary],
    classes: list[ClassSummary],
    *,
    min_macro_f1: float | None,
    min_accuracy: float | None,
    min_class_support: int | None,
    min_class_precision: float | None,
    min_class_recall: float | None,
    min_class_f1: float | None,
    max_holyc_macro_f1_loss: float | None,
    include_dataset_breakdown: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    filtered_scopes = [item for item in scopes if include_dataset_breakdown or item.scope == "overall"]
    filtered_classes = [item for item in classes if include_dataset_breakdown or item.scope == "overall"]

    by_scope: dict[tuple[str, str, str, str], dict[str, ScopeSummary]] = {}
    for summary in filtered_scopes:
        by_scope.setdefault((summary.source, summary.scope, summary.dataset, summary.split), {})[summary.engine] = summary
        add_min_finding(
            findings,
            source=summary.source,
            scope=summary.scope,
            dataset=summary.dataset,
            split=summary.split,
            engine=summary.engine,
            answer_index="",
            metric="macro_f1",
            value=summary.macro_f1,
            limit=min_macro_f1,
        )
        add_min_finding(
            findings,
            source=summary.source,
            scope=summary.scope,
            dataset=summary.dataset,
            split=summary.split,
            engine=summary.engine,
            answer_index="",
            metric="accuracy",
            value=summary.accuracy,
            limit=min_accuracy,
        )

    for summary in filtered_classes:
        add_min_finding(
            findings,
            source=summary.source,
            scope=summary.scope,
            dataset=summary.dataset,
            split=summary.split,
            engine=summary.engine,
            answer_index=summary.answer_index,
            metric="class_support",
            value=summary.support,
            limit=min_class_support,
        )
        if summary.support > 0:
            add_min_finding(
                findings,
                source=summary.source,
                scope=summary.scope,
                dataset=summary.dataset,
                split=summary.split,
                engine=summary.engine,
                answer_index=summary.answer_index,
                metric="class_precision",
                value=summary.precision,
                limit=min_class_precision,
            )
            add_min_finding(
                findings,
                source=summary.source,
                scope=summary.scope,
                dataset=summary.dataset,
                split=summary.split,
                engine=summary.engine,
                answer_index=summary.answer_index,
                metric="class_recall",
                value=summary.recall,
                limit=min_class_recall,
            )
            add_min_finding(
                findings,
                source=summary.source,
                scope=summary.scope,
                dataset=summary.dataset,
                split=summary.split,
                engine=summary.engine,
                answer_index=summary.answer_index,
                metric="class_f1",
                value=summary.f1,
                limit=min_class_f1,
            )

    if max_holyc_macro_f1_loss is not None:
        for engines in by_scope.values():
            holyc = engines.get("holyc")
            llama = engines.get("llama")
            if holyc is None or llama is None:
                continue
            loss = llama.macro_f1 - holyc.macro_f1
            if loss > max_holyc_macro_f1_loss:
                findings.append(
                    Finding(
                        "error",
                        holyc.source,
                        holyc.scope,
                        holyc.dataset,
                        holyc.split,
                        "holyc",
                        "",
                        "macro_f1_loss_vs_llama",
                        loss,
                        max_holyc_macro_f1_loss,
                        f"HolyC macro F1 loss {loss} is above llama.cpp limit {max_holyc_macro_f1_loss}",
                    )
                )
    return findings


def audit_reports(paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    scopes, classes, findings = collect_summaries(paths)
    findings.extend(
        audit_summaries(
            scopes,
            classes,
            min_macro_f1=args.min_macro_f1,
            min_accuracy=args.min_accuracy,
            min_class_support=args.min_class_support,
            min_class_precision=args.min_class_precision,
            min_class_recall=args.min_class_recall,
            min_class_f1=args.min_class_f1,
            max_holyc_macro_f1_loss=args.max_holyc_macro_f1_loss,
            include_dataset_breakdown=args.include_dataset_breakdown,
        )
    )
    error_count = sum(1 for finding in findings if finding.severity == "error")
    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "inputs": [str(path) for path in paths],
        "include_dataset_breakdown": args.include_dataset_breakdown,
        "scope_summary_count": len(scopes),
        "class_summary_count": len(classes),
        "error_count": error_count,
        "scope_summaries": [asdict(summary) for summary in scopes],
        "class_summaries": [asdict(summary) for summary in classes],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Confusion Audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Inputs: {len(payload['inputs'])}",
        f"- Scope summaries: {payload['scope_summary_count']}",
        f"- Class summaries: {payload['class_summary_count']}",
        f"- Findings: {len(payload['findings'])}",
        "",
    ]
    if payload["findings"]:
        lines.extend(["## Findings", ""])
        for finding in payload["findings"]:
            lines.append(
                f"- `{finding['metric']}` {finding['source']} {finding['scope']} "
                f"{finding['dataset']} {finding['split']} {finding['engine']} "
                f"{finding['answer_index']}: {finding['message']}"
            )
    else:
        lines.append("No confusion-matrix gate findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    failures = 1 if payload["status"] == "fail" else 0
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_confusion_audit",
            "tests": "1",
            "failures": str(failures),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "confusion_gates"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": f"{len(payload['findings'])} finding(s)"})
        failure.text = "\n".join(finding["message"] for finding in payload["findings"])
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="eval_compare JSON report(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_confusion_audit_latest")
    parser.add_argument("--min-macro-f1", type=non_negative_float)
    parser.add_argument("--min-accuracy", type=non_negative_float)
    parser.add_argument("--min-class-support", type=non_negative_int)
    parser.add_argument("--min-class-precision", type=non_negative_float)
    parser.add_argument("--min-class-recall", type=non_negative_float)
    parser.add_argument("--min-class-f1", type=non_negative_float)
    parser.add_argument("--max-holyc-macro-f1-loss", type=non_negative_float)
    parser.add_argument("--include-dataset-breakdown", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    payload = audit_reports(args.reports, args)
    output_dir = args.output_dir
    stem = args.output_stem
    write_json(output_dir / f"{stem}.json", payload)
    write_markdown(output_dir / f"{stem}.md", payload)
    write_csv(
        output_dir / f"{stem}.csv",
        payload["findings"],
        [
            "severity",
            "source",
            "scope",
            "dataset",
            "split",
            "engine",
            "answer_index",
            "metric",
            "value",
            "limit",
            "message",
        ],
    )
    write_csv(
        output_dir / f"{stem}_scopes.csv",
        payload["scope_summaries"],
        ["source", "scope", "dataset", "split", "engine", "record_count", "macro_f1", "accuracy"],
    )
    write_csv(
        output_dir / f"{stem}_classes.csv",
        payload["class_summaries"],
        [
            "source",
            "scope",
            "dataset",
            "split",
            "engine",
            "answer_index",
            "label",
            "support",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_junit(output_dir / f"{stem}_junit.xml", payload)
    print(json.dumps({"status": payload["status"], "findings": len(payload["findings"])}, sort_keys=True))
    return 2 if args.fail_on_findings and payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
