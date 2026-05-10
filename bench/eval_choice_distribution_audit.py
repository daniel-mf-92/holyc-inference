#!/usr/bin/env python3
"""Audit eval prediction choice distributions against a local gold set.

This host-side tool reads local gold/prediction artifacts only. It never
launches QEMU, never touches the TempleOS guest, and performs no network I/O.
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


BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_pack


RESULT_KEYS = ("predictions", "results", "rows", "records")
PREDICTION_KEYS = ("prediction", "predicted", "predicted_index", "answer", "answer_index", "choice")
SCORE_KEYS = ("scores", "logprobs", "choice_scores", "choice_logprobs")


@dataclass(frozen=True)
class GoldCase:
    record_id: str
    dataset: str
    split: str
    answer_index: int
    choice_count: int


@dataclass(frozen=True)
class Prediction:
    engine: str
    record_id: str
    predicted_index: int


@dataclass(frozen=True)
class DistributionRow:
    engine: str
    dataset: str
    split: str
    choice_index: int
    gold_count: int
    predicted_count: int
    correct_count: int
    incorrect_count: int
    prediction_share: float
    gold_share: float
    prediction_gold_delta: float


@dataclass(frozen=True)
class Finding:
    engine: str
    record_id: str
    severity: str
    kind: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield payload


def flatten_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    yielded = False
    for key in RESULT_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list):
            yielded = True
            for item in rows:
                if isinstance(item, dict):
                    yield item
    if not yielded:
        yield payload


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return list(read_jsonl(path))
    if suffix == ".json":
        return list(flatten_json_payload(json.loads(path.read_text(encoding="utf-8"))))
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"{path}: unsupported input suffix {path.suffix}")


def row_id(row: dict[str, Any], label: str) -> str:
    value = row.get("id") or row.get("record_id") or row.get("question_id") or row.get("prompt_id")
    if value in (None, ""):
        raise ValueError(f"{label}: missing record id")
    return str(value)


def parse_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def parse_scores(value: Any, label: str) -> list[float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}: score field must be a JSON list: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}: score field must be a non-empty list")
    scores: list[float] = []
    for item in value:
        try:
            score = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: score field must contain only numbers") from exc
        if not math.isfinite(score):
            raise ValueError(f"{label}: score field must contain finite numbers")
        scores.append(score)
    return scores


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def prediction_index(row: dict[str, Any], label: str) -> int:
    for key in PREDICTION_KEYS:
        parsed = parse_int(row.get(key))
        if parsed is not None:
            return parsed
    for key in SCORE_KEYS:
        scores = parse_scores(row.get(key), label)
        if scores is not None:
            return argmax(scores)
    raise ValueError(f"{label}: missing prediction or scores")


def load_gold(path: Path, dataset: str, split: str) -> dict[str, GoldCase]:
    records = dataset_pack.normalize_records(dataset_pack.read_jsonl(path), dataset, split)
    gold: dict[str, GoldCase] = {}
    for record in records:
        if record.record_id in gold:
            raise ValueError(f"{path}: duplicate gold id {record.record_id!r}")
        gold[record.record_id] = GoldCase(
            record_id=record.record_id,
            dataset=record.dataset,
            split=record.split,
            answer_index=record.answer_index,
            choice_count=len(record.choices),
        )
    return gold


def load_predictions(spec: str) -> tuple[str, list[Prediction], list[Finding]]:
    if "=" not in spec:
        raise ValueError("--predictions must use ENGINE=PATH")
    engine, raw_path = spec.split("=", 1)
    engine = engine.strip()
    if not engine:
        raise ValueError("--predictions engine label must be non-empty")
    path = Path(raw_path)
    predictions: list[Prediction] = []
    findings: list[Finding] = []
    seen: set[str] = set()
    for row_number, row in enumerate(read_rows(path), 1):
        label = f"{path}:{row_number}"
        try:
            record_id = row_id(row, label)
            predicted = prediction_index(row, label)
        except ValueError as exc:
            findings.append(Finding(engine, "", "error", "parse_error", "prediction", str(exc)))
            continue
        if record_id in seen:
            findings.append(Finding(engine, record_id, "error", "duplicate_prediction", "record_id", "duplicate prediction id"))
            continue
        seen.add(record_id)
        predictions.append(Prediction(engine, record_id, predicted))
    return engine, predictions, findings


def audit(args: argparse.Namespace) -> tuple[list[DistributionRow], list[Finding]]:
    gold = load_gold(args.gold, args.dataset, args.split)
    findings: list[Finding] = []
    distribution_rows: list[DistributionRow] = []
    if len(gold) < args.min_gold_rows:
        findings.append(Finding("-", "", "error", "min_gold_rows", "gold", f"found {len(gold)}, expected at least {args.min_gold_rows}"))

    for spec in args.predictions:
        engine, predictions, parse_findings = load_predictions(spec)
        findings.extend(parse_findings)
        by_id = {prediction.record_id: prediction for prediction in predictions}
        unknown_ids = sorted(set(by_id) - set(gold))
        missing_ids = sorted(set(gold) - set(by_id))
        for record_id in unknown_ids:
            findings.append(Finding(engine, record_id, "error", "unknown_prediction", "record_id", "prediction id is absent from gold set"))
        for record_id in missing_ids:
            findings.append(Finding(engine, record_id, "error", "missing_prediction", "record_id", "gold record has no prediction"))

        buckets: dict[tuple[str, str, int], dict[str, int]] = {}
        totals: dict[tuple[str, str], int] = {}
        for case in gold.values():
            totals[(case.dataset, case.split)] = totals.get((case.dataset, case.split), 0) + 1
            for choice_index in range(case.choice_count):
                buckets.setdefault(
                    (case.dataset, case.split, choice_index),
                    {"gold": 0, "predicted": 0, "correct": 0, "incorrect": 0},
                )
            buckets[(case.dataset, case.split, case.answer_index)]["gold"] += 1

            prediction = by_id.get(case.record_id)
            if prediction is None:
                continue
            if prediction.predicted_index < 0 or prediction.predicted_index >= case.choice_count:
                findings.append(
                    Finding(
                        engine,
                        case.record_id,
                        "error",
                        "out_of_range_prediction",
                        "predicted_index",
                        f"predicted {prediction.predicted_index}, valid range is 0..{case.choice_count - 1}",
                    )
                )
                continue
            bucket = buckets[(case.dataset, case.split, prediction.predicted_index)]
            bucket["predicted"] += 1
            if prediction.predicted_index == case.answer_index:
                bucket["correct"] += 1
            else:
                bucket["incorrect"] += 1

        engine_valid_predictions = sum(1 for prediction in predictions if prediction.record_id in gold)
        if engine_valid_predictions < args.min_predictions:
            findings.append(
                Finding(engine, "", "error", "min_predictions", "predictions", f"found {engine_valid_predictions}, expected at least {args.min_predictions}")
            )

        grouped_predicted: dict[tuple[str, str], int] = {}
        grouped_max: dict[tuple[str, str], tuple[int, int]] = {}
        for key, counts in buckets.items():
            dataset, split, choice_index = key
            total = totals[(dataset, split)]
            predicted_count = counts["predicted"]
            grouped_predicted[(dataset, split)] = grouped_predicted.get((dataset, split), 0) + predicted_count
            current = grouped_max.get((dataset, split), (-1, -1))
            if predicted_count > current[1]:
                grouped_max[(dataset, split)] = (choice_index, predicted_count)
            prediction_share = predicted_count / total if total else 0.0
            gold_share = counts["gold"] / total if total else 0.0
            distribution_rows.append(
                DistributionRow(
                    engine=engine,
                    dataset=dataset,
                    split=split,
                    choice_index=choice_index,
                    gold_count=counts["gold"],
                    predicted_count=predicted_count,
                    correct_count=counts["correct"],
                    incorrect_count=counts["incorrect"],
                    prediction_share=prediction_share,
                    gold_share=gold_share,
                    prediction_gold_delta=prediction_share - gold_share,
                )
            )

        for dataset_split, total_predictions in grouped_predicted.items():
            if total_predictions < args.choice_collapse_min_predictions:
                continue
            choice_index, max_count = grouped_max[dataset_split]
            share = max_count / total_predictions if total_predictions else 0.0
            if share > args.max_choice_share:
                dataset, split = dataset_split
                findings.append(
                    Finding(
                        engine,
                        "",
                        "error",
                        "choice_collapse",
                        "predicted_index",
                        f"{dataset}/{split}: choice {choice_index} has {share:.6g} prediction share across {total_predictions} predictions",
                    )
                )

    return sorted(distribution_rows, key=lambda row: (row.engine, row.dataset, row.split, row.choice_index)), findings


def write_json(path: Path, rows: list[DistributionRow], findings: list[Finding]) -> None:
    engines = sorted({row.engine for row in rows})
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": {
            "engines": engines,
            "distribution_rows": len(rows),
            "findings": len(findings),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[DistributionRow]) -> None:
    fields = list(DistributionRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    fields = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, rows: list[DistributionRow], findings: list[Finding]) -> None:
    lines = [
        "# Eval Choice Distribution Audit",
        "",
        f"- Distribution rows: {len(rows)}",
        f"- Findings: {len(findings)}",
        "",
        "| Engine | Dataset | Split | Choice | Gold | Predicted | Pred share | Gold share |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.engine} | {row.dataset} | {row.split} | {row.choice_index} | {row.gold_count} | {row.predicted_count} | {row.prediction_share:.6g} | {row.gold_share:.6g} |"
        )
    lines.append("")
    if findings:
        lines.extend(["## Findings", ""])
        lines.extend(f"- {finding.severity}: {finding.engine} {finding.kind} {finding.detail}" for finding in findings)
    else:
        lines.append("No choice distribution findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_choice_distribution_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "choice_distribution"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} choice distribution findings"})
        failure.text = "\n".join(f"{finding.engine} {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Gold eval JSONL dataset")
    parser.add_argument("--dataset", default="eval", help="Fallback dataset name for gold rows")
    parser.add_argument("--split", default="validation", help="Fallback split name for gold rows")
    parser.add_argument("--predictions", action="append", required=True, help="Prediction artifact as ENGINE=PATH")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Directory for report artifacts")
    parser.add_argument("--output-stem", default="eval_choice_distribution_audit_latest", help="Report filename stem")
    parser.add_argument("--min-gold-rows", type=int, default=1)
    parser.add_argument("--min-predictions", type=int, default=1)
    parser.add_argument("--choice-collapse-min-predictions", type=int, default=10)
    parser.add_argument("--max-choice-share", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows, findings = audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", rows, findings)
    write_markdown(args.output_dir / f"{stem}.md", rows, findings)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
