#!/usr/bin/env python3
"""Offline multiple-choice eval comparator for HolyC vs llama.cpp outputs.

The comparator consumes a local gold JSONL dataset and two local prediction
files, aligns rows by record id, computes accuracy and engine agreement, and
writes JSON plus Markdown reports under bench/results. It is host-side only and
does not launch QEMU or use network services.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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
    choices: list[str]


@dataclass(frozen=True)
class Prediction:
    record_id: str
    predicted_index: int
    raw_prediction: Any
    scores: list[float] | None


@dataclass(frozen=True)
class EvalRow:
    record_id: str
    dataset: str
    split: str
    answer_index: int
    holyc_prediction: int
    llama_prediction: int
    holyc_correct: bool
    llama_correct: bool
    engines_agree: bool


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return list(read_jsonl(path))
    if suffix == ".json":
        return list(flatten_json_payload(json.loads(path.read_text(encoding="utf-8"))))
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"{path}: unsupported prediction format; use JSON, JSONL, or CSV")


def case_id(row: dict[str, Any], row_label: str) -> str:
    value = row.get("id") or row.get("record_id") or row.get("question_id") or row.get("prompt_id")
    if value is None or str(value).strip() == "":
        raise ValueError(f"{row_label}: missing record id")
    return str(value).strip()


def load_gold(path: Path, dataset: str, split: str) -> dict[str, GoldCase]:
    rows = dataset_pack.read_jsonl(path)
    records = dataset_pack.normalize_records(rows, dataset, split)
    gold: dict[str, GoldCase] = {}
    for record in records:
        if record.record_id in gold:
            raise ValueError(f"{path}: duplicate gold id {record.record_id!r}")
        gold[record.record_id] = GoldCase(
            record_id=record.record_id,
            dataset=record.dataset,
            split=record.split,
            answer_index=record.answer_index,
            choices=record.choices,
        )
    return gold


def parse_scores(value: Any, row_label: str) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{row_label}: scores string must be JSON list: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"{row_label}: scores must be a non-empty list")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row_label}: scores must contain numbers") from exc


def argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], 1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def parse_prediction_index(value: Any, choices: list[str], row_label: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    if len(text) == 1 and "A" <= text.upper() <= "Z":
        return ord(text.upper()) - ord("A")
    normalized_choices = [choice.strip().lower() for choice in choices]
    if text.lower() in normalized_choices:
        return normalized_choices.index(text.lower())
    raise ValueError(f"{row_label}: cannot map prediction {value!r} to a choice index")


def normalize_prediction(row: dict[str, Any], gold: GoldCase, source: Path, index: int) -> Prediction:
    row_label = f"{source}:{index + 1}"
    record_id = case_id(row, row_label)
    scores = parse_scores(first_present(row, SCORE_KEYS), row_label)
    raw_prediction = first_present(row, PREDICTION_KEYS)
    predicted_index = parse_prediction_index(raw_prediction, gold.choices, row_label)
    if predicted_index is None and scores is not None:
        predicted_index = argmax(scores)
        raw_prediction = predicted_index
    if predicted_index is None:
        raise ValueError(f"{row_label}: missing prediction or scores")
    if predicted_index < 0 or predicted_index >= len(gold.choices):
        raise ValueError(f"{row_label}: prediction index {predicted_index} is outside choice range")
    return Prediction(record_id=record_id, predicted_index=predicted_index, raw_prediction=raw_prediction, scores=scores)


def load_predictions(path: Path, gold: dict[str, GoldCase]) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    for index, row in enumerate(read_prediction_rows(path)):
        record_id = case_id(row, f"{path}:{index + 1}")
        if record_id not in gold:
            raise ValueError(f"{path}:{index + 1}: prediction id {record_id!r} not found in gold set")
        if record_id in predictions:
            raise ValueError(f"{path}: duplicate prediction id {record_id!r}")
        predictions[record_id] = normalize_prediction(row, gold[record_id], path, index)
    return predictions


def accuracy(correct_count: int, total: int) -> float:
    return correct_count / total if total else 0.0


def compare(
    gold: dict[str, GoldCase],
    holyc_predictions: dict[str, Prediction],
    llama_predictions: dict[str, Prediction],
) -> tuple[list[EvalRow], dict[str, Any]]:
    missing_holyc = sorted(set(gold) - set(holyc_predictions))
    missing_llama = sorted(set(gold) - set(llama_predictions))
    if missing_holyc:
        raise ValueError(f"HolyC predictions missing {len(missing_holyc)} ids: {', '.join(missing_holyc[:5])}")
    if missing_llama:
        raise ValueError(f"llama.cpp predictions missing {len(missing_llama)} ids: {', '.join(missing_llama[:5])}")

    rows: list[EvalRow] = []
    for record_id, case in sorted(gold.items()):
        holyc = holyc_predictions[record_id]
        llama = llama_predictions[record_id]
        rows.append(
            EvalRow(
                record_id=record_id,
                dataset=case.dataset,
                split=case.split,
                answer_index=case.answer_index,
                holyc_prediction=holyc.predicted_index,
                llama_prediction=llama.predicted_index,
                holyc_correct=holyc.predicted_index == case.answer_index,
                llama_correct=llama.predicted_index == case.answer_index,
                engines_agree=holyc.predicted_index == llama.predicted_index,
            )
        )

    total = len(rows)
    holyc_correct = sum(1 for row in rows if row.holyc_correct)
    llama_correct = sum(1 for row in rows if row.llama_correct)
    agreements = sum(1 for row in rows if row.engines_agree)
    summary = {
        "agreement": accuracy(agreements, total),
        "agreement_count": agreements,
        "holyc_accuracy": accuracy(holyc_correct, total),
        "holyc_correct": holyc_correct,
        "llama_accuracy": accuracy(llama_correct, total),
        "llama_correct": llama_correct,
        "record_count": total,
    }
    summary["accuracy_delta_holyc_minus_llama"] = summary["holyc_accuracy"] - summary["llama_accuracy"]
    return rows, summary


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Eval Compare Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Dataset: {report['dataset']}",
        f"Split: {report['split']}",
        f"Quantization: {report['quantization'] or '-'}",
        f"Model: {report['model'] or '-'}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Records | {summary['record_count']} |",
        f"| HolyC accuracy | {summary['holyc_accuracy']:.4f} |",
        f"| llama.cpp accuracy | {summary['llama_accuracy']:.4f} |",
        f"| Accuracy delta | {summary['accuracy_delta_holyc_minus_llama']:.4f} |",
        f"| Agreement | {summary['agreement']:.4f} |",
        "",
        "## Disagreements",
        "",
    ]
    disagreements = [row for row in report["rows"] if not row["engines_agree"]]
    if disagreements:
        lines.append("| ID | Gold | HolyC | llama.cpp |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in disagreements:
            lines.append(
                f"| {row['record_id']} | {row['answer_index']} | "
                f"{row['holyc_prediction']} | {row['llama_prediction']} |"
            )
    else:
        lines.append("No prediction disagreements.")
    return "\n".join(lines) + "\n"


def write_csv_report(path: Path, rows: list[EvalRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "dataset",
                "split",
                "answer_index",
                "holyc_prediction",
                "llama_prediction",
                "holyc_correct",
                "llama_correct",
                "engines_agree",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_report(
    rows: list[EvalRow],
    summary: dict[str, Any],
    args: argparse.Namespace,
    gold_path: Path,
    holyc_path: Path,
    llama_path: Path,
) -> tuple[Path, Path, Path]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "generated_at": iso_now(),
        "gold_sha256": file_sha256(gold_path),
        "holyc_predictions_sha256": file_sha256(holyc_path),
        "llama_predictions_sha256": file_sha256(llama_path),
        "model": args.model,
        "quantization": args.quantization,
        "rows": [asdict(row) for row in rows],
        "split": args.split,
        "summary": summary,
    }
    stem = args.output_stem
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv_report(csv_path, rows)
    return json_path, md_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Local gold JSONL dataset")
    parser.add_argument("--holyc", type=Path, required=True, help="HolyC prediction JSON/JSONL/CSV")
    parser.add_argument("--llama", type=Path, required=True, help="llama.cpp prediction JSON/JSONL/CSV")
    parser.add_argument("--dataset", default="eval")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_compare_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        gold = load_gold(args.gold, args.dataset, args.split)
        holyc_predictions = load_predictions(args.holyc, gold)
        llama_predictions = load_predictions(args.llama, gold)
        rows, summary = compare(gold, holyc_predictions, llama_predictions)
        json_path, md_path, csv_path = write_report(rows, summary, args, args.gold, args.holyc, args.llama)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_csv={csv_path}")
    print(f"holyc_accuracy={summary['holyc_accuracy']:.4f}")
    print(f"llama_accuracy={summary['llama_accuracy']:.4f}")
    print(f"agreement={summary['agreement']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
