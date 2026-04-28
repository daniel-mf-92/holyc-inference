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
CONFIDENCE_Z = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.98: 2.3263478740408408,
    0.99: 2.5758293035489004,
}


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
    holyc_confidence: float | None
    llama_confidence: float | None
    holyc_margin: float | None
    llama_margin: float | None
    holyc_gold_rank: int | None
    llama_gold_rank: int | None


@dataclass(frozen=True)
class EvalRegression:
    metric: str
    value: float
    threshold: float
    message: str


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
        scores = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row_label}: scores must contain numbers") from exc
    if not all(math.isfinite(score) for score in scores):
        raise ValueError(f"{row_label}: scores must contain only finite numbers")
    return scores


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
    if scores is not None and len(scores) != len(gold.choices):
        raise ValueError(
            f"{row_label}: scores length {len(scores)} does not match choice count {len(gold.choices)}"
        )
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


def z_for_confidence(confidence_level: float) -> float:
    rounded = round(confidence_level, 2)
    if rounded not in CONFIDENCE_Z:
        raise ValueError("--confidence-level must be one of: 0.80, 0.90, 0.95, 0.98, 0.99")
    return CONFIDENCE_Z[rounded]


def wilson_interval(successes: int, total: int, confidence_level: float) -> dict[str, Any]:
    if total <= 0:
        return {
            "confidence_level": confidence_level,
            "lower": 0.0,
            "method": "wilson",
            "point": 0.0,
            "successes": successes,
            "total": total,
            "upper": 0.0,
        }

    z = z_for_confidence(confidence_level)
    point = successes / total
    denominator = 1.0 + (z * z / total)
    center = (point + (z * z / (2.0 * total))) / denominator
    margin = (
        z
        * math.sqrt((point * (1.0 - point) / total) + (z * z / (4.0 * total * total)))
        / denominator
    )
    return {
        "confidence_level": confidence_level,
        "lower": max(0.0, center - margin),
        "method": "wilson",
        "point": point,
        "successes": successes,
        "total": total,
        "upper": min(1.0, center + margin),
    }


def exact_mcnemar_test(holyc_only_correct: int, llama_only_correct: int) -> dict[str, Any]:
    discordant = holyc_only_correct + llama_only_correct
    if discordant == 0:
        return {
            "discordant_count": 0,
            "holyc_only_correct": holyc_only_correct,
            "llama_only_correct": llama_only_correct,
            "method": "exact_binomial_two_sided",
            "p_value": 1.0,
        }

    tail = min(holyc_only_correct, llama_only_correct)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return {
        "discordant_count": discordant,
        "holyc_only_correct": holyc_only_correct,
        "llama_only_correct": llama_only_correct,
        "method": "exact_binomial_two_sided",
        "p_value": min(1.0, 2.0 * probability),
    }


def add_confidence_intervals(summary: dict[str, Any], confidence_level: float) -> dict[str, Any]:
    total = int(summary["record_count"])
    enriched = dict(summary)
    enriched["confidence_intervals"] = {
        "agreement": wilson_interval(int(summary["agreement_count"]), total, confidence_level),
        "holyc_accuracy": wilson_interval(int(summary["holyc_correct"]), total, confidence_level),
        "llama_accuracy": wilson_interval(int(summary["llama_correct"]), total, confidence_level),
    }
    return enriched


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def class_count(gold: dict[str, GoldCase]) -> int:
    return max((len(case.choices) for case in gold.values()), default=0)


def classification_metrics(rows: list[EvalRow], engine: str, labels: list[int]) -> dict[str, Any]:
    matrix = [[0 for _ in labels] for _ in labels]
    label_to_offset = {label: offset for offset, label in enumerate(labels)}

    for row in rows:
        prediction = row.holyc_prediction if engine == "holyc" else row.llama_prediction
        gold_offset = label_to_offset[row.answer_index]
        pred_offset = label_to_offset[prediction]
        matrix[gold_offset][pred_offset] += 1

    per_answer_index: list[dict[str, Any]] = []
    supported_f1: list[float] = []
    for label in labels:
        offset = label_to_offset[label]
        true_positive = matrix[offset][offset]
        support = sum(matrix[offset])
        predicted_count = sum(row[offset] for row in matrix)
        precision = safe_div(true_positive, predicted_count)
        recall = safe_div(true_positive, support)
        f1 = f1_score(precision, recall)
        if support:
            supported_f1.append(f1)
        per_answer_index.append(
            {
                "answer_index": label,
                "f1": f1,
                "precision": precision,
                "predicted_count": predicted_count,
                "recall": recall,
                "support": support,
                "true_positive": true_positive,
            }
        )

    return {
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix,
        },
        "macro_f1": sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0,
        "per_answer_index": per_answer_index,
    }


def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def prediction_confidence(scores: list[float] | None, predicted_index: int) -> float | None:
    if scores is None:
        return None
    return softmax(scores)[predicted_index]


def prediction_margin(scores: list[float] | None, predicted_index: int) -> float | None:
    if scores is None:
        return None
    probabilities = softmax(scores)
    predicted_probability = probabilities[predicted_index]
    runner_up = max(
        (value for index, value in enumerate(probabilities) if index != predicted_index),
        default=0.0,
    )
    return predicted_probability - runner_up


def gold_rank(scores: list[float] | None, answer_index: int) -> int | None:
    if scores is None:
        return None
    gold_score = scores[answer_index]
    return 1 + sum(1 for score in scores if score > gold_score)


def rank_metrics(rows: list[EvalRow], engine: str, *, max_k: int = 3) -> dict[str, Any]:
    ranks = [
        row.holyc_gold_rank if engine == "holyc" else row.llama_gold_rank
        for row in rows
    ]
    scored_ranks = [rank for rank in ranks if rank is not None]
    topk = {
        f"top_{k}_accuracy": safe_div(sum(1 for rank in scored_ranks if rank <= k), len(scored_ranks))
        for k in range(1, max_k + 1)
    }
    return {
        **topk,
        "mean_gold_rank": safe_div(sum(scored_ranks), len(scored_ranks)),
        "mean_reciprocal_rank": safe_div(sum(1.0 / rank for rank in scored_ranks), len(scored_ranks)),
        "score_coverage": safe_div(len(scored_ranks), len(rows)),
        "scored_count": len(scored_ranks),
        "total_count": len(rows),
    }


def calibration_metrics(
    gold: dict[str, GoldCase],
    predictions: dict[str, Prediction],
    rows: list[EvalRow],
    engine: str,
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        prediction = predictions[row.record_id]
        if prediction.scores is None:
            continue
        probabilities = softmax(prediction.scores)
        case = gold[row.record_id]
        correct = row.holyc_correct if engine == "holyc" else row.llama_correct
        confidence = probabilities[prediction.predicted_index]
        brier = sum(
            (probability - (1.0 if index == case.answer_index else 0.0)) ** 2
            for index, probability in enumerate(probabilities)
        )
        scored.append({"brier": brier, "confidence": confidence, "correct": correct})

    bins: list[dict[str, Any]] = []
    ece = 0.0
    for offset in range(bin_count):
        lower = offset / bin_count
        upper = (offset + 1) / bin_count
        if offset == bin_count - 1:
            in_bin = [item for item in scored if lower <= item["confidence"] <= upper]
        else:
            in_bin = [item for item in scored if lower <= item["confidence"] < upper]
        accuracy_value = safe_div(sum(1 for item in in_bin if item["correct"]), len(in_bin))
        confidence_value = safe_div(sum(item["confidence"] for item in in_bin), len(in_bin))
        contribution = (len(in_bin) / len(scored)) * abs(accuracy_value - confidence_value) if scored else 0.0
        ece += contribution
        bins.append(
            {
                "accuracy": accuracy_value,
                "confidence": confidence_value,
                "count": len(in_bin),
                "ece_contribution": contribution,
                "lower": lower,
                "upper": upper,
            }
        )

    return {
        "accuracy_when_scored": safe_div(sum(1 for item in scored if item["correct"]), len(scored)),
        "brier_score": safe_div(sum(item["brier"] for item in scored), len(scored)),
        "calibration_bins": bins,
        "ece": ece,
        "mean_confidence": safe_div(sum(item["confidence"] for item in scored), len(scored)),
        "score_coverage": safe_div(len(scored), len(rows)),
        "scored_count": len(scored),
        "total_count": len(rows),
    }


def dataset_breakdown(rows: list[EvalRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[EvalRow]] = {}
    for row in rows:
        groups.setdefault((row.dataset, row.split), []).append(row)

    breakdown: list[dict[str, Any]] = []
    for (dataset, split), group_rows in sorted(groups.items()):
        total = len(group_rows)
        holyc_correct = sum(1 for row in group_rows if row.holyc_correct)
        llama_correct = sum(1 for row in group_rows if row.llama_correct)
        agreement_count = sum(1 for row in group_rows if row.engines_agree)
        breakdown.append(
            {
                "accuracy_delta_holyc_minus_llama": accuracy(holyc_correct, total)
                - accuracy(llama_correct, total),
                "agreement": accuracy(agreement_count, total),
                "agreement_count": agreement_count,
                "dataset": dataset,
                "holyc_accuracy": accuracy(holyc_correct, total),
                "holyc_correct": holyc_correct,
                "llama_accuracy": accuracy(llama_correct, total),
                "llama_correct": llama_correct,
                "record_count": total,
                "split": split,
            }
        )
    return breakdown


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
                holyc_confidence=prediction_confidence(holyc.scores, holyc.predicted_index),
                llama_confidence=prediction_confidence(llama.scores, llama.predicted_index),
                holyc_margin=prediction_margin(holyc.scores, holyc.predicted_index),
                llama_margin=prediction_margin(llama.scores, llama.predicted_index),
                holyc_gold_rank=gold_rank(holyc.scores, case.answer_index),
                llama_gold_rank=gold_rank(llama.scores, case.answer_index),
            )
        )

    total = len(rows)
    holyc_correct = sum(1 for row in rows if row.holyc_correct)
    llama_correct = sum(1 for row in rows if row.llama_correct)
    agreements = sum(1 for row in rows if row.engines_agree)
    both_correct = sum(1 for row in rows if row.holyc_correct and row.llama_correct)
    both_wrong = sum(1 for row in rows if not row.holyc_correct and not row.llama_correct)
    holyc_only_correct = sum(1 for row in rows if row.holyc_correct and not row.llama_correct)
    llama_only_correct = sum(1 for row in rows if row.llama_correct and not row.holyc_correct)
    labels = list(range(class_count(gold)))
    holyc_metrics = classification_metrics(rows, "holyc", labels)
    llama_metrics = classification_metrics(rows, "llama", labels)
    holyc_calibration = calibration_metrics(gold, holyc_predictions, rows, "holyc")
    llama_calibration = calibration_metrics(gold, llama_predictions, rows, "llama")
    summary = {
        "agreement": accuracy(agreements, total),
        "agreement_count": agreements,
        "class_count": len(labels),
        "dataset_breakdown": dataset_breakdown(rows),
        "holyc_accuracy": accuracy(holyc_correct, total),
        "holyc_calibration": holyc_calibration,
        "holyc_confusion_matrix": holyc_metrics["confusion_matrix"],
        "holyc_correct": holyc_correct,
        "holyc_macro_f1": holyc_metrics["macro_f1"],
        "holyc_per_answer_index": holyc_metrics["per_answer_index"],
        "holyc_rank_metrics": rank_metrics(rows, "holyc"),
        "llama_accuracy": accuracy(llama_correct, total),
        "llama_calibration": llama_calibration,
        "llama_confusion_matrix": llama_metrics["confusion_matrix"],
        "llama_correct": llama_correct,
        "llama_macro_f1": llama_metrics["macro_f1"],
        "llama_per_answer_index": llama_metrics["per_answer_index"],
        "llama_rank_metrics": rank_metrics(rows, "llama"),
        "mcnemar_exact": exact_mcnemar_test(holyc_only_correct, llama_only_correct),
        "paired_correctness": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "holyc_only_correct": holyc_only_correct,
            "llama_only_correct": llama_only_correct,
        },
        "record_count": total,
    }
    summary["accuracy_delta_holyc_minus_llama"] = summary["holyc_accuracy"] - summary["llama_accuracy"]
    summary["macro_f1_delta_holyc_minus_llama"] = summary["holyc_macro_f1"] - summary["llama_macro_f1"]
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
        f"Status: {report['status']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Records | {summary['record_count']} |",
        f"| HolyC accuracy | {summary['holyc_accuracy']:.4f} |",
        f"| llama.cpp accuracy | {summary['llama_accuracy']:.4f} |",
        f"| Accuracy delta | {summary['accuracy_delta_holyc_minus_llama']:.4f} |",
        f"| HolyC macro F1 | {summary['holyc_macro_f1']:.4f} |",
        f"| llama.cpp macro F1 | {summary['llama_macro_f1']:.4f} |",
        f"| Macro F1 delta | {summary['macro_f1_delta_holyc_minus_llama']:.4f} |",
        f"| Agreement | {summary['agreement']:.4f} |",
        "",
    ]
    paired = summary.get("paired_correctness", {})
    mcnemar = summary.get("mcnemar_exact", {})
    if paired and mcnemar:
        lines.extend(
            [
                "## Paired Correctness",
                "",
                "| Both correct | Both wrong | HolyC only correct | llama.cpp only correct | Discordant | McNemar p-value | Method |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                f"| {paired['both_correct']} | {paired['both_wrong']} | "
                f"{paired['holyc_only_correct']} | {paired['llama_only_correct']} | "
                f"{mcnemar['discordant_count']} | {mcnemar['p_value']:.6f} | {mcnemar['method']} |",
                "",
            ]
        )
    lines.extend(
        [
            "## Score Calibration",
            "",
            "| Engine | Score coverage | Mean confidence | Accuracy when scored | Brier score | ECE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for engine_label, key in (("HolyC", "holyc_calibration"), ("llama.cpp", "llama_calibration")):
        metrics = summary[key]
        lines.append(
            f"| {engine_label} | {metrics['scored_count']}/{metrics['total_count']} "
            f"({metrics['score_coverage']:.4f}) | {metrics['mean_confidence']:.4f} | "
            f"{metrics['accuracy_when_scored']:.4f} | {metrics['brier_score']:.4f} | {metrics['ece']:.4f} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Score Ranking",
            "",
            "| Engine | Score coverage | Top-1 | Top-2 | Top-3 | Mean gold rank | MRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for engine_label, key in (("HolyC", "holyc_rank_metrics"), ("llama.cpp", "llama_rank_metrics")):
        metrics = summary[key]
        lines.append(
            f"| {engine_label} | {metrics['scored_count']}/{metrics['total_count']} "
            f"({metrics['score_coverage']:.4f}) | {metrics['top_1_accuracy']:.4f} | "
            f"{metrics['top_2_accuracy']:.4f} | {metrics['top_3_accuracy']:.4f} | "
            f"{metrics['mean_gold_rank']:.4f} | {metrics['mean_reciprocal_rank']:.4f} |"
        )
    lines.append("")
    breakdown = summary.get("dataset_breakdown", [])
    if breakdown:
        lines.extend(
            [
                "## Dataset Breakdown",
                "",
                "| Dataset | Split | Records | HolyC accuracy | llama.cpp accuracy | Accuracy delta | Agreement |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in breakdown:
            lines.append(
                f"| {item['dataset']} | {item['split']} | {item['record_count']} | "
                f"{item['holyc_accuracy']:.4f} | {item['llama_accuracy']:.4f} | "
                f"{item['accuracy_delta_holyc_minus_llama']:.4f} | {item['agreement']:.4f} |"
            )
        lines.append("")
    intervals = summary.get("confidence_intervals", {})
    if intervals:
        lines.extend(
            [
                "## Confidence Intervals",
                "",
                "| Metric | Point | Lower | Upper | Confidence | Method |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for metric in ("holyc_accuracy", "llama_accuracy", "agreement"):
            interval = intervals[metric]
            lines.append(
                f"| {metric} | {interval['point']:.4f} | {interval['lower']:.4f} | "
                f"{interval['upper']:.4f} | {interval['confidence_level']:.2f} | {interval['method']} |"
            )
        lines.append("")
    lines.extend(["## Quality Gates", ""])
    if report["regressions"]:
        lines.append("| Metric | Value | Threshold | Finding |")
        lines.append("| --- | ---: | ---: | --- |")
        for regression in report["regressions"]:
            lines.append(
                f"| {regression['metric']} | {regression['value']:.4f} | "
                f"{regression['threshold']:.4f} | {regression['message']} |"
            )
    else:
        lines.append("No quality gate regressions.")
    lines.extend(
        [
            "",
            "## Per-Answer F1",
            "",
            "| Answer index | HolyC support | HolyC F1 | llama.cpp support | llama.cpp F1 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for holyc_metric, llama_metric in zip(summary["holyc_per_answer_index"], summary["llama_per_answer_index"]):
        lines.append(
            f"| {holyc_metric['answer_index']} | {holyc_metric['support']} | {holyc_metric['f1']:.4f} | "
            f"{llama_metric['support']} | {llama_metric['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Confusion Matrices",
            "",
            "Rows are gold answer indexes; columns are predicted answer indexes.",
            "",
            "### HolyC",
            "",
        ]
    )
    lines.extend(confusion_matrix_markdown(summary["holyc_confusion_matrix"]))
    lines.extend(["", "### llama.cpp", ""])
    lines.extend(confusion_matrix_markdown(summary["llama_confusion_matrix"]))
    lines.extend(
        [
            "",
            "## Disagreements",
            "",
        ]
    )
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


def confusion_matrix_markdown(confusion: dict[str, Any]) -> list[str]:
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    header = "| Gold \\ Pred | " + " | ".join(str(label) for label in labels) + " |"
    separator = "| ---: | " + " | ".join("---:" for _ in labels) + " |"
    lines = [header, separator]
    for label, counts in zip(labels, matrix):
        lines.append("| " + str(label) + " | " + " | ".join(str(count) for count in counts) + " |")
    return lines


def find_regressions(
    summary: dict[str, Any],
    *,
    min_holyc_accuracy: float | None = None,
    min_agreement: float | None = None,
    max_accuracy_drop: float | None = None,
) -> list[EvalRegression]:
    regressions: list[EvalRegression] = []
    if min_holyc_accuracy is not None and summary["holyc_accuracy"] < min_holyc_accuracy:
        regressions.append(
            EvalRegression(
                metric="holyc_accuracy",
                value=summary["holyc_accuracy"],
                threshold=min_holyc_accuracy,
                message=(
                    f"HolyC accuracy {summary['holyc_accuracy']:.4f} "
                    f"is below minimum {min_holyc_accuracy:.4f}"
                ),
            )
        )
    if min_agreement is not None and summary["agreement"] < min_agreement:
        regressions.append(
            EvalRegression(
                metric="agreement",
                value=summary["agreement"],
                threshold=min_agreement,
                message=f"engine agreement {summary['agreement']:.4f} is below minimum {min_agreement:.4f}",
            )
        )
    if max_accuracy_drop is not None:
        allowed_delta = -abs(max_accuracy_drop)
        delta = summary["accuracy_delta_holyc_minus_llama"]
        if delta < allowed_delta:
            regressions.append(
                EvalRegression(
                    metric="accuracy_delta_holyc_minus_llama",
                    value=delta,
                    threshold=allowed_delta,
                    message=(
                        f"HolyC accuracy delta {delta:.4f} is below allowed drop "
                        f"{allowed_delta:.4f}"
                    ),
                )
            )
    return regressions


def write_junit(regressions: list[EvalRegression], path: Path) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_compare",
            "tests": "1",
            "failures": str(len(regressions)),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"classname": "eval_compare", "name": "quality_gates"})
    if regressions:
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "eval_regression",
                "message": "; ".join(regression.message for regression in regressions),
            },
        )
        failure.text = "\n".join(
            f"{regression.metric}: value={regression.value:.6f} threshold={regression.threshold:.6f}"
            for regression in regressions
        )
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


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
                "holyc_confidence",
                "llama_confidence",
                "holyc_margin",
                "llama_margin",
                "holyc_gold_rank",
                "llama_gold_rank",
            ],
            lineterminator="\n",
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
    regressions = find_regressions(
        summary,
        min_holyc_accuracy=args.min_holyc_accuracy,
        min_agreement=args.min_agreement,
        max_accuracy_drop=args.max_accuracy_drop,
    )
    report = {
        "dataset": args.dataset,
        "generated_at": iso_now(),
        "gold_sha256": file_sha256(gold_path),
        "holyc_predictions_sha256": file_sha256(holyc_path),
        "llama_predictions_sha256": file_sha256(llama_path),
        "max_accuracy_drop": args.max_accuracy_drop,
        "model": args.model,
        "min_agreement": args.min_agreement,
        "min_holyc_accuracy": args.min_holyc_accuracy,
        "quantization": args.quantization,
        "regressions": [asdict(regression) for regression in regressions],
        "rows": [asdict(row) for row in rows],
        "split": args.split,
        "status": "fail" if regressions else "pass",
        "summary": summary,
    }
    stem = args.output_stem
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    csv_path = output_dir / f"{stem}.csv"
    junit_path = output_dir / f"{stem}_junit.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv_report(csv_path, rows)
    write_junit(regressions, junit_path)
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
    parser.add_argument("--min-holyc-accuracy", type=float, help="Minimum HolyC accuracy before CI gate failure")
    parser.add_argument("--min-agreement", type=float, help="Minimum HolyC/llama.cpp agreement before CI gate failure")
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        help="Allowed HolyC accuracy drop versus llama.cpp before CI gate failure",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Wilson confidence level for accuracy/agreement intervals",
    )
    parser.add_argument("--fail-on-regression", action="store_true", help="Return non-zero when quality gates fail")
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
        summary = add_confidence_intervals(summary, args.confidence_level)
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
    regressions = find_regressions(
        summary,
        min_holyc_accuracy=args.min_holyc_accuracy,
        min_agreement=args.min_agreement,
        max_accuracy_drop=args.max_accuracy_drop,
    )
    print(f"regressions={len(regressions)}")
    if args.fail_on_regression and regressions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
