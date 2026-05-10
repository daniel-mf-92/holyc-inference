#!/usr/bin/env python3
"""Audit eval predictions for declared-prediction vs score-order drift.

This host-side tool reads local scored prediction artifacts only. It never
launches QEMU, touches the TempleOS guest, or performs network I/O.
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


RESULT_KEYS = ("predictions", "results", "rows", "records")
ID_KEYS = ("id", "record_id", "question_id", "prompt_id")
PREDICTION_KEYS = ("prediction", "predicted", "predicted_index", "answer", "answer_index", "choice")
SCORE_KEYS = ("scores", "logprobs", "choice_scores", "choice_logprobs")


@dataclass(frozen=True)
class ScoreOrderRecord:
    source: str
    engine: str
    row_number: int
    record_id: str
    prediction_index: int | None
    score_count: int
    top_score_index: int | None
    top_score: float | None
    runner_up_score: float | None
    top_margin: float | None
    top_tie_count: int | None
    checked: bool
    matches_top_score: bool | None


@dataclass(frozen=True)
class EngineSummary:
    source: str
    engine: str
    record_count: int
    checked_count: int
    match_count: int
    mismatch_count: int
    top_tie_count: int
    missing_prediction_count: int
    missing_scores_count: int
    invalid_count: int
    match_rate: float
    mean_top_margin: float
    min_top_margin: float


@dataclass(frozen=True)
class Finding:
    severity: str
    engine: str
    record_id: str
    metric: str
    value: float | int | str
    limit: float | int | str
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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
    for key in ID_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"{label}: missing record id")


def parse_prediction(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            if len(stripped) == 1 and "A" <= stripped.upper() <= "Z":
                return ord(stripped.upper()) - ord("A")
            return None
        if math.isfinite(parsed) and parsed.is_integer():
            return int(parsed)
    return None


def prediction_index(row: dict[str, Any]) -> int | None:
    for key in PREDICTION_KEYS:
        parsed = parse_prediction(row.get(key))
        if parsed is not None:
            return parsed
    return None


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
        if isinstance(item, bool):
            raise ValueError(f"{label}: score field must contain only numbers")
        try:
            score = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: score field must contain only numbers") from exc
        if not math.isfinite(score):
            raise ValueError(f"{label}: score field must contain finite numbers")
        scores.append(score)
    return scores


def score_vector(row: dict[str, Any], label: str) -> list[float] | None:
    for key in SCORE_KEYS:
        scores = parse_scores(row.get(key), label)
        if scores is not None:
            return scores
    return None


def top_stats(scores: list[float], epsilon: float) -> tuple[int, float, float | None, float | None, int]:
    ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
    top_index, top_score = ranked[0]
    top_tie_count = sum(1 for _, score in ranked if abs(score - top_score) <= epsilon)
    if len(ranked) == 1:
        return top_index, top_score, None, None, top_tie_count
    runner_up = ranked[1][1]
    return top_index, top_score, runner_up, top_score - runner_up, top_tie_count


def load_engine_records(
    spec: str,
    *,
    epsilon: float,
    require_both: bool,
) -> tuple[str, list[ScoreOrderRecord], list[Finding]]:
    if "=" not in spec:
        raise ValueError("--predictions must use ENGINE=PATH")
    engine, raw_path = spec.split("=", 1)
    engine = engine.strip()
    if not engine:
        raise ValueError("--predictions engine label must be non-empty")
    path = Path(raw_path)
    records: list[ScoreOrderRecord] = []
    findings: list[Finding] = []
    seen: set[str] = set()

    for row_number, row in enumerate(read_rows(path), 1):
        label = f"{path}:{row_number}"
        try:
            record_id = row_id(row, label)
            predicted = prediction_index(row)
            scores = score_vector(row, label)
        except ValueError as exc:
            findings.append(Finding("error", engine, f"row:{row_number}", "parse_error", "invalid", "valid", str(exc)))
            continue
        if record_id in seen:
            findings.append(Finding("error", engine, record_id, "duplicate_id", 1, 0, f"{engine} duplicate prediction id {record_id}"))
        seen.add(record_id)

        if scores is None:
            records.append(
                ScoreOrderRecord(str(path), engine, row_number, record_id, predicted, 0, None, None, None, None, None, False, None)
            )
            if require_both:
                findings.append(Finding("error", engine, record_id, "missing_scores", "missing", "present", f"{engine} {record_id} is missing scores"))
            continue
        top_index, top_score, runner_up_score, margin, tie_count = top_stats(scores, epsilon)
        checked = predicted is not None
        matches = predicted == top_index if checked else None
        records.append(
            ScoreOrderRecord(
                str(path),
                engine,
                row_number,
                record_id,
                predicted,
                len(scores),
                top_index,
                top_score,
                runner_up_score,
                margin,
                tie_count,
                checked,
                matches,
            )
        )
        if predicted is None:
            if require_both:
                findings.append(Finding("error", engine, record_id, "missing_prediction", "missing", "present", f"{engine} {record_id} is missing a declared prediction"))
            continue
        if predicted < 0 or predicted >= len(scores):
            findings.append(Finding("error", engine, record_id, "prediction_out_of_range", predicted, f"0..{len(scores) - 1}", f"{engine} {record_id} prediction index is outside the score vector"))
        elif predicted != top_index:
            findings.append(Finding("error", engine, record_id, "prediction_score_mismatch", predicted, top_index, f"{engine} {record_id} declared prediction does not match top score index"))
        if tie_count > 1:
            findings.append(Finding("error", engine, record_id, "top_score_tie", tie_count, 1, f"{engine} {record_id} has an ambiguous top-score tie"))

    return engine, records, findings


def summarize(source: str, engine: str, records: list[ScoreOrderRecord], invalid_count: int) -> EngineSummary:
    checked = [record for record in records if record.checked]
    matches = [record for record in checked if record.matches_top_score]
    mismatches = [record for record in checked if record.matches_top_score is False]
    margins = [record.top_margin for record in records if record.top_margin is not None]
    return EngineSummary(
        source=source,
        engine=engine,
        record_count=len(records) + invalid_count,
        checked_count=len(checked),
        match_count=len(matches),
        mismatch_count=len(mismatches),
        top_tie_count=sum(1 for record in records if record.top_tie_count is not None and record.top_tie_count > 1),
        missing_prediction_count=sum(1 for record in records if record.prediction_index is None),
        missing_scores_count=sum(1 for record in records if record.score_count == 0),
        invalid_count=invalid_count,
        match_rate=len(matches) / len(checked) if checked else 0.0,
        mean_top_margin=sum(margins) / len(margins) if margins else 0.0,
        min_top_margin=min(margins) if margins else 0.0,
    )


def audit(args: argparse.Namespace) -> tuple[list[ScoreOrderRecord], list[EngineSummary], list[Finding]]:
    records: list[ScoreOrderRecord] = []
    summaries: list[EngineSummary] = []
    findings: list[Finding] = []
    for spec in args.predictions:
        try:
            engine, engine_records, engine_findings = load_engine_records(spec, epsilon=args.epsilon, require_both=args.require_both)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            engine = spec.split("=", 1)[0] if "=" in spec else "unknown"
            findings.append(Finding("error", engine, "", "load_error", "invalid", "valid", str(exc)))
            continue
        findings.extend(engine_findings)
        records.extend(engine_records)
        source = engine_records[0].source if engine_records else spec.split("=", 1)[1]
        invalid_count = sum(1 for finding in engine_findings if finding.metric == "parse_error")
        summary = summarize(source, engine, engine_records, invalid_count)
        summaries.append(summary)
        if summary.checked_count < args.min_checked_records:
            findings.append(Finding("error", engine, "", "checked_count", summary.checked_count, args.min_checked_records, f"{engine} has too few rows with both prediction and scores"))
        if summary.match_rate < args.min_match_rate:
            findings.append(Finding("error", engine, "", "match_rate", summary.match_rate, args.min_match_rate, f"{engine} prediction/score match rate is below threshold"))
        if summary.top_tie_count > args.max_top_ties:
            findings.append(Finding("error", engine, "", "top_tie_count", summary.top_tie_count, args.max_top_ties, f"{engine} has too many top-score ties"))
    return records, summaries, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, output_stem: str, records: list[ScoreOrderRecord], summaries: list[EngineSummary], findings: list[Finding]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "eval_score_order_audit",
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "engines": [summary.engine for summary in summaries],
            "record_count": sum(summary.record_count for summary in summaries),
            "checked_count": sum(summary.checked_count for summary in summaries),
            "mismatch_count": sum(summary.mismatch_count for summary in summaries),
            "finding_count": len(findings),
        },
        "engine_summaries": [asdict(summary) for summary in summaries],
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    (output_dir / f"{output_stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / f"{output_stem}.csv", [asdict(summary) for summary in summaries], list(EngineSummary.__dataclass_fields__))
    write_csv(output_dir / f"{output_stem}_records.csv", [asdict(record) for record in records], list(ScoreOrderRecord.__dataclass_fields__))
    write_csv(output_dir / f"{output_stem}_findings.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))

    lines = [
        "# Eval Score Order Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Engines: {', '.join(payload['summary']['engines']) if payload['summary']['engines'] else '(none)'}",
        f"- Checked records: {payload['summary']['checked_count']}",
        f"- Prediction/score mismatches: {payload['summary']['mismatch_count']}",
        f"- Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.append("## Findings")
        for finding in findings:
            target = f"{finding.engine}:{finding.record_id}" if finding.record_id else finding.engine
            lines.append(f"- {finding.severity}: {target} {finding.metric}={finding.value} limit={finding.limit} - {finding.message}")
    else:
        lines.append("No score order findings.")
    (output_dir / f"{output_stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    suite = ET.Element("testsuite", name="holyc_eval_score_order_audit", tests="1", failures=str(1 if findings else 0), errors="0")
    case = ET.SubElement(suite, "testcase", name="score_order")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} score order findings")
        failure.text = "\n".join(finding.message for finding in findings)
    ET.ElementTree(suite).write(output_dir / f"{output_stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit prediction fields against local score-vector ordering")
    parser.add_argument("--predictions", action="append", required=True, help="Prediction artifact as ENGINE=PATH; may be repeated")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_score_order_audit_latest")
    parser.add_argument("--epsilon", type=float, default=1e-9, help="Score tie tolerance")
    parser.add_argument("--min-checked-records", type=int, default=1, help="Minimum rows per engine containing both prediction and scores")
    parser.add_argument("--min-match-rate", type=float, default=1.0, help="Minimum prediction/top-score match rate per engine")
    parser.add_argument("--max-top-ties", type=int, default=0, help="Maximum allowed top-score ties per engine")
    parser.add_argument("--require-both", action="store_true", help="Fail rows missing either a prediction or scores")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epsilon < 0:
        parser.error("--epsilon must be non-negative")
    if args.min_checked_records < 0:
        parser.error("--min-checked-records must be non-negative")
    if not 0 <= args.min_match_rate <= 1:
        parser.error("--min-match-rate must be between 0 and 1")
    if args.max_top_ties < 0:
        parser.error("--max-top-ties must be non-negative")
    records, summaries, findings = audit(args)
    write_outputs(args.output_dir, args.output_stem, records, summaries, findings)
    print(f"eval_score_order_audit={'fail' if findings else 'pass'} findings={len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
