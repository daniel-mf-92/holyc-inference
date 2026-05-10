#!/usr/bin/env python3
"""Audit perplexity/logprob input artifacts before HolyC-vs-llama comparison.

This host-side tool reads local JSON, JSONL, or CSV records only. It never
launches QEMU, touches the TempleOS guest, or uses network services.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RESULT_KEYS = ("perplexities", "results", "rows", "records")
LOGPROB_KEYS = ("token_logprobs", "token_log_probs", "logprobs", "log_probs")
TOTAL_NLL_KEYS = ("total_nll", "nll", "negative_log_likelihood")
MEAN_NLL_KEYS = ("mean_nll", "nll_per_token", "avg_nll")
TOKEN_COUNT_KEYS = ("token_count", "tokens", "eval_tokens", "num_tokens")
PERPLEXITY_KEYS = ("perplexity", "ppl")


@dataclass(frozen=True)
class InputRecord:
    source: str
    row: int
    record_id: str
    dataset: str
    split: str
    token_count: int | None
    total_nll: float | None
    mean_nll: float | None
    perplexity: float | None
    logprob_count: int
    derived_total_nll: float | None
    derived_mean_nll: float | None
    derived_perplexity: float | None


@dataclass(frozen=True)
class SourceSummary:
    source: str
    rows: int
    valid_rows: int
    duplicate_ids: int
    token_count: int
    datasets: str
    splits: str


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def metadata_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get(key)
    return "" if value is None else str(value).strip()


def finite_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def positive_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, list):
        return len(value)
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def parse_float_list(value: Any) -> list[float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list) or not value:
        return None
    values: list[float] = []
    for item in value:
        parsed = finite_float(item)
        if parsed is None:
            return None
        values.append(parsed)
    return values


def safe_exp(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 709.0:
        return math.inf
    return math.exp(value)


def close_enough(actual: float, expected: float, tolerance_pct: float) -> bool:
    tolerance = max(1e-9, abs(expected) * tolerance_pct / 100.0)
    return abs(actual - expected) <= tolerance


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


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
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
        return
    if suffix == ".json":
        yield from flatten_json_payload(json.loads(path.read_text(encoding="utf-8")))
        return
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
        return
    raise ValueError(f"{path}: unsupported format; use JSON, JSONL, or CSV")


def normalize_row(source: Path, row_number: int, row: dict[str, Any], args: argparse.Namespace) -> tuple[InputRecord, list[Finding]]:
    findings: list[Finding] = []
    record_id = str(row.get("id") or row.get("record_id") or row.get("question_id") or row.get("prompt_id") or "").strip()
    if not record_id:
        findings.append(Finding(str(source), row_number, "error", "missing_record_id", "id", "record id is required"))
        record_id = f"row-{row_number}"

    dataset = metadata_value(row, "dataset")
    split = metadata_value(row, "split")
    if args.require_dataset and not dataset:
        findings.append(Finding(str(source), row_number, "error", "missing_dataset", "dataset", "dataset metadata is required"))
    if args.require_split and not split:
        findings.append(Finding(str(source), row_number, "error", "missing_split", "split", "split metadata is required"))

    logprobs = parse_float_list(first_present(row, LOGPROB_KEYS))
    raw_logprobs = first_present(row, LOGPROB_KEYS)
    if raw_logprobs not in (None, "") and logprobs is None:
        findings.append(Finding(str(source), row_number, "error", "invalid_logprobs", "token_logprobs", "logprob field must be a non-empty numeric list"))
    if logprobs is not None:
        for index, value in enumerate(logprobs):
            if value > 0.0:
                findings.append(Finding(str(source), row_number, "error", "positive_logprob", "token_logprobs", f"token_logprobs[{index}] must be <= 0"))

    token_count = positive_int(first_present(row, TOKEN_COUNT_KEYS))
    if token_count is None and logprobs is not None:
        token_count = len(logprobs)
    elif token_count is None:
        findings.append(Finding(str(source), row_number, "error", "missing_token_count", "token_count", "token_count or token_logprobs is required"))
    elif logprobs is not None and token_count != len(logprobs):
        findings.append(Finding(str(source), row_number, "error", "token_count_mismatch", "token_count", f"token_count {token_count} != {len(logprobs)} logprobs"))

    total_nll = finite_float(first_present(row, TOTAL_NLL_KEYS))
    mean_nll = finite_float(first_present(row, MEAN_NLL_KEYS))
    perplexity = finite_float(first_present(row, PERPLEXITY_KEYS))
    if total_nll is not None and total_nll < 0:
        findings.append(Finding(str(source), row_number, "error", "negative_total_nll", "total_nll", "total NLL must be non-negative"))
    if mean_nll is not None and mean_nll < 0:
        findings.append(Finding(str(source), row_number, "error", "negative_mean_nll", "mean_nll", "mean NLL must be non-negative"))
    if perplexity is not None and perplexity < 1.0:
        findings.append(Finding(str(source), row_number, "error", "invalid_perplexity", "perplexity", "perplexity must be >= 1"))

    derived_total_nll = -sum(logprobs) if logprobs is not None else None
    if derived_total_nll is None and total_nll is not None:
        derived_total_nll = total_nll
    if derived_total_nll is None and mean_nll is not None and token_count:
        derived_total_nll = mean_nll * token_count
    if derived_total_nll is None and perplexity is not None and token_count:
        derived_total_nll = math.log(perplexity) * token_count

    derived_mean_nll = derived_total_nll / token_count if derived_total_nll is not None and token_count else None
    derived_perplexity = safe_exp(derived_mean_nll)

    if token_count and total_nll is not None and mean_nll is not None:
        expected_mean = total_nll / token_count
        if not close_enough(mean_nll, expected_mean, args.tolerance_pct):
            findings.append(Finding(str(source), row_number, "error", "mean_nll_drift", "mean_nll", f"stored {mean_nll:.8g}, expected {expected_mean:.8g}"))
    if derived_mean_nll is not None and perplexity is not None:
        expected_ppl = safe_exp(derived_mean_nll)
        if expected_ppl is not None and math.isfinite(expected_ppl) and not close_enough(perplexity, expected_ppl, args.tolerance_pct):
            findings.append(Finding(str(source), row_number, "error", "perplexity_drift", "perplexity", f"stored {perplexity:.8g}, expected {expected_ppl:.8g}"))
    if derived_total_nll is None:
        findings.append(Finding(str(source), row_number, "error", "missing_nll_signal", "nll", "need token_logprobs, total_nll, mean_nll, or perplexity"))

    return (
        InputRecord(
            source=str(source),
            row=row_number,
            record_id=record_id,
            dataset=dataset,
            split=split,
            token_count=token_count,
            total_nll=total_nll,
            mean_nll=mean_nll,
            perplexity=perplexity,
            logprob_count=len(logprobs or []),
            derived_total_nll=derived_total_nll,
            derived_mean_nll=derived_mean_nll,
            derived_perplexity=derived_perplexity,
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[InputRecord], list[SourceSummary], list[Finding]]:
    records: list[InputRecord] = []
    summaries: list[SourceSummary] = []
    findings: list[Finding] = []
    for source in paths:
        source_records: list[InputRecord] = []
        try:
            rows = list(read_rows(source))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(source), 0, "error", "load_error", "artifact", str(exc)))
            continue
        seen_ids: dict[str, int] = {}
        duplicate_ids = 0
        for row_number, row in enumerate(rows, 1):
            record, row_findings = normalize_row(source, row_number, row, args)
            if record.record_id in seen_ids:
                duplicate_ids += 1
                findings.append(Finding(str(source), row_number, "error", "duplicate_record_id", "id", f"{record.record_id} also appears on row {seen_ids[record.record_id]}"))
            seen_ids.setdefault(record.record_id, row_number)
            source_records.append(record)
            records.append(record)
            findings.extend(row_findings)
        if len(source_records) < args.min_records_per_source:
            findings.append(Finding(str(source), 0, "error", "min_records_per_source", "records", f"found {len(source_records)}, expected at least {args.min_records_per_source}"))
        source_tokens = sum(record.token_count or 0 for record in source_records)
        if source_tokens < args.min_tokens_per_source:
            findings.append(Finding(str(source), 0, "error", "min_tokens_per_source", "token_count", f"found {source_tokens}, expected at least {args.min_tokens_per_source}"))
        valid_rows = len([record for record in source_records if record.derived_total_nll is not None and record.token_count])
        summaries.append(
            SourceSummary(
                source=str(source),
                rows=len(source_records),
                valid_rows=valid_rows,
                duplicate_ids=duplicate_ids,
                token_count=source_tokens,
                datasets=",".join(sorted({record.dataset for record in source_records if record.dataset})),
                splits=",".join(sorted({record.split for record in source_records if record.split})),
            )
        )
    if len(records) < args.min_records:
        findings.append(Finding("-", 0, "error", "min_records", "records", f"found {len(records)}, expected at least {args.min_records}"))
    if sum(record.token_count or 0 for record in records) < args.min_tokens:
        findings.append(Finding("-", 0, "error", "min_tokens", "token_count", f"found {sum(record.token_count or 0 for record in records)}, expected at least {args.min_tokens}"))
    return records, summaries, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_report(records: list[InputRecord], summaries: list[SourceSummary], findings: list[Finding], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    stem = args.output_stem
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "summary": {
            "sources": len(summaries),
            "records": len(records),
            "valid_records": len([record for record in records if record.derived_total_nll is not None and record.token_count]),
            "tokens": sum(record.token_count or 0 for record in records),
            "findings": len(findings),
        },
        "thresholds": {
            "min_records": args.min_records,
            "min_records_per_source": args.min_records_per_source,
            "min_tokens": args.min_tokens,
            "min_tokens_per_source": args.min_tokens_per_source,
            "require_dataset": args.require_dataset,
            "require_split": args.require_split,
            "tolerance_pct": args.tolerance_pct,
        },
        "sources": [asdict(summary) for summary in summaries],
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", [asdict(record) for record in records], list(InputRecord.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_sources.csv", [asdict(summary) for summary in summaries], list(SourceSummary.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))

    markdown = ["# Perplexity Input Audit", "", f"Status: {status}", f"Sources: {len(summaries)}", f"Records: {len(records)}", f"Findings: {len(findings)}", ""]
    if findings:
        markdown.extend(["## Findings", ""])
        markdown.extend(f"- {finding.severity}: {finding.source}:{finding.row} {finding.kind} {finding.field} - {finding.detail}" for finding in findings)
    else:
        markdown.append("No perplexity input findings.")
    (args.output_dir / f"{stem}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    suite = ET.Element("testsuite", {"name": "holyc_perplexity_input_audit", "tests": "1", "failures": "1" if status == "fail" else "0"})
    case = ET.SubElement(suite, "testcase", {"name": "perplexity_inputs"})
    if status == "fail":
        failure = ET.SubElement(case, "failure", {"type": "perplexity_input_audit"})
        failure.text = "\n".join(f"{finding.kind}: {finding.source}:{finding.row} {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Perplexity JSON, JSONL, or CSV artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="perplexity_input_audit_latest")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-records-per-source", type=int, default=1)
    parser.add_argument("--min-tokens", type=int, default=1)
    parser.add_argument("--min-tokens-per-source", type=int, default=1)
    parser.add_argument("--require-dataset", action="store_true")
    parser.add_argument("--require-split", action="store_true")
    parser.add_argument("--tolerance-pct", type=float, default=0.001)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.min_records, args.min_records_per_source, args.min_tokens, args.min_tokens_per_source) < 0 or args.tolerance_pct < 0:
        parser.error("minimum gates and --tolerance-pct must be >= 0")
    records, summaries, findings = audit(args.inputs, args)
    write_report(records, summaries, findings, args)
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
