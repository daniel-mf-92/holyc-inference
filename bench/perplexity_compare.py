#!/usr/bin/env python3
"""Offline perplexity comparator for HolyC vs llama.cpp logprob outputs.

The comparator consumes local HolyC and llama.cpp JSON, JSONL, or CSV records,
aligns rows by record id, computes token-weighted negative log likelihood and
perplexity, and writes JSON plus Markdown reports under bench/results. It is
host-side only and does not launch QEMU or use network services.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
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
class PerplexityRecord:
    record_id: str
    token_count: int
    total_nll: float

    @property
    def nll_per_token(self) -> float:
        return self.total_nll / self.token_count

    @property
    def perplexity(self) -> float:
        return safe_exp(self.nll_per_token)


@dataclass(frozen=True)
class CompareRow:
    record_id: str
    holyc_token_count: int
    llama_token_count: int
    holyc_nll_per_token: float
    llama_nll_per_token: float
    nll_delta_holyc_minus_llama: float
    holyc_perplexity: float
    llama_perplexity: float
    perplexity_delta_holyc_minus_llama: float


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_exp(value: float) -> float:
    if value > 709.0:
        return math.inf
    return math.exp(value)


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def case_id(row: dict[str, Any], row_label: str) -> str:
    value = row.get("id") or row.get("record_id") or row.get("question_id") or row.get("prompt_id")
    if value is None or str(value).strip() == "":
        raise ValueError(f"{row_label}: missing record id")
    return str(value).strip()


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
    raise ValueError(f"{path}: unsupported format; use JSON, JSONL, or CSV")


def parse_float(value: Any, row_label: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row_label}: {field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{row_label}: {field} must be finite")
    return parsed


def parse_int(value: Any, row_label: str, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return len(value)
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row_label}: {field} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{row_label}: {field} must be positive")
    return parsed


def parse_float_list(value: Any, row_label: str, field: str) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{row_label}: {field} string must be a JSON list: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"{row_label}: {field} must be a non-empty list")
    return [parse_float(item, row_label, field) for item in value]


def normalize_record(row: dict[str, Any], source: Path, index: int) -> PerplexityRecord:
    row_label = f"{source}:{index + 1}"
    record_id = case_id(row, row_label)
    logprobs = parse_float_list(first_present(row, LOGPROB_KEYS), row_label, "token_logprobs")
    token_count = parse_int(first_present(row, TOKEN_COUNT_KEYS), row_label, "token_count")

    if logprobs is not None:
        if token_count is not None and token_count != len(logprobs):
            raise ValueError(
                f"{row_label}: token_count {token_count} does not match {len(logprobs)} logprobs"
            )
        return PerplexityRecord(record_id=record_id, token_count=len(logprobs), total_nll=-sum(logprobs))

    if token_count is None:
        raise ValueError(f"{row_label}: missing token_logprobs or token_count")

    total_nll_value = first_present(row, TOTAL_NLL_KEYS)
    if total_nll_value is not None:
        total_nll = parse_float(total_nll_value, row_label, "total_nll")
        if total_nll < 0:
            raise ValueError(f"{row_label}: total_nll must be non-negative")
        return PerplexityRecord(record_id=record_id, token_count=token_count, total_nll=total_nll)

    mean_nll_value = first_present(row, MEAN_NLL_KEYS)
    if mean_nll_value is not None:
        mean_nll = parse_float(mean_nll_value, row_label, "mean_nll")
        if mean_nll < 0:
            raise ValueError(f"{row_label}: mean_nll must be non-negative")
        return PerplexityRecord(record_id=record_id, token_count=token_count, total_nll=mean_nll * token_count)

    perplexity_value = first_present(row, PERPLEXITY_KEYS)
    if perplexity_value is not None:
        perplexity = parse_float(perplexity_value, row_label, "perplexity")
        if perplexity <= 0:
            raise ValueError(f"{row_label}: perplexity must be positive")
        return PerplexityRecord(
            record_id=record_id,
            token_count=token_count,
            total_nll=math.log(perplexity) * token_count,
        )

    raise ValueError(f"{row_label}: missing token_logprobs, total_nll, mean_nll, or perplexity")


def load_records(path: Path) -> dict[str, PerplexityRecord]:
    records: dict[str, PerplexityRecord] = {}
    for index, row in enumerate(read_rows(path)):
        record = normalize_record(row, path, index)
        if record.record_id in records:
            raise ValueError(f"{path}: duplicate record id {record.record_id!r}")
        records[record.record_id] = record
    if not records:
        raise ValueError(f"{path}: no perplexity records found")
    return records


def summarize(records: Iterable[PerplexityRecord]) -> dict[str, Any]:
    rows = list(records)
    token_count = sum(row.token_count for row in rows)
    total_nll = sum(row.total_nll for row in rows)
    nll_per_token = total_nll / token_count if token_count else math.nan
    return {
        "record_count": len(rows),
        "token_count": token_count,
        "total_nll": total_nll,
        "nll_per_token": nll_per_token,
        "perplexity": safe_exp(nll_per_token),
    }


def compare(
    holyc: dict[str, PerplexityRecord],
    llama: dict[str, PerplexityRecord],
    allow_token_count_mismatch: bool = False,
) -> tuple[list[CompareRow], dict[str, Any]]:
    missing_holyc = sorted(set(llama) - set(holyc))
    missing_llama = sorted(set(holyc) - set(llama))
    if missing_holyc:
        raise ValueError(f"HolyC records missing {len(missing_holyc)} ids: {', '.join(missing_holyc[:5])}")
    if missing_llama:
        raise ValueError(f"llama.cpp records missing {len(missing_llama)} ids: {', '.join(missing_llama[:5])}")

    mismatched_ids = [
        record_id
        for record_id in sorted(holyc)
        if holyc[record_id].token_count != llama[record_id].token_count
    ]
    if mismatched_ids and not allow_token_count_mismatch:
        raise ValueError(
            "token count mismatch for "
            f"{len(mismatched_ids)} ids: {', '.join(mismatched_ids[:5])}; "
            "pass --allow-token-count-mismatch to report anyway"
        )

    rows: list[CompareRow] = []
    for record_id in sorted(holyc):
        holyc_row = holyc[record_id]
        llama_row = llama[record_id]
        rows.append(
            CompareRow(
                record_id=record_id,
                holyc_token_count=holyc_row.token_count,
                llama_token_count=llama_row.token_count,
                holyc_nll_per_token=holyc_row.nll_per_token,
                llama_nll_per_token=llama_row.nll_per_token,
                nll_delta_holyc_minus_llama=holyc_row.nll_per_token - llama_row.nll_per_token,
                holyc_perplexity=holyc_row.perplexity,
                llama_perplexity=llama_row.perplexity,
                perplexity_delta_holyc_minus_llama=holyc_row.perplexity - llama_row.perplexity,
            )
        )

    holyc_summary = summarize(holyc.values())
    llama_summary = summarize(llama.values())
    summary = {
        "record_count": len(rows),
        "token_count_mismatches": len(mismatched_ids),
        "holyc": holyc_summary,
        "llama": llama_summary,
        "nll_delta_holyc_minus_llama": holyc_summary["nll_per_token"] - llama_summary["nll_per_token"],
        "perplexity_delta_holyc_minus_llama": holyc_summary["perplexity"] - llama_summary["perplexity"],
        "max_abs_record_nll_delta": max((abs(row.nll_delta_holyc_minus_llama) for row in rows), default=0.0),
    }
    return rows, summary


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Perplexity Compare Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Dataset: {report['dataset'] or '-'}",
        f"Split: {report['split'] or '-'}",
        f"Quantization: {report['quantization'] or '-'}",
        f"Model: {report['model'] or '-'}",
        "",
        "## Summary",
        "",
        "| Metric | HolyC | llama.cpp | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Records | {summary['holyc']['record_count']} | {summary['llama']['record_count']} | - |",
        f"| Tokens | {summary['holyc']['token_count']} | {summary['llama']['token_count']} | - |",
        f"| NLL/token | {summary['holyc']['nll_per_token']:.6f} | "
        f"{summary['llama']['nll_per_token']:.6f} | {summary['nll_delta_holyc_minus_llama']:.6f} |",
        f"| Perplexity | {summary['holyc']['perplexity']:.6f} | "
        f"{summary['llama']['perplexity']:.6f} | {summary['perplexity_delta_holyc_minus_llama']:.6f} |",
        f"| Token count mismatches | {summary['token_count_mismatches']} | - | - |",
        "",
        "## Largest NLL Deltas",
        "",
    ]
    rows = sorted(report["rows"], key=lambda row: abs(row["nll_delta_holyc_minus_llama"]), reverse=True)
    if rows:
        lines.append("| ID | HolyC NLL/token | llama.cpp NLL/token | Delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in rows[: report["top_n"]]:
            lines.append(
                f"| {row['record_id']} | {row['holyc_nll_per_token']:.6f} | "
                f"{row['llama_nll_per_token']:.6f} | {row['nll_delta_holyc_minus_llama']:.6f} |"
            )
    else:
        lines.append("No aligned records.")
    return "\n".join(lines) + "\n"


def write_csv_report(path: Path, rows: list[CompareRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "holyc_token_count",
                "llama_token_count",
                "holyc_nll_per_token",
                "llama_nll_per_token",
                "nll_delta_holyc_minus_llama",
                "holyc_perplexity",
                "llama_perplexity",
                "perplexity_delta_holyc_minus_llama",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_report(
    rows: list[CompareRow],
    summary: dict[str, Any],
    args: argparse.Namespace,
    holyc_path: Path,
    llama_path: Path,
) -> tuple[Path, Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "generated_at": iso_now(),
        "holyc_sha256": file_sha256(holyc_path),
        "llama_sha256": file_sha256(llama_path),
        "model": args.model,
        "quantization": args.quantization,
        "rows": [asdict(row) for row in rows],
        "split": args.split,
        "summary": summary,
        "top_n": args.top_n,
    }
    json_path = args.output_dir / f"{args.output_stem}.json"
    md_path = args.output_dir / f"{args.output_stem}.md"
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv_report(csv_path, rows)
    return json_path, md_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holyc", type=Path, required=True, help="HolyC logprob/perplexity JSON/JSONL/CSV")
    parser.add_argument("--llama", type=Path, required=True, help="llama.cpp logprob/perplexity JSON/JSONL/CSV")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--allow-token-count-mismatch", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="perplexity_compare_latest")
    parser.add_argument("--top-n", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        holyc = load_records(args.holyc)
        llama = load_records(args.llama)
        rows, summary = compare(holyc, llama, args.allow_token_count_mismatch)
        json_path, md_path, csv_path = write_report(rows, summary, args, args.holyc, args.llama)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_csv={csv_path}")
    print(f"holyc_perplexity={summary['holyc']['perplexity']:.6f}")
    print(f"llama_perplexity={summary['llama']['perplexity']:.6f}")
    print(f"ppl_delta_holyc_minus_llama={summary['perplexity_delta_holyc_minus_llama']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
