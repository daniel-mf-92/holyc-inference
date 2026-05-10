#!/usr/bin/env python3
"""Audit QEMU prompt benchmark artifacts for prompt-length bucket coverage.

This host-side tool reads saved benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
DEFAULT_BUCKETS = ("short:0:255", "medium:256:1023", "long:1024:")
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class LengthBucket:
    name: str
    min_bytes: int
    max_bytes: int | None


@dataclass(frozen=True)
class PromptLengthSample:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    prompt_sha256: str
    phase: str
    exit_class: str
    prompt_bytes: int | None
    tokens: int | None
    wall_tok_per_s: float | None
    ttft_us: float | None


@dataclass(frozen=True)
class BucketSummary:
    bucket: str
    min_bytes: int
    max_bytes: int | None
    rows: int
    successful_rows: int
    failed_rows: int
    failure_pct: float
    unique_prompts: int
    tokens_total: int
    prompt_bytes_min: int | None
    prompt_bytes_max: int | None
    wall_tok_per_s_p50: float | None
    wall_tok_per_s_p05: float | None
    ttft_us_p50: float | None
    ttft_us_p95: float | None
    sources: str


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    bucket: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_int(value: Any) -> int | None:
    number = finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def row_text(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def iter_input_files(paths: Iterable[Path], patterns: list[str]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            seen: set[Path] = set()
            for pattern in patterns:
                for child in sorted(path.rglob(pattern)):
                    if child.is_file() and child not in seen:
                        seen.add(child)
                        yield child
        elif path.is_file():
            yield path


def flatten_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return

    inherited = {key: value for key, value in payload.items() if key not in RESULT_KEYS and key != "warmups"}
    yielded = False
    for key in RESULT_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list):
            yielded = True
            for item in rows:
                if isinstance(item, dict):
                    merged = dict(inherited)
                    merged.update(item)
                    yield merged
    if not yielded:
        yield payload


def load_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        yield from flatten_json_payload(json.loads(path.read_text(encoding="utf-8")))
        return
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield from flatten_json_payload(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        return
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    raise ValueError(f"{path}: unsupported input suffix {path.suffix}")


def parse_bucket(value: str) -> LengthBucket:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("bucket must use name:min:max, with empty max allowed")
    name, min_text, max_text = parts
    if not name:
        raise argparse.ArgumentTypeError("bucket name must be non-empty")
    try:
        min_bytes = int(min_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bucket min must be an integer") from exc
    if min_bytes < 0:
        raise argparse.ArgumentTypeError("bucket min must be non-negative")
    max_bytes: int | None = None
    if max_text:
        try:
            max_bytes = int(max_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("bucket max must be an integer") from exc
        if max_bytes < min_bytes:
            raise argparse.ArgumentTypeError("bucket max must be >= min")
    return LengthBucket(name=name, min_bytes=min_bytes, max_bytes=max_bytes)


def bucket_for_prompt(prompt_bytes: int | None, buckets: list[LengthBucket]) -> LengthBucket | None:
    if prompt_bytes is None:
        return None
    for bucket in buckets:
        if prompt_bytes < bucket.min_bytes:
            continue
        if bucket.max_bytes is not None and prompt_bytes > bucket.max_bytes:
            continue
        return bucket
    return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sample_from_row(source: Path, row_number: int, row: dict[str, Any]) -> PromptLengthSample:
    return PromptLengthSample(
        source=str(source),
        row=row_number,
        profile=row_text(row, "profile"),
        model=row_text(row, "model"),
        quantization=row_text(row, "quantization"),
        prompt=row_text(row, "prompt", "prompt_id"),
        prompt_sha256=row_text(row, "prompt_sha256", "guest_prompt_sha256", default=""),
        phase=row_text(row, "phase", default="measured"),
        exit_class=row_text(row, "exit_class"),
        prompt_bytes=finite_int(row.get("prompt_bytes")),
        tokens=finite_int(row.get("tokens")),
        wall_tok_per_s=finite_float(row.get("wall_tok_per_s")),
        ttft_us=finite_float(row.get("ttft_us")),
    )


def summarize_bucket(bucket: LengthBucket, samples: list[PromptLengthSample]) -> BucketSummary:
    successful = [sample for sample in samples if sample.exit_class == "ok"]
    failed = len(samples) - len(successful)
    prompt_byte_values = [sample.prompt_bytes for sample in samples if sample.prompt_bytes is not None]
    wall_tok_values = [sample.wall_tok_per_s for sample in successful if sample.wall_tok_per_s is not None]
    ttft_values = [sample.ttft_us for sample in successful if sample.ttft_us is not None]
    prompt_keys = {
        sample.prompt_sha256 or sample.prompt
        for sample in samples
        if sample.prompt_sha256 or sample.prompt
    }
    return BucketSummary(
        bucket=bucket.name,
        min_bytes=bucket.min_bytes,
        max_bytes=bucket.max_bytes,
        rows=len(samples),
        successful_rows=len(successful),
        failed_rows=failed,
        failure_pct=round((failed * 100.0 / len(samples)) if samples else 0.0, 6),
        unique_prompts=len(prompt_keys),
        tokens_total=sum(sample.tokens or 0 for sample in successful),
        prompt_bytes_min=min(prompt_byte_values) if prompt_byte_values else None,
        prompt_bytes_max=max(prompt_byte_values) if prompt_byte_values else None,
        wall_tok_per_s_p50=percentile(wall_tok_values, 0.50),
        wall_tok_per_s_p05=percentile(wall_tok_values, 0.05),
        ttft_us_p50=percentile(ttft_values, 0.50),
        ttft_us_p95=percentile(ttft_values, 0.95),
        sources=";".join(sorted({sample.source for sample in samples})),
    )


def collect_samples(paths: list[Path], patterns: list[str], include_warmups: bool) -> tuple[list[PromptLengthSample], list[Finding]]:
    samples: list[PromptLengthSample] = []
    findings: list[Finding] = []
    for path in iter_input_files(paths, patterns):
        try:
            rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "", str(exc)))
            continue
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                findings.append(Finding(str(path), index, "error", "invalid_row", "", "row must be an object"))
                continue
            sample = sample_from_row(path, index, row)
            if not include_warmups and sample.phase != "measured":
                continue
            if sample.prompt_bytes is None:
                findings.append(Finding(str(path), index, "error", "missing_prompt_bytes", "", "row has no finite integer prompt_bytes"))
            samples.append(sample)
    return samples, findings


def build_report(samples: list[PromptLengthSample], buckets: list[LengthBucket], findings: list[Finding], args: argparse.Namespace) -> dict[str, Any]:
    bucketed: dict[str, list[PromptLengthSample]] = {bucket.name: [] for bucket in buckets}
    for sample in samples:
        bucket = bucket_for_prompt(sample.prompt_bytes, buckets)
        if bucket is None:
            if sample.prompt_bytes is not None:
                findings.append(
                    Finding(
                        sample.source,
                        sample.row,
                        "error",
                        "unbucketed_prompt_bytes",
                        "",
                        f"prompt_bytes={sample.prompt_bytes} does not fit any configured bucket",
                    )
                )
            continue
        bucketed[bucket.name].append(sample)

    summaries = [summarize_bucket(bucket, bucketed[bucket.name]) for bucket in buckets]
    for summary in summaries:
        if args.require_buckets and summary.rows == 0:
            findings.append(Finding("", 0, "error", "empty_bucket", summary.bucket, "bucket has no benchmark rows"))
        if summary.successful_rows < args.min_successful_samples_per_bucket:
            findings.append(
                Finding(
                    "",
                    0,
                    "error",
                    "insufficient_successful_samples",
                    summary.bucket,
                    f"successful_rows={summary.successful_rows} below required {args.min_successful_samples_per_bucket}",
                )
            )
        if summary.unique_prompts < args.min_prompts_per_bucket:
            findings.append(
                Finding(
                    "",
                    0,
                    "error",
                    "insufficient_unique_prompts",
                    summary.bucket,
                    f"unique_prompts={summary.unique_prompts} below required {args.min_prompts_per_bucket}",
                )
            )
        if summary.rows and summary.failure_pct > args.max_failure_pct:
            findings.append(
                Finding(
                    "",
                    0,
                    "error",
                    "bucket_failure_pct",
                    summary.bucket,
                    f"failure_pct={summary.failure_pct:.3f} above allowed {args.max_failure_pct:.3f}",
                )
            )

    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "samples": len(samples),
            "successful_samples": sum(1 for sample in samples if sample.exit_class == "ok"),
            "failed_samples": sum(1 for sample in samples if sample.exit_class != "ok"),
            "buckets": len(summaries),
            "nonempty_buckets": sum(1 for summary in summaries if summary.rows),
            "findings": len(findings),
        },
        "config": {
            "include_warmups": args.include_warmups,
            "min_successful_samples_per_bucket": args.min_successful_samples_per_bucket,
            "min_prompts_per_bucket": args.min_prompts_per_bucket,
            "max_failure_pct": args.max_failure_pct,
        },
        "buckets": [asdict(summary) for summary in summaries],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [field.name for field in BucketSummary.__dataclass_fields__.values()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


def write_findings_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    fieldnames = [field.name for field in Finding.__dataclass_fields__.values()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(finding)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# QEMU Prompt Length Bucket Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "| bucket | rows | ok | failed | unique prompts | byte range | p50 wall tok/s | p95 TTFT us |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for bucket in report["buckets"]:
        byte_range = f"{bucket['prompt_bytes_min'] or ''}-{bucket['prompt_bytes_max'] or ''}"
        lines.append(
            "| {bucket} | {rows} | {ok} | {failed} | {prompts} | {byte_range} | {tok} | {ttft} |".format(
                bucket=bucket["bucket"],
                rows=bucket["rows"],
                ok=bucket["successful_rows"],
                failed=bucket["failed_rows"],
                prompts=bucket["unique_prompts"],
                byte_range=byte_range,
                tok="" if bucket["wall_tok_per_s_p50"] is None else f"{bucket['wall_tok_per_s_p50']:.6g}",
                ttft="" if bucket["ttft_us_p95"] is None else f"{bucket['ttft_us_p95']:.6g}",
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            prefix = f"{finding['source']}:{finding['row']}" if finding["source"] else finding["bucket"]
            lines.append(f"- `{finding['kind']}` {prefix}: {finding['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[dict[str, Any]]) -> None:
    suite = ET.Element("testsuite", name="qemu_prompt_length_bucket_audit", tests=str(max(1, len(findings))), failures=str(len(findings)))
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", name=f"{finding['kind']}:{finding['bucket'] or finding['row']}")
            failure = ET.SubElement(case, "failure", message=finding["detail"], type=finding["kind"])
            failure.text = json.dumps(finding, sort_keys=True)
    else:
        ET.SubElement(suite, "testcase", name="prompt_length_bucket_audit_passed")
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV files or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; may be repeated")
    parser.add_argument("--bucket", action="append", default=[], type=parse_bucket, help="Prompt byte bucket as name:min:max; empty max is open-ended")
    parser.add_argument("--include-warmups", action="store_true", help="Include warmup rows in bucket coverage")
    parser.add_argument("--require-buckets", action="store_true", help="Fail when any configured bucket is empty")
    parser.add_argument("--min-successful-samples-per-bucket", type=int, default=0)
    parser.add_argument("--min-prompts-per-bucket", type=int, default=0)
    parser.add_argument("--max-failure-pct", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_prompt_length_bucket_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_successful_samples_per_bucket < 0 or args.min_prompts_per_bucket < 0:
        parser.error("minimum bucket gates must be non-negative")
    if args.max_failure_pct < 0:
        parser.error("--max-failure-pct must be non-negative")

    buckets = args.bucket or [parse_bucket(value) for value in DEFAULT_BUCKETS]
    patterns = args.pattern or list(DEFAULT_PATTERNS)
    samples, findings = collect_samples(args.inputs, patterns, args.include_warmups)
    report = build_report(samples, buckets, findings, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", report["buckets"])
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", report["findings"])
    write_junit(args.output_dir / f"{stem}_junit.xml", report["findings"])
    print(f"{report['status']} buckets={report['summary']['buckets']} samples={report['summary']['samples']} findings={report['summary']['findings']}")
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
