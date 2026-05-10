#!/usr/bin/env python3
"""Report HolyC vs llama.cpp eval quality by prompt-length bucket.

This host-side tool reads local gold/prediction artifacts only. It does not
launch QEMU, open sockets, or touch the TempleOS guest.
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

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_compare


@dataclass(frozen=True)
class BucketRow:
    bucket: str
    min_prompt_bytes: int
    max_prompt_bytes: int
    record_count: int
    holyc_correct: int
    llama_correct: int
    both_correct: int
    holyc_only_correct: int
    llama_only_correct: int
    both_wrong: int
    agreement_count: int
    holyc_accuracy: float
    llama_accuracy: float
    agreement: float
    holyc_accuracy_delta_vs_llama: float


@dataclass(frozen=True)
class Finding:
    severity: str
    bucket: str
    metric: str
    value: float | int
    limit: float | int
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def prompt_bytes(prompt: str) -> int:
    return len(prompt.encode("utf-8"))


def parse_bucket_edges(text: str) -> list[int]:
    edges: list[int] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            edge = int(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"bucket edge {stripped!r} is not an integer") from exc
        if edge <= 0:
            raise argparse.ArgumentTypeError("bucket edges must be positive")
        edges.append(edge)
    if sorted(set(edges)) != edges:
        raise argparse.ArgumentTypeError("bucket edges must be strictly increasing")
    return edges


def bucket_for_size(size: int, edges: list[int]) -> tuple[str, int, int]:
    lower = 0
    for edge in edges:
        if size <= edge:
            return f"{lower}-{edge}", lower, edge
        lower = edge + 1
    return f"{lower}+", lower, -1


def pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_bucket_rows(
    gold: dict[str, eval_compare.GoldCase],
    holyc: dict[str, eval_compare.Prediction],
    llama: dict[str, eval_compare.Prediction],
    edges: list[int],
) -> list[BucketRow]:
    buckets: dict[tuple[str, int, int], list[tuple[eval_compare.GoldCase, eval_compare.Prediction, eval_compare.Prediction]]] = {}
    for record_id in sorted(gold):
        case = gold[record_id]
        if record_id not in holyc or record_id not in llama:
            continue
        key = bucket_for_size(prompt_bytes(case.prompt), edges)
        buckets.setdefault(key, []).append((case, holyc[record_id], llama[record_id]))

    rows: list[BucketRow] = []
    for (bucket, min_bytes, max_bytes), records in sorted(buckets.items(), key=lambda item: item[0][1]):
        count = len(records)
        holyc_correct = 0
        llama_correct = 0
        both_correct = 0
        holyc_only = 0
        llama_only = 0
        both_wrong = 0
        agreement = 0
        for case, holyc_prediction, llama_prediction in records:
            h_ok = holyc_prediction.predicted_index == case.answer_index
            l_ok = llama_prediction.predicted_index == case.answer_index
            holyc_correct += int(h_ok)
            llama_correct += int(l_ok)
            both_correct += int(h_ok and l_ok)
            holyc_only += int(h_ok and not l_ok)
            llama_only += int(l_ok and not h_ok)
            both_wrong += int((not h_ok) and (not l_ok))
            agreement += int(holyc_prediction.predicted_index == llama_prediction.predicted_index)
        holyc_accuracy = pct(holyc_correct, count)
        llama_accuracy = pct(llama_correct, count)
        rows.append(
            BucketRow(
                bucket=bucket,
                min_prompt_bytes=min_bytes,
                max_prompt_bytes=max_bytes,
                record_count=count,
                holyc_correct=holyc_correct,
                llama_correct=llama_correct,
                both_correct=both_correct,
                holyc_only_correct=holyc_only,
                llama_only_correct=llama_only,
                both_wrong=both_wrong,
                agreement_count=agreement,
                holyc_accuracy=holyc_accuracy,
                llama_accuracy=llama_accuracy,
                agreement=pct(agreement, count),
                holyc_accuracy_delta_vs_llama=round(holyc_accuracy - llama_accuracy, 6),
            )
        )
    return rows


def audit_rows(
    rows: list[BucketRow],
    *,
    min_records_per_bucket: int,
    min_holyc_accuracy: float | None,
    max_holyc_accuracy_loss: float | None,
) -> list[Finding]:
    findings: list[Finding] = []
    for row in rows:
        if row.record_count < min_records_per_bucket:
            findings.append(
                Finding(
                    "error",
                    row.bucket,
                    "record_count",
                    row.record_count,
                    min_records_per_bucket,
                    f"bucket {row.bucket} has {row.record_count} paired records, below {min_records_per_bucket}",
                )
            )
        if min_holyc_accuracy is not None and row.holyc_accuracy < min_holyc_accuracy:
            findings.append(
                Finding(
                    "error",
                    row.bucket,
                    "holyc_accuracy",
                    row.holyc_accuracy,
                    min_holyc_accuracy,
                    f"bucket {row.bucket} HolyC accuracy {row.holyc_accuracy:.6f} is below {min_holyc_accuracy:.6f}",
                )
            )
        if max_holyc_accuracy_loss is not None:
            loss = row.llama_accuracy - row.holyc_accuracy
            if loss > max_holyc_accuracy_loss:
                findings.append(
                    Finding(
                        "error",
                        row.bucket,
                        "holyc_accuracy_loss",
                        round(loss, 6),
                        max_holyc_accuracy_loss,
                        f"bucket {row.bucket} HolyC trails llama.cpp by {loss:.6f}",
                    )
                )
    return findings


def build_report(rows: list[BucketRow], findings: list[Finding], args: argparse.Namespace) -> dict[str, Any]:
    total = sum(row.record_count for row in rows)
    holyc_correct = sum(row.holyc_correct for row in rows)
    llama_correct = sum(row.llama_correct for row in rows)
    agreement = sum(row.agreement_count for row in rows)
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "dataset": args.dataset,
        "split": args.split,
        "bucket_edges": args.bucket_edges,
        "inputs": {
            "gold": str(args.gold),
            "holyc_predictions": str(args.holyc),
            "llama_predictions": str(args.llama),
        },
        "summary": {
            "buckets": len(rows),
            "paired_records": total,
            "holyc_accuracy": pct(holyc_correct, total),
            "llama_accuracy": pct(llama_correct, total),
            "agreement": pct(agreement, total),
            "findings": len(findings),
        },
        "buckets": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[BucketRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BucketRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Eval Length Bucket Report",
        "",
        f"- Status: {report['status']}",
        f"- Paired records: {report['summary']['paired_records']}",
        f"- Buckets: {report['summary']['buckets']}",
        f"- HolyC accuracy: {report['summary']['holyc_accuracy']:.6f}",
        f"- llama.cpp accuracy: {report['summary']['llama_accuracy']:.6f}",
        "",
        "| Bucket | Records | HolyC Acc | llama.cpp Acc | Agreement | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["buckets"]:
        lines.append(
            f"| {row['bucket']} | {row['record_count']} | {row['holyc_accuracy']:.6f} | "
            f"{row['llama_accuracy']:.6f} | {row['agreement']:.6f} | "
            f"{row['holyc_accuracy_delta_vs_llama']:.6f} |"
        )
    if report["findings"]:
        lines.extend(["", "## Findings"])
        for finding in report["findings"]:
            lines.append(f"- {finding['severity']}: {finding['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_length_bucket_report",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "length_bucket_quality"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} length bucket finding(s)"})
        failure.text = "\n".join(finding.message for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Gold eval JSONL dataset")
    parser.add_argument("--holyc", type=Path, required=True, help="HolyC prediction JSON/JSONL/CSV")
    parser.add_argument("--llama", type=Path, required=True, help="llama.cpp prediction JSON/JSONL/CSV")
    parser.add_argument("--dataset", default="eval", help="Dataset name fallback for normalization")
    parser.add_argument("--split", default="validation", help="Split name fallback for normalization")
    parser.add_argument("--bucket-edges", type=parse_bucket_edges, default=[128, 256, 512, 1024])
    parser.add_argument("--min-records-per-bucket", type=int, default=1)
    parser.add_argument("--min-holyc-accuracy", type=float)
    parser.add_argument("--max-holyc-accuracy-loss", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_length_bucket_report_latest")
    args = parser.parse_args(argv)

    gold = eval_compare.load_gold(args.gold, args.dataset, args.split)
    holyc = eval_compare.load_predictions(args.holyc, gold)
    llama = eval_compare.load_predictions(args.llama, gold)
    rows = build_bucket_rows(gold, holyc, llama, args.bucket_edges)
    findings = audit_rows(
        rows,
        min_records_per_bucket=args.min_records_per_bucket,
        min_holyc_accuracy=args.min_holyc_accuracy,
        max_holyc_accuracy_loss=args.max_holyc_accuracy_loss,
    )
    report = build_report(rows, findings, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.output_stem}.json", report)
    write_csv(args.output_dir / f"{args.output_stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{args.output_stem}.md", report)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
