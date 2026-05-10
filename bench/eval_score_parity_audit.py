#!/usr/bin/env python3
"""Audit HolyC-vs-llama prediction score-vector parity before eval_compare.

This host-side tool reads a local gold JSONL dataset plus local HolyC and
llama.cpp prediction files. It verifies that both engines cover the same gold
record ids and, when requested, that every paired row carries a finite score
vector with matching choice-count shape. It never launches QEMU or touches the
TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
class ScoreRecord:
    source: str
    row_number: int
    record_id: str
    predicted_index: int | None
    has_scores: bool
    score_count: int
    choice_count: int
    top_score_index: int | None
    top_score_tie_count: int | None
    score_min: float | None
    score_max: float | None


@dataclass(frozen=True)
class PairRecord:
    record_id: str
    choice_count: int
    holyc_present: bool
    llama_present: bool
    holyc_has_scores: bool
    llama_has_scores: bool
    holyc_score_count: int
    llama_score_count: int
    holyc_predicted_index: int | None
    llama_predicted_index: int | None
    holyc_top_score_index: int | None
    llama_top_score_index: int | None
    holyc_top_score_tie_count: int | None
    llama_top_score_tie_count: int | None
    score_shape_match: bool
    both_scored: bool


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    record_id: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_finding(findings: list[Finding], source: str, record_id: str, kind: str, detail: str) -> None:
    findings.append(Finding("error", source, record_id, kind, detail))


def top_index(scores: list[float]) -> int:
    best_index = 0
    best_value = scores[0]
    for index, value in enumerate(scores[1:], 1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def load_engine_records(
    path: Path,
    source: str,
    gold: dict[str, eval_compare.GoldCase],
    findings: list[Finding],
) -> dict[str, ScoreRecord]:
    records: dict[str, ScoreRecord] = {}
    try:
        rows = eval_compare.read_prediction_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_finding(findings, source, "-", "load_error", f"cannot read predictions: {exc}")
        return records

    for index, row in enumerate(rows, 1):
        row_label = f"{path}:{index}"
        try:
            record_id = eval_compare.case_id(row, row_label)
        except ValueError as exc:
            append_finding(findings, source, "-", "missing_id", str(exc))
            continue
        if record_id in records:
            append_finding(findings, source, record_id, "duplicate_id", "duplicate prediction row")
            continue
        gold_case = gold.get(record_id)
        if gold_case is None:
            append_finding(findings, source, record_id, "extra_id", "prediction id is not present in gold")
            continue
        try:
            prediction = eval_compare.normalize_prediction(row, gold_case, path, index - 1)
        except ValueError as exc:
            append_finding(findings, source, record_id, "invalid_prediction", str(exc))
            continue

        scores = prediction.scores
        top_score_index = top_index(scores) if scores else None
        top_score_tie_count = None
        score_min = None
        score_max = None
        if scores:
            score_max = max(scores)
            score_min = min(scores)
            top_score_tie_count = sum(1 for score in scores if score == score_max)
        records[record_id] = ScoreRecord(
            source=source,
            row_number=index,
            record_id=record_id,
            predicted_index=prediction.predicted_index,
            has_scores=scores is not None,
            score_count=len(scores) if scores else 0,
            choice_count=len(gold_case.choices),
            top_score_index=top_score_index,
            top_score_tie_count=top_score_tie_count,
            score_min=score_min,
            score_max=score_max,
        )
    return records


def build_pairs(
    gold: dict[str, eval_compare.GoldCase],
    holyc: dict[str, ScoreRecord],
    llama: dict[str, ScoreRecord],
    findings: list[Finding],
    require_scores: bool,
) -> list[PairRecord]:
    pairs: list[PairRecord] = []
    for record_id, gold_case in sorted(gold.items()):
        holyc_record = holyc.get(record_id)
        llama_record = llama.get(record_id)
        if holyc_record is None:
            append_finding(findings, "holyc", record_id, "missing_id", "gold id is missing from HolyC predictions")
        if llama_record is None:
            append_finding(findings, "llama.cpp", record_id, "missing_id", "gold id is missing from llama.cpp predictions")

        holyc_has_scores = bool(holyc_record and holyc_record.has_scores)
        llama_has_scores = bool(llama_record and llama_record.has_scores)
        holyc_score_count = holyc_record.score_count if holyc_record else 0
        llama_score_count = llama_record.score_count if llama_record else 0
        both_scored = holyc_has_scores and llama_has_scores
        score_shape_match = both_scored and holyc_score_count == llama_score_count == len(gold_case.choices)

        if holyc_record and require_scores and not holyc_has_scores:
            append_finding(findings, "holyc", record_id, "missing_scores", "HolyC row has no score vector")
        if llama_record and require_scores and not llama_has_scores:
            append_finding(findings, "llama.cpp", record_id, "missing_scores", "llama.cpp row has no score vector")
        if holyc_record and llama_record and holyc_has_scores != llama_has_scores:
            append_finding(findings, "pair", record_id, "score_presence_mismatch", "only one engine has a score vector")
        if both_scored and not score_shape_match:
            append_finding(
                findings,
                "pair",
                record_id,
                "score_shape_mismatch",
                (
                    f"HolyC score_count={holyc_score_count}, llama.cpp score_count={llama_score_count}, "
                    f"gold choices={len(gold_case.choices)}"
                ),
            )

        pairs.append(
            PairRecord(
                record_id=record_id,
                choice_count=len(gold_case.choices),
                holyc_present=holyc_record is not None,
                llama_present=llama_record is not None,
                holyc_has_scores=holyc_has_scores,
                llama_has_scores=llama_has_scores,
                holyc_score_count=holyc_score_count,
                llama_score_count=llama_score_count,
                holyc_predicted_index=holyc_record.predicted_index if holyc_record else None,
                llama_predicted_index=llama_record.predicted_index if llama_record else None,
                holyc_top_score_index=holyc_record.top_score_index if holyc_record else None,
                llama_top_score_index=llama_record.top_score_index if llama_record else None,
                holyc_top_score_tie_count=holyc_record.top_score_tie_count if holyc_record else None,
                llama_top_score_tie_count=llama_record.top_score_tie_count if llama_record else None,
                score_shape_match=score_shape_match,
                both_scored=both_scored,
            )
        )
    return pairs


def load_gold(path: Path, dataset: str, split: str, findings: list[Finding]) -> dict[str, eval_compare.GoldCase]:
    try:
        return eval_compare.load_gold(path, dataset, split)
    except (OSError, ValueError) as exc:
        append_finding(findings, "gold", "-", "load_error", f"cannot load gold dataset: {exc}")
        return {}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    findings: list[Finding] = []
    gold = load_gold(args.gold, args.dataset, args.split, findings)
    holyc = load_engine_records(args.holyc, "holyc", gold, findings) if gold else {}
    llama = load_engine_records(args.llama, "llama.cpp", gold, findings) if gold else {}
    pairs = build_pairs(gold, holyc, llama, findings, args.require_scores)
    paired_rows = sum(1 for pair in pairs if pair.holyc_present and pair.llama_present)
    paired_scored_rows = sum(1 for pair in pairs if pair.both_scored)
    shape_match_rows = sum(1 for pair in pairs if pair.score_shape_match)
    holyc_scored_rows = sum(1 for pair in pairs if pair.holyc_has_scores)
    llama_scored_rows = sum(1 for pair in pairs if pair.llama_has_scores)
    holyc_top_score_tie_rows = sum(1 for pair in pairs if (pair.holyc_top_score_tie_count or 0) > 1)
    llama_top_score_tie_rows = sum(1 for pair in pairs if (pair.llama_top_score_tie_count or 0) > 1)
    score_parity_pct = (paired_scored_rows / paired_rows * 100.0) if paired_rows else 0.0
    shape_match_pct = (shape_match_rows / paired_scored_rows * 100.0) if paired_scored_rows else 0.0
    holyc_top_score_tie_pct = (holyc_top_score_tie_rows / holyc_scored_rows * 100.0) if holyc_scored_rows else 0.0
    llama_top_score_tie_pct = (llama_top_score_tie_rows / llama_scored_rows * 100.0) if llama_scored_rows else 0.0

    if args.min_paired_rows is not None and paired_rows < args.min_paired_rows:
        append_finding(
            findings,
            "pair",
            "-",
            "min_paired_rows",
            f"paired rows {paired_rows} below required {args.min_paired_rows}",
        )
    if args.min_score_parity_pct is not None and score_parity_pct < args.min_score_parity_pct:
        append_finding(
            findings,
            "pair",
            "-",
            "min_score_parity_pct",
            f"paired scored rows {score_parity_pct:.2f}% below required {args.min_score_parity_pct:.2f}%",
        )
    if args.max_top_score_tie_pct is not None:
        if holyc_top_score_tie_pct > args.max_top_score_tie_pct:
            append_finding(
                findings,
                "holyc",
                "-",
                "max_top_score_tie_pct",
                f"HolyC top-score tie rows {holyc_top_score_tie_pct:.2f}% above allowed {args.max_top_score_tie_pct:.2f}%",
            )
        if llama_top_score_tie_pct > args.max_top_score_tie_pct:
            append_finding(
                findings,
                "llama.cpp",
                "-",
                "max_top_score_tie_pct",
                f"llama.cpp top-score tie rows {llama_top_score_tie_pct:.2f}% above allowed {args.max_top_score_tie_pct:.2f}%",
            )

    status = "fail" if findings else "pass"
    return {
        "dataset": args.dataset,
        "files": {
            "gold": {"path": str(args.gold), "sha256": file_sha256(args.gold) if args.gold.exists() else ""},
            "holyc": {"path": str(args.holyc), "sha256": file_sha256(args.holyc) if args.holyc.exists() else ""},
            "llama": {"path": str(args.llama), "sha256": file_sha256(args.llama) if args.llama.exists() else ""},
        },
        "findings": [asdict(finding) for finding in findings],
        "generated_at": iso_now(),
        "gold_record_count": len(gold),
        "pairs": [asdict(pair) for pair in pairs],
        "split": args.split,
        "status": status,
        "summary": {
            "findings": len(findings),
            "holyc_rows": len(holyc),
            "holyc_scored_rows": holyc_scored_rows,
            "holyc_top_score_tie_pct": holyc_top_score_tie_pct,
            "holyc_top_score_tie_rows": holyc_top_score_tie_rows,
            "llama_rows": len(llama),
            "llama_scored_rows": llama_scored_rows,
            "llama_top_score_tie_pct": llama_top_score_tie_pct,
            "llama_top_score_tie_rows": llama_top_score_tie_rows,
            "paired_rows": paired_rows,
            "paired_scored_rows": paired_scored_rows,
            "score_parity_pct": score_parity_pct,
            "score_shape_match_rows": shape_match_rows,
            "score_shape_match_pct": shape_match_pct,
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Eval Score Parity Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Dataset: {report['dataset']}",
        f"Split: {report['split']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Gold records | {report['gold_record_count']} |",
        f"| HolyC rows | {summary['holyc_rows']} |",
        f"| llama.cpp rows | {summary['llama_rows']} |",
        f"| Paired rows | {summary['paired_rows']} |",
        f"| Paired scored rows | {summary['paired_scored_rows']} |",
        f"| Score parity % | {summary['score_parity_pct']:.2f} |",
        f"| Score shape match % | {summary['score_shape_match_pct']:.2f} |",
        f"| HolyC top-score tie rows | {summary['holyc_top_score_tie_rows']} |",
        f"| HolyC top-score tie % | {summary['holyc_top_score_tie_pct']:.2f} |",
        f"| llama.cpp top-score tie rows | {summary['llama_top_score_tie_rows']} |",
        f"| llama.cpp top-score tie % | {summary['llama_top_score_tie_pct']:.2f} |",
        f"| Findings | {summary['findings']} |",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.append("| Severity | Source | Record | Kind | Detail |")
        lines.append("| --- | --- | --- | --- | --- |")
        for finding in report["findings"]:
            lines.append(
                f"| {finding['severity']} | {finding['source']} | {finding['record_id']} | "
                f"{finding['kind']} | {finding['detail']} |"
            )
    else:
        lines.append("No score parity findings.")
    return "\n".join(lines) + "\n"


def write_findings_csv(report: dict[str, Any], path: Path) -> None:
    fields = ["severity", "source", "record_id", "kind", "detail"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow({field: finding[field] for field in fields})


def write_pairs_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "record_id",
        "choice_count",
        "holyc_present",
        "llama_present",
        "holyc_has_scores",
        "llama_has_scores",
        "holyc_score_count",
        "llama_score_count",
        "holyc_predicted_index",
        "llama_predicted_index",
        "holyc_top_score_index",
        "llama_top_score_index",
        "holyc_top_score_tie_count",
        "llama_top_score_tie_count",
        "score_shape_match",
        "both_scored",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for pair in report["pairs"]:
            writer.writerow({field: pair.get(field, "") for field in fields})


def write_junit(report: dict[str, Any], path: Path) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_score_parity_audit",
            "tests": "1",
            "failures": "1" if report["findings"] else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"classname": "eval_score_parity_audit", "name": "score_parity"})
    if report["findings"]:
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "eval_score_parity_audit_error",
                "message": f"{len(report['findings'])} score parity finding(s)",
            },
        )
        failure.text = "\n".join(
            f"{finding['source']} {finding['record_id']}: {finding['detail']}"
            for finding in report["findings"]
        )
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def write_report(report: dict[str, Any], output_dir: Path, output_stem: str) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_stem}.json"
    md_path = output_dir / f"{output_stem}.md"
    findings_csv = output_dir / f"{output_stem}.csv"
    pairs_csv = output_dir / f"{output_stem}_pairs.csv"
    junit_path = output_dir / f"{output_stem}_junit.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_findings_csv(report, findings_csv)
    write_pairs_csv(report, pairs_csv)
    write_junit(report, junit_path)
    return json_path, md_path, findings_csv, pairs_csv, junit_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Local gold JSONL dataset")
    parser.add_argument("--holyc", type=Path, required=True, help="HolyC prediction JSON/JSONL/CSV")
    parser.add_argument("--llama", type=Path, required=True, help="llama.cpp prediction JSON/JSONL/CSV")
    parser.add_argument("--dataset", default="eval")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--require-scores", action="store_true", help="Fail when either engine lacks score vectors")
    parser.add_argument("--min-paired-rows", type=int, help="Fail when fewer gold rows are present in both engines")
    parser.add_argument(
        "--min-score-parity-pct",
        type=float,
        help="Fail when fewer paired rows have score vectors in both engines",
    )
    parser.add_argument(
        "--max-top-score-tie-pct",
        type=float,
        help="Fail when either engine has more than this percentage of scored rows with tied top scores",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_score_parity_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_paired_rows is not None and args.min_paired_rows < 0:
        print("error: --min-paired-rows must be non-negative", file=sys.stderr)
        return 2
    if args.min_score_parity_pct is not None and not 0.0 <= args.min_score_parity_pct <= 100.0:
        print("error: --min-score-parity-pct must be between 0 and 100", file=sys.stderr)
        return 2
    if args.max_top_score_tie_pct is not None and not 0.0 <= args.max_top_score_tie_pct <= 100.0:
        print("error: --max-top-score-tie-pct must be between 0 and 100", file=sys.stderr)
        return 2
    report = build_report(args)
    json_path, md_path, findings_csv, pairs_csv, junit_path = write_report(report, args.output_dir, args.output_stem)
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_csv={findings_csv}")
    print(f"wrote_pairs_csv={pairs_csv}")
    print(f"wrote_junit={junit_path}")
    print(f"status={report['status']}")
    print(f"findings={report['summary']['findings']}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
