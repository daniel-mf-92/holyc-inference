#!/usr/bin/env python3
"""Audit QEMU prompt benchmark artifacts for prompt efficiency telemetry.

This host-side tool reads saved benchmark artifacts only. It never launches
QEMU and never touches the TempleOS guest.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class EfficiencyRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    exit_class: str
    tokens: int | None
    prompt_bytes: int | None
    elapsed_us: float | None
    wall_elapsed_us: float | None
    tokens_per_prompt_byte: float | None
    prompt_bytes_per_s: float | None
    wall_prompt_bytes_per_s: float | None


@dataclass(frozen=True)
class EfficiencyFinding:
    source: str
    row: int
    severity: str
    kind: str
    metric: str
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


def close_enough(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(0.01, abs(expected) * 0.001)


def measured_ok(row: dict[str, Any]) -> bool:
    phase = str(row.get("phase") or "measured")
    exit_class = str(row.get("exit_class") or "")
    timed_out = bool(row.get("timed_out"))
    return phase == "measured" and exit_class == "ok" and not timed_out


def efficiency_row(source: Path, row_number: int, raw: dict[str, Any], args: argparse.Namespace) -> tuple[EfficiencyRow, list[EfficiencyFinding]]:
    findings: list[EfficiencyFinding] = []
    prompt_bytes = finite_int(raw.get("prompt_bytes"))
    tokens = finite_int(raw.get("tokens"))
    elapsed_us = finite_float(raw.get("elapsed_us"))
    wall_elapsed_us = finite_float(raw.get("wall_elapsed_us"))
    tokens_per_prompt_byte = finite_float(raw.get("tokens_per_prompt_byte"))
    prompt_bytes_per_s = finite_float(raw.get("prompt_bytes_per_s"))
    wall_prompt_bytes_per_s = finite_float(raw.get("wall_prompt_bytes_per_s"))

    if prompt_bytes is None:
        findings.append(EfficiencyFinding(str(source), row_number, "error", "missing_prompt_bytes", "prompt_bytes", "prompt byte count is required"))
    elif prompt_bytes <= 0:
        findings.append(EfficiencyFinding(str(source), row_number, "error", "nonpositive_prompt_bytes", "prompt_bytes", "prompt byte count must be positive"))

    if tokens is not None and prompt_bytes is not None and prompt_bytes > 0:
        expected = tokens / prompt_bytes
        if tokens_per_prompt_byte is None:
            findings.append(EfficiencyFinding(str(source), row_number, "error", "missing_tokens_per_prompt_byte", "tokens_per_prompt_byte", "derived token density is required"))
        elif not close_enough(tokens_per_prompt_byte, expected):
            findings.append(
                EfficiencyFinding(
                    str(source),
                    row_number,
                    "error",
                    "tokens_per_prompt_byte_drift",
                    "tokens_per_prompt_byte",
                    f"expected about {expected:.6f} from tokens/prompt_bytes",
                )
            )
        elif tokens_per_prompt_byte < args.min_tokens_per_prompt_byte:
            findings.append(
                EfficiencyFinding(
                    str(source),
                    row_number,
                    "error",
                    "min_tokens_per_prompt_byte",
                    "tokens_per_prompt_byte",
                    f"{tokens_per_prompt_byte:.6f} below minimum {args.min_tokens_per_prompt_byte:.6f}",
                )
            )

    if prompt_bytes is not None and prompt_bytes > 0 and elapsed_us is not None and elapsed_us > 0:
        expected = prompt_bytes * 1_000_000.0 / elapsed_us
        if prompt_bytes_per_s is None:
            findings.append(EfficiencyFinding(str(source), row_number, "error", "missing_prompt_bytes_per_s", "prompt_bytes_per_s", "guest prompt byte rate is required"))
        elif not close_enough(prompt_bytes_per_s, expected):
            findings.append(
                EfficiencyFinding(
                    str(source),
                    row_number,
                    "error",
                    "prompt_bytes_per_s_drift",
                    "prompt_bytes_per_s",
                    f"expected about {expected:.6f} from prompt_bytes/elapsed_us",
                )
            )

    if prompt_bytes is not None and prompt_bytes > 0 and wall_elapsed_us is not None and wall_elapsed_us > 0:
        expected = prompt_bytes * 1_000_000.0 / wall_elapsed_us
        if wall_prompt_bytes_per_s is None:
            findings.append(EfficiencyFinding(str(source), row_number, "error", "missing_wall_prompt_bytes_per_s", "wall_prompt_bytes_per_s", "wall prompt byte rate is required"))
        elif not close_enough(wall_prompt_bytes_per_s, expected):
            findings.append(
                EfficiencyFinding(
                    str(source),
                    row_number,
                    "error",
                    "wall_prompt_bytes_per_s_drift",
                    "wall_prompt_bytes_per_s",
                    f"expected about {expected:.6f} from prompt_bytes/wall_elapsed_us",
                )
            )
        elif wall_prompt_bytes_per_s < args.min_wall_prompt_bytes_per_s:
            findings.append(
                EfficiencyFinding(
                    str(source),
                    row_number,
                    "error",
                    "min_wall_prompt_bytes_per_s",
                    "wall_prompt_bytes_per_s",
                    f"{wall_prompt_bytes_per_s:.6f} below minimum {args.min_wall_prompt_bytes_per_s:.6f}",
                )
            )

    return (
        EfficiencyRow(
            str(source),
            row_number,
            row_text(raw, "profile"),
            row_text(raw, "model"),
            row_text(raw, "quantization"),
            row_text(raw, "prompt", "prompt_id"),
            row_text(raw, "phase", default="measured"),
            row_text(raw, "exit_class"),
            tokens,
            prompt_bytes,
            elapsed_us,
            wall_elapsed_us,
            tokens_per_prompt_byte,
            prompt_bytes_per_s,
            wall_prompt_bytes_per_s,
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[EfficiencyRow], list[EfficiencyFinding]]:
    rows: list[EfficiencyRow] = []
    findings: list[EfficiencyFinding] = []
    files = list(iter_input_files(paths, args.pattern))
    if len(files) < args.min_artifacts:
        findings.append(EfficiencyFinding("", 0, "error", "min_artifacts", "artifacts", f"found {len(files)} artifacts, need {args.min_artifacts}"))
    for path in files:
        try:
            raw_rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(EfficiencyFinding(str(path), 0, "error", "load_error", "artifact", str(exc)))
            continue
        for row_number, raw in enumerate(raw_rows, 1):
            if args.measured_ok_only and not measured_ok(raw):
                continue
            row, row_findings = efficiency_row(path, row_number, raw, args)
            rows.append(row)
            findings.extend(row_findings)
    if len(rows) < args.min_rows:
        findings.append(EfficiencyFinding("", 0, "error", "min_rows", "rows", f"found {len(rows)} measured rows, need {args.min_rows}"))
    return rows, findings


def write_csv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# QEMU Prompt Efficiency Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Rows: {payload['summary']['rows']}",
        f"- Findings: {payload['summary']['findings']}",
        "",
    ]
    if payload["findings"]:
        lines.append("## Findings")
        lines.append("")
        for finding in payload["findings"]:
            lines.append(f"- {finding['severity']} {finding['kind']} {finding['source']}:{finding['row']} {finding['detail']}")
    else:
        lines.append("No prompt efficiency findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[EfficiencyFinding]) -> None:
    root = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_efficiency_audit",
            "tests": "1",
            "failures": str(len(findings)),
        },
    )
    case = ET.SubElement(root, "testcase", {"name": "prompt_efficiency"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} prompt efficiency findings"})
        failure.text = "\n".join(f"{item.kind}: {item.source}:{item.row}: {item.detail}" for item in findings)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob for directory inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_prompt_efficiency_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-tokens-per-prompt-byte", type=float, default=0.0)
    parser.add_argument("--min-wall-prompt-bytes-per-s", type=float, default=0.0)
    parser.add_argument("--include-warmups-and-failures", action="store_false", dest="measured_ok_only")
    parser.set_defaults(measured_ok_only=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "rows": len(rows),
            "findings": len(findings),
            "min_tokens_per_prompt_byte": args.min_tokens_per_prompt_byte,
            "min_wall_prompt_bytes_per_s": args.min_wall_prompt_bytes_per_s,
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / f"{stem}.md", payload)
    write_csv(args.output_dir / f"{stem}.csv", rows, list(EfficiencyRow.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", findings, list(EfficiencyFinding.__dataclass_fields__))
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
