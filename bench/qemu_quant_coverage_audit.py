#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for quantization coverage.

This host-side tool reads existing benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = (
    "qemu_prompt_bench_latest.json",
    "qemu_prompt_bench_????????T??????Z.json",
    "qemu_prompt_bench_latest.jsonl",
    "qemu_prompt_bench_????????T??????Z.jsonl",
    "qemu_prompt_bench_latest.csv",
    "qemu_prompt_bench_????????T??????Z.csv",
)
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class QuantCoverageRow:
    source: str
    profile: str
    model: str
    quantization: str
    rows: int
    ok_rows: int
    airgap_ok_rows: int
    explicit_nic_none_rows: int
    prompts: int


@dataclass(frozen=True)
class Finding:
    source: str
    group: str
    severity: str
    kind: str
    metric: str
    value: int | str
    threshold: int | str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def row_text(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


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
                if stripped:
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


def row_ok(row: dict[str, Any]) -> bool:
    if "success" in row:
        parsed = parse_bool(row.get("success"))
        if parsed is not None:
            return parsed
    if "exit_class" in row:
        return str(row.get("exit_class") or "").strip().lower() == "ok"
    return str(row.get("returncode") or "0") == "0" and parse_bool(row.get("timed_out")) is not True


def row_airgap_ok(row: dict[str, Any]) -> bool:
    parsed = parse_bool(row.get("command_airgap_ok"))
    if parsed is not None:
        return parsed
    metadata = row.get("command_airgap")
    if isinstance(metadata, dict):
        parsed = parse_bool(metadata.get("ok"))
        if parsed is not None:
            return parsed
    return False


def row_explicit_nic_none(row: dict[str, Any]) -> bool:
    parsed = parse_bool(row.get("command_has_explicit_nic_none"))
    if parsed is not None:
        return parsed
    metadata = row.get("command_airgap")
    if isinstance(metadata, dict):
        parsed = parse_bool(metadata.get("explicit_nic_none"))
        if parsed is not None:
            return parsed
    return False


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[QuantCoverageRow], list[Finding]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    load_findings: list[Finding] = []
    for path in iter_input_files(paths, args.pattern):
        try:
            for row in load_rows(path):
                if row_text(row, "phase", default="measured") == "warmup" and not args.include_warmup:
                    continue
                profile = row_text(row, "profile", default="")
                model = row_text(row, "model", default="")
                quantization = row_text(row, "quantization", "quant", default="")
                if not quantization:
                    continue
                if not (profile or model or quantization or row.get("prompt") or row.get("prompt_id")):
                    continue
                key = (str(path), profile or "-", model or "-", quantization or "-")
                grouped.setdefault(key, []).append(row)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            load_findings.append(Finding(str(path), "artifact", "error", "load_error", "artifact", str(exc), "valid input", "could not load benchmark artifact"))

    rows: list[QuantCoverageRow] = []
    by_model: dict[tuple[str, str], set[str]] = {}
    for (source, profile, model, quantization), items in sorted(grouped.items()):
        prompts = {row_text(item, "prompt", "prompt_id", default=f"row-{index}") for index, item in enumerate(items)}
        rows.append(
            QuantCoverageRow(
                source=source,
                profile=profile,
                model=model,
                quantization=quantization,
                rows=len(items),
                ok_rows=sum(1 for item in items if row_ok(item)),
                airgap_ok_rows=sum(1 for item in items if row_airgap_ok(item)),
                explicit_nic_none_rows=sum(1 for item in items if row_explicit_nic_none(item)),
                prompts=len(prompts),
            )
        )
        by_model.setdefault((profile, model), set()).add(quantization)

    findings = list(load_findings)
    if len(rows) < args.min_groups:
        findings.append(Finding("", "coverage", "error", "min_groups", "groups", len(rows), args.min_groups, "too few quantization coverage groups found"))

    required = set(args.require_quant)
    for row in rows:
        label = f"{row.profile}/{row.model}/{row.quantization}"
        if row.rows < args.min_rows_per_quant:
            findings.append(Finding(row.source, label, "error", "min_rows_per_quant", "rows", row.rows, args.min_rows_per_quant, "quantization has too few benchmark rows"))
        if row.ok_rows < args.min_ok_rows_per_quant:
            findings.append(Finding(row.source, label, "error", "min_ok_rows_per_quant", "ok_rows", row.ok_rows, args.min_ok_rows_per_quant, "quantization has too few successful benchmark rows"))
        if row.prompts < args.min_prompts_per_quant:
            findings.append(Finding(row.source, label, "error", "min_prompts_per_quant", "prompts", row.prompts, args.min_prompts_per_quant, "quantization has too few unique prompts"))
        if args.require_airgap_command and row.airgap_ok_rows < row.rows:
            findings.append(
                Finding(
                    row.source,
                    label,
                    "error",
                    "missing_airgap_command",
                    "airgap_ok_rows",
                    row.airgap_ok_rows,
                    row.rows,
                    "every quantization benchmark row must carry passing air-gap command telemetry",
                )
            )
        if args.require_airgap_command and row.explicit_nic_none_rows < row.rows:
            findings.append(
                Finding(
                    row.source,
                    label,
                    "error",
                    "missing_explicit_nic_none",
                    "explicit_nic_none_rows",
                    row.explicit_nic_none_rows,
                    row.rows,
                    "every quantization benchmark row must prove an explicit -nic none launch",
                )
            )

    for (profile, model), present in sorted(by_model.items()):
        missing = sorted(required - present)
        if missing:
            findings.append(
                Finding(
                    "",
                    f"{profile}/{model}",
                    "error",
                    "missing_required_quantization",
                    "quantization",
                    ",".join(sorted(present)) or "-",
                    ",".join(sorted(required)),
                    f"missing required quantization(s): {', '.join(missing)}",
                )
            )
    return rows, findings


def build_report(rows: list[QuantCoverageRow], findings: list[Finding], required: list[str]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "required_quantizations": required,
        "summary": {
            "groups": len(rows),
            "rows": sum(row.rows for row in rows),
            "ok_rows": sum(row.ok_rows for row in rows),
            "airgap_ok_rows": sum(row.airgap_ok_rows for row in rows),
            "explicit_nic_none_rows": sum(row.explicit_nic_none_rows for row in rows),
            "findings": len(findings),
        },
        "groups": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[QuantCoverageRow], findings: list[Finding]) -> None:
    lines = ["# QEMU Quantization Coverage Audit", "", f"Groups: {len(rows)}", f"Findings: {len(findings)}", ""]
    if findings:
        lines.extend(["| Group | Kind | Metric | Value | Threshold | Detail |", "| --- | --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.group} | {finding.kind} | {finding.metric} | {finding.value} | {finding.threshold} | {finding.detail.replace('|', '\\|')} |")
    else:
        lines.append("No quantization coverage findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[QuantCoverageRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QuantCoverageRow.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", {"name": "holyc_qemu_quant_coverage_audit", "tests": str(max(1, len(findings) or 1)), "failures": str(len(findings)), "errors": "0"})
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "qemu_quant_coverage_audit", "name": "quant_coverage"})
    for index, finding in enumerate(findings, 1):
        case = ET.SubElement(suite, "testcase", {"classname": "qemu_quant_coverage_audit", "name": f"{finding.kind}_{index}"})
        failure = ET.SubElement(case, "failure", {"type": finding.kind, "message": finding.detail})
        failure.text = f"{finding.group} {finding.metric}={finding.value} threshold={finding.threshold}"
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="benchmark artifacts or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob for directory inputs")
    parser.add_argument("--include-warmup", action="store_true", help="include warmup rows in the audit")
    parser.add_argument("--require-quant", action="append", default=["Q4_0", "Q8_0"], help="required quantization label; repeatable")
    parser.add_argument("--min-groups", type=int, default=1, help="minimum quantization groups required")
    parser.add_argument("--min-rows-per-quant", type=int, default=1, help="minimum rows per quantization")
    parser.add_argument("--min-ok-rows-per-quant", type=int, default=1, help="minimum successful rows per quantization")
    parser.add_argument("--min-prompts-per-quant", type=int, default=1, help="minimum unique prompts per quantization")
    parser.add_argument("--require-airgap-command", action="store_true", help="require every row to carry passing -nic none command telemetry")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="output directory")
    parser.add_argument("--output-stem", default="qemu_quant_coverage_audit_latest", help="output filename stem")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)
    report = build_report(rows, findings, args.require_quant)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", rows, findings)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
