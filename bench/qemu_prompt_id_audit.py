#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for prompt identity drift.

This host-side tool reads existing benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
RESULT_KEYS = ("warmups", "benchmarks", "results", "runs", "rows")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PromptIdentity:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    phase: str
    prompt: str
    prompt_sha256: str
    launch_index: str
    iteration: str


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    prompt: str
    prompt_sha256: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def text_value(value: Any, default: str = "") -> str:
    if value in (None, ""):
        return default
    return str(value)


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

    inherited = {key: value for key, value in payload.items() if key not in RESULT_KEYS}
    yielded = False
    for key in RESULT_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list):
            yielded = True
            for item in rows:
                if isinstance(item, dict):
                    merged = dict(inherited)
                    merged.update(item)
                    if key == "warmups" and "phase" not in merged:
                        merged["phase"] = "warmup"
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


def prompt_identity(source: Path, row_number: int, row: dict[str, Any]) -> PromptIdentity:
    return PromptIdentity(
        source=str(source),
        row=row_number,
        profile=text_value(row.get("profile")),
        model=text_value(row.get("model")),
        quantization=text_value(row.get("quantization")),
        phase=text_value(row.get("phase"), "measured"),
        prompt=text_value(row.get("prompt") or row.get("prompt_id")),
        prompt_sha256=text_value(row.get("prompt_sha256") or row.get("guest_prompt_sha256")).lower(),
        launch_index=text_value(row.get("launch_index")),
        iteration=text_value(row.get("iteration")),
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[PromptIdentity], list[Finding]]:
    rows: list[PromptIdentity] = []
    findings: list[Finding] = []
    files = 0

    for path in iter_input_files(paths, args.pattern):
        files += 1
        try:
            loaded_rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "", "", str(exc)))
            continue

        prompt_to_sha: dict[str, str] = {}
        sha_to_prompt: dict[str, str] = {}
        for row_number, raw in enumerate(loaded_rows, 1):
            row = prompt_identity(path, row_number, raw)
            rows.append(row)
            if not row.prompt:
                findings.append(Finding(row.source, row.row, "error", "missing_prompt", row.prompt, row.prompt_sha256, "prompt/prompt_id is absent"))
            if not row.prompt_sha256:
                findings.append(Finding(row.source, row.row, "error", "missing_prompt_sha256", row.prompt, row.prompt_sha256, "prompt_sha256 is absent"))
            elif args.require_sha256_format and not SHA256_RE.fullmatch(row.prompt_sha256):
                findings.append(
                    Finding(row.source, row.row, "error", "invalid_prompt_sha256", row.prompt, row.prompt_sha256, "prompt_sha256 must be 64 lowercase hex characters")
                )

            previous_sha = prompt_to_sha.setdefault(row.prompt, row.prompt_sha256)
            if row.prompt and row.prompt_sha256 and previous_sha != row.prompt_sha256:
                findings.append(
                    Finding(
                        row.source,
                        row.row,
                        "error",
                        "prompt_hash_drift",
                        row.prompt,
                        row.prompt_sha256,
                        f"prompt was previously observed with sha {previous_sha}",
                    )
                )

            previous_prompt = sha_to_prompt.setdefault(row.prompt_sha256, row.prompt)
            if row.prompt and row.prompt_sha256 and previous_prompt != row.prompt:
                findings.append(
                    Finding(
                        row.source,
                        row.row,
                        "error",
                        "prompt_hash_collision",
                        row.prompt,
                        row.prompt_sha256,
                        f"sha was previously observed for prompt {previous_prompt!r}",
                    )
                )

    if files == 0:
        findings.append(Finding("", 0, "error", "no_input_files", "", "", "no benchmark artifacts matched input paths/patterns"))
    if len(rows) < args.min_rows:
        findings.append(Finding("", 0, "error", "min_rows", "", "", f"found {len(rows)} row(s), below minimum {args.min_rows}"))
    return rows, findings


def build_report(rows: list[PromptIdentity], findings: list[Finding]) -> dict[str, Any]:
    unique_prompts = {row.prompt for row in rows if row.prompt}
    unique_hashes = {row.prompt_sha256 for row in rows if row.prompt_sha256}
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "rows": len(rows),
            "unique_prompts": len(unique_prompts),
            "unique_prompt_sha256": len(unique_hashes),
            "missing_prompts": sum(1 for finding in findings if finding.kind == "missing_prompt"),
            "missing_prompt_sha256": sum(1 for finding in findings if finding.kind == "missing_prompt_sha256"),
            "prompt_hash_drifts": sum(1 for finding in findings if finding.kind == "prompt_hash_drift"),
            "prompt_hash_collisions": sum(1 for finding in findings if finding.kind == "prompt_hash_collision"),
            "findings": len(findings),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[PromptIdentity]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PromptIdentity.__dataclass_fields__))
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
    summary = report["summary"]
    lines = [
        "# QEMU Prompt Identity Audit",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- status: {report['status']}",
        f"- rows: {summary['rows']}",
        f"- unique_prompts: {summary['unique_prompts']}",
        f"- unique_prompt_sha256: {summary['unique_prompt_sha256']}",
        f"- prompt_hash_drifts: {summary['prompt_hash_drifts']}",
        f"- prompt_hash_collisions: {summary['prompt_hash_collisions']}",
        f"- findings: {summary['findings']}",
        "",
    ]
    if report["findings"]:
        lines.extend(["## Findings", "", "| source | row | kind | prompt | detail |", "| --- | ---: | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['source']} | {finding['row']} | {finding['kind']} | {finding['prompt']} | {finding['detail']} |")
    else:
        lines.append("No prompt identity findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_id_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    testcase = ET.SubElement(testsuite, "testcase", {"name": "prompt_identity"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"message": f"{len(findings)} finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern used when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Directory for audit outputs")
    parser.add_argument("--output-stem", default="qemu_prompt_id_audit_latest", help="Output filename stem")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum benchmark rows required")
    parser.add_argument("--require-sha256-format", action=argparse.BooleanOptionalAction, default=True, help="Require prompt_sha256 to be 64 lowercase hex characters")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)
    report = build_report(rows, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
