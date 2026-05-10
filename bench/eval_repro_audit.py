#!/usr/bin/env python3
"""Audit eval prediction reproducibility metadata across engines.

This host-side tool checks existing HolyC and llama.cpp prediction artifacts for
apples-to-apples decoding metadata. It does not launch QEMU, touch TempleOS
images, or require network access.
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

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_compare


REPRO_KEYS = ("seed", "temperature", "top_k", "top_p", "max_tokens")


@dataclass(frozen=True)
class ArtifactRepro:
    source: str
    rows: int
    metadata: dict[str, list[str]]
    missing_counts: dict[str, int]
    inconsistent_keys: list[str]


@dataclass(frozen=True)
class Finding:
    gate: str
    source: str
    key: str
    value: str
    expected: str
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def metadata_value(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get(key)
    if value is None or str(value).strip() == "":
        return None
    return normalize_value(value)


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.12g}"
        return str(value)
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number):
        return f"{number:.12g}"
    return text


def read_rows(path: Path) -> list[dict[str, Any]]:
    return eval_compare.read_prediction_rows(path)


def summarize_artifact(path: Path, keys: Iterable[str]) -> ArtifactRepro:
    rows = read_rows(path)
    values: dict[str, set[str]] = {key: set() for key in keys}
    missing_counts: dict[str, int] = {key: 0 for key in keys}
    for row in rows:
        for key in keys:
            value = metadata_value(row, key)
            if value is None:
                missing_counts[key] += 1
            else:
                values[key].add(value)
    metadata = {key: sorted(found) for key, found in values.items()}
    inconsistent = [key for key, found in metadata.items() if len(found) > 1]
    return ArtifactRepro(
        source=str(path),
        rows=len(rows),
        metadata=metadata,
        missing_counts=missing_counts,
        inconsistent_keys=inconsistent,
    )


def first_value(artifact: ArtifactRepro, key: str) -> str:
    values = artifact.metadata.get(key, [])
    return values[0] if len(values) == 1 else ""


def evaluate(artifacts: list[ArtifactRepro], args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    if len(artifacts) < args.min_artifacts:
        findings.append(
            Finding("min_artifacts", "", "", str(len(artifacts)), str(args.min_artifacts), "too few prediction artifacts")
        )
    for artifact in artifacts:
        if artifact.rows < args.min_rows:
            findings.append(Finding("min_rows", artifact.source, "", str(artifact.rows), str(args.min_rows), "too few prediction rows"))
        for key in artifact.inconsistent_keys:
            findings.append(
                Finding(
                    "inconsistent_metadata",
                    artifact.source,
                    key,
                    "|".join(artifact.metadata.get(key, [])),
                    "single value",
                    "artifact has multiple values for reproducibility metadata",
                )
            )
        for key, count in artifact.missing_counts.items():
            if args.require_metadata and count:
                findings.append(
                    Finding(
                        "missing_metadata",
                        artifact.source,
                        key,
                        str(count),
                        "0",
                        "rows are missing reproducibility metadata",
                    )
                )
    for key in args.require_key:
        for artifact in artifacts:
            value = first_value(artifact, key)
            if not value:
                findings.append(Finding("required_key", artifact.source, key, "", "present", "required metadata key is absent"))
    for key, expected in args.expect:
        expected_norm = normalize_value(expected)
        for artifact in artifacts:
            value = first_value(artifact, key)
            if value != expected_norm:
                findings.append(Finding("expected_value", artifact.source, key, value, expected_norm, "metadata value does not match expected value"))
    for key in args.compare_key:
        values = {first_value(artifact, key) for artifact in artifacts}
        if len(values) > 1:
            findings.append(Finding("cross_engine_mismatch", "all", key, "|".join(sorted(values)), "same", "metadata differs across artifacts"))
    if args.require_deterministic:
        for artifact in artifacts:
            seed = first_value(artifact, "seed")
            temperature = first_value(artifact, "temperature")
            if not seed:
                findings.append(Finding("deterministic_seed", artifact.source, "seed", "", "present", "deterministic eval requires a seed"))
            if temperature not in {"0", "0.0"}:
                findings.append(
                    Finding(
                        "deterministic_temperature",
                        artifact.source,
                        "temperature",
                        temperature,
                        "0",
                        "deterministic eval requires temperature 0",
                    )
                )
    return findings


def write_csv(path: Path, artifacts: list[ArtifactRepro]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["source", "rows", *REPRO_KEYS, *[f"missing_{key}" for key in REPRO_KEYS], "inconsistent_keys"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for artifact in artifacts:
            row: dict[str, Any] = {
                "source": artifact.source,
                "rows": artifact.rows,
                "inconsistent_keys": "|".join(artifact.inconsistent_keys),
            }
            for key in REPRO_KEYS:
                row[key] = "|".join(artifact.metadata.get(key, []))
                row[f"missing_{key}"] = artifact.missing_counts.get(key, 0)
            writer.writerow(row)


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate", "source", "key", "value", "expected", "message"])
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Reproducibility Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Artifacts: {payload['summary']['artifacts']}",
        f"- Rows: {payload['summary']['rows']}",
        f"- Findings: {payload['summary']['findings']}",
        "",
        "## Artifacts",
        "",
        "| Source | Rows | Seed | Temperature | top_k | top_p | max_tokens |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for artifact in payload["artifacts"]:
        metadata = artifact["metadata"]
        lines.append(
            "| {source} | {rows} | {seed} | {temperature} | {top_k} | {top_p} | {max_tokens} |".format(
                source=artifact["source"],
                rows=artifact["rows"],
                seed=", ".join(metadata.get("seed", [])),
                temperature=", ".join(metadata.get("temperature", [])),
                top_k=", ".join(metadata.get("top_k", [])),
                top_p=", ".join(metadata.get("top_p", [])),
                max_tokens=", ".join(metadata.get("max_tokens", [])),
            )
        )
    if payload["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in payload["findings"]:
            lines.append(f"- {finding['gate']} {finding['source']} {finding['key']}: {finding['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", name="holyc_eval_repro_audit", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="eval_repro_audit")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} reproducibility finding(s)")
        failure.text = "\n".join(f"{finding.gate}: {finding.source} {finding.key} {finding.message}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_expect(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
        key, expected = value.split("=", 1)
        if not key.strip():
            raise argparse.ArgumentTypeError(f"expected non-empty key in {value!r}")
        parsed.append((key.strip(), expected.strip()))
    return parsed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path, help="Prediction JSON/JSONL/CSV artifacts to audit.")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_repro_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=2)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--require-deterministic", action="store_true")
    parser.add_argument("--require-key", action="append", default=[])
    parser.add_argument("--compare-key", action="append", default=list(REPRO_KEYS))
    parser.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    args.expect = parse_expect(args.expect)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    artifacts = [summarize_artifact(path, REPRO_KEYS) for path in args.predictions]
    findings = evaluate(artifacts, args)
    status = "pass" if not findings else "fail"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "summary": {
            "artifacts": len(artifacts),
            "rows": sum(artifact.rows for artifact in artifacts),
            "findings": len(findings),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    (stem.with_suffix(".json")).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), artifacts)
    write_findings_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings)
    write_markdown(stem.with_suffix(".md"), payload)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", findings)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
