#!/usr/bin/env python3
"""Audit eval prediction artifacts for apples-to-apples model identity.

This host-side tool reads existing HolyC and llama.cpp prediction artifacts
only. It does not launch QEMU, touch TempleOS images, or use networking.
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
from typing import Any, Iterable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_compare


DEFAULT_IDENTITY_KEYS = (
    "model",
    "model_sha256",
    "tokenizer_sha256",
    "quantization",
    "prompt_template_sha256",
)


@dataclass(frozen=True)
class ArtifactIdentity:
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


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def metadata_value(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get(key)
    if value is None or str(value).strip() == "":
        return None
    return normalize_value(value)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return eval_compare.read_prediction_rows(path)


def summarize_artifact(path: Path, keys: Iterable[str]) -> ArtifactIdentity:
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
    return ArtifactIdentity(
        source=str(path),
        rows=len(rows),
        metadata=metadata,
        missing_counts=missing_counts,
        inconsistent_keys=inconsistent,
    )


def first_value(artifact: ArtifactIdentity, key: str) -> str:
    values = artifact.metadata.get(key, [])
    return values[0] if len(values) == 1 else ""


def evaluate(artifacts: list[ArtifactIdentity], args: argparse.Namespace) -> list[Finding]:
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
                    "inconsistent_identity",
                    artifact.source,
                    key,
                    "|".join(artifact.metadata.get(key, [])),
                    "single value",
                    "artifact has multiple identity values",
                )
            )
        for key, count in artifact.missing_counts.items():
            if args.require_identity and count:
                findings.append(
                    Finding(
                        "missing_identity",
                        artifact.source,
                        key,
                        str(count),
                        "0",
                        "rows are missing model identity metadata",
                    )
                )

    for key in args.require_key:
        for artifact in artifacts:
            value = first_value(artifact, key)
            if not value:
                findings.append(Finding("required_key", artifact.source, key, "", "present", "required identity key is absent"))

    for key, expected in args.expect:
        expected_norm = normalize_value(expected)
        for artifact in artifacts:
            value = first_value(artifact, key)
            if value != expected_norm:
                findings.append(Finding("expected_value", artifact.source, key, value, expected_norm, "identity value does not match expected value"))

    for key in args.compare_key:
        values = {first_value(artifact, key) for artifact in artifacts}
        if len(values) > 1:
            findings.append(Finding("cross_engine_mismatch", "all", key, "|".join(sorted(values)), "same", "identity differs across artifacts"))
    return findings


def write_csv(path: Path, artifacts: list[ArtifactIdentity], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["source", "rows", *keys, *[f"missing_{key}" for key in keys], "inconsistent_keys"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for artifact in artifacts:
            row: dict[str, Any] = {
                "source": artifact.source,
                "rows": artifact.rows,
                "inconsistent_keys": "|".join(artifact.inconsistent_keys),
            }
            for key in keys:
                row[key] = "|".join(artifact.metadata.get(key, []))
                row[f"missing_{key}"] = artifact.missing_counts.get(key, 0)
            writer.writerow(row)


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate", "source", "key", "value", "expected", "message"])
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, payload: dict[str, Any], keys: list[str]) -> None:
    lines = [
        "# Eval Identity Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Artifacts: {payload['summary']['artifacts']}",
        f"- Rows: {payload['summary']['rows']}",
        f"- Findings: {payload['summary']['findings']}",
        "",
        "## Artifacts",
        "",
        "| Source | Rows | " + " | ".join(keys) + " |",
        "| --- | ---: | " + " | ".join("---" for _ in keys) + " |",
    ]
    for artifact in payload["artifacts"]:
        values = [artifact["source"], str(artifact["rows"])]
        for key in keys:
            values.append(", ".join(artifact["metadata"].get(key, [])) or "-")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Findings", ""])
    if payload["findings"]:
        lines.append("| Gate | Source | Key | Value | Expected | Message |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for finding in payload["findings"]:
            lines.append(
                f"| {finding['gate']} | {finding['source'] or '-'} | {finding['key'] or '-'} | "
                f"{finding['value'] or '-'} | {finding['expected'] or '-'} | {finding['message']} |"
            )
    else:
        lines.append("No identity findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_identity_audit",
            "tests": "1",
            "failures": "1" if payload["status"] == "fail" else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "identity"})
    if payload["status"] == "fail":
        failure = ET.SubElement(case, "failure", {"message": "eval identity audit failed"})
        failure.text = "\n".join(
            f"{finding['gate']} {finding['source']} {finding['key']}: {finding['message']}"
            for finding in payload["findings"]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_key_value(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    key, value = text.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("expected non-empty KEY")
    return key, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path, help="Prediction JSON/JSONL/CSV artifacts")
    parser.add_argument("--identity-key", action="append", default=[], help="Metadata key to summarize and optionally gate")
    parser.add_argument("--require-key", action="append", default=[], help="Metadata key that must be present with one value")
    parser.add_argument("--compare-key", action="append", default=[], help="Metadata key that must match across artifacts")
    parser.add_argument("--expect", action="append", default=[], type=parse_key_value, metavar="KEY=VALUE")
    parser.add_argument("--require-identity", action="store_true", help="Require every summarized identity key on every row")
    parser.add_argument("--min-artifacts", type=int, default=2)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_identity_audit_latest")
    return parser


def build_payload(artifacts: list[ArtifactIdentity], findings: list[Finding], keys: list[str]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "identity_keys": keys,
        "summary": {
            "artifacts": len(artifacts),
            "rows": sum(artifact.rows for artifact in artifacts),
            "findings": len(findings),
            "artifacts_with_inconsistent_identity": sum(1 for artifact in artifacts if artifact.inconsistent_keys),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keys = list(dict.fromkeys([*DEFAULT_IDENTITY_KEYS, *args.identity_key, *args.require_key, *args.compare_key, *[key for key, _ in args.expect]]))

    artifacts: list[ArtifactIdentity] = []
    findings: list[Finding] = []
    for path in args.artifacts:
        try:
            artifacts.append(summarize_artifact(path, keys))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding("read_error", str(path), "", "", "readable predictions", str(exc)))

    findings.extend(evaluate(artifacts, args))
    payload = build_payload(artifacts, findings, keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    findings_path = args.output_dir / f"{args.output_stem}_findings.csv"
    markdown_path = args.output_dir / f"{args.output_stem}.md"
    junit_path = args.output_dir / f"{args.output_stem}_junit.xml"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, artifacts, keys)
    write_findings_csv(findings_path, findings)
    write_markdown(markdown_path, payload, keys)
    write_junit(junit_path, payload)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
