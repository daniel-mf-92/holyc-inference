#!/usr/bin/env python3
"""Audit quant block manifests against local Q4_0/Q8_0 artifacts.

This host-side audit keeps quantization validation reproducible by checking that
manifest metadata still matches the raw block streams produced by packers or
builds. It never launches QEMU or touches the TempleOS guest.
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


BLOCK_BYTES = {"q4_0": 18, "q8_0": 34}
BLOCK_ELEMENTS = 32


@dataclass(frozen=True)
class QuantManifestFinding:
    severity: str
    kind: str
    scope: str
    detail: str


@dataclass(frozen=True)
class QuantArtifactAudit:
    path: str
    format: str
    expected_sha256: str
    actual_sha256: str
    expected_bytes: int | None
    actual_bytes: int | None
    expected_blocks: int | None
    actual_blocks: int | None
    expected_elements: int | None
    element_capacity: int | None


@dataclass(frozen=True)
class QuantManifestAudit:
    timestamp: str
    manifest_path: str
    status: str
    artifact_count: int
    artifacts: list[QuantArtifactAudit]
    findings: list[QuantManifestFinding]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("artifacts", manifest.get("blocks"))
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return [manifest]


def int_field(entry: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = entry.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def audit_entry(
    entry: dict[str, Any],
    *,
    root: Path,
    index: int,
    findings: list[QuantManifestFinding],
) -> QuantArtifactAudit:
    scope = str(entry.get("name") or entry.get("path") or f"artifact[{index}]")
    quant_format = str(entry.get("format") or entry.get("quant_format") or "").lower()
    path = resolve_path(root, entry.get("path") or entry.get("file"))
    expected_sha256 = str(entry.get("sha256") or entry.get("block_sha256") or "")
    expected_bytes = int_field(entry, "bytes", "byte_count")
    expected_blocks = int_field(entry, "blocks", "block_count")
    expected_elements = int_field(entry, "elements", "element_count")

    actual_sha256 = ""
    actual_bytes: int | None = None
    actual_blocks: int | None = None
    element_capacity: int | None = None

    if quant_format not in BLOCK_BYTES:
        findings.append(
            QuantManifestFinding("error", "unsupported_format", scope, f"unsupported quant format {quant_format!r}")
        )

    if path is None:
        findings.append(QuantManifestFinding("error", "missing_path", scope, "artifact path is missing"))
    elif not path.exists():
        findings.append(QuantManifestFinding("error", "missing_artifact", str(path), "artifact file does not exist"))
    else:
        actual_bytes = path.stat().st_size
        actual_sha256 = file_sha256(path)
        if quant_format in BLOCK_BYTES:
            block_bytes = BLOCK_BYTES[quant_format]
            if actual_bytes % block_bytes:
                findings.append(
                    QuantManifestFinding(
                        "error",
                        "block_size_mismatch",
                        str(path),
                        f"{actual_bytes} bytes is not a multiple of {block_bytes}",
                    )
                )
            actual_blocks = actual_bytes // block_bytes
            element_capacity = actual_blocks * BLOCK_ELEMENTS

    if expected_sha256 and actual_sha256 and expected_sha256 != actual_sha256:
        findings.append(
            QuantManifestFinding(
                "error",
                "sha256_mismatch",
                scope,
                f"expected {expected_sha256}, got {actual_sha256}",
            )
        )
    if expected_bytes is not None and actual_bytes is not None and expected_bytes != actual_bytes:
        findings.append(
            QuantManifestFinding(
                "error",
                "byte_count_mismatch",
                scope,
                f"expected {expected_bytes} bytes, got {actual_bytes}",
            )
        )
    if expected_blocks is not None and actual_blocks is not None and expected_blocks != actual_blocks:
        findings.append(
            QuantManifestFinding(
                "error",
                "block_count_mismatch",
                scope,
                f"expected {expected_blocks} blocks, got {actual_blocks}",
            )
        )
    if expected_elements is not None and element_capacity is not None and expected_elements > element_capacity:
        findings.append(
            QuantManifestFinding(
                "error",
                "element_count_over_capacity",
                scope,
                f"expected {expected_elements} elements exceeds capacity {element_capacity}",
            )
        )

    return QuantArtifactAudit(
        path=str(path) if path else "",
        format=quant_format,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        expected_bytes=expected_bytes,
        actual_bytes=actual_bytes,
        expected_blocks=expected_blocks,
        actual_blocks=actual_blocks,
        expected_elements=expected_elements,
        element_capacity=element_capacity,
    )


def audit_manifest(manifest_path: Path, *, root: Path | None = None) -> QuantManifestAudit:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} did not contain a JSON object")
    audit_root = root or manifest_path.parent
    findings: list[QuantManifestFinding] = []
    artifacts = [
        audit_entry(entry, root=audit_root, index=index, findings=findings)
        for index, entry in enumerate(manifest_entries(manifest))
    ]
    if not artifacts:
        findings.append(QuantManifestFinding("error", "empty_manifest", str(manifest_path), "manifest has no artifacts"))
    status = "fail" if findings else "pass"
    return QuantManifestAudit(
        timestamp=iso_now(),
        manifest_path=str(manifest_path),
        status=status,
        artifact_count=len(artifacts),
        artifacts=artifacts,
        findings=findings,
    )


def audit_to_json(audit: QuantManifestAudit) -> dict[str, Any]:
    return asdict(audit)


def write_json(path: Path, audit: QuantManifestAudit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit_to_json(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, audit: QuantManifestAudit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "format",
                "actual_bytes",
                "actual_blocks",
                "element_capacity",
                "expected_elements",
                "finding_count",
            ],
        )
        writer.writeheader()
        for artifact in audit.artifacts:
            writer.writerow(
                {
                    "path": artifact.path,
                    "format": artifact.format,
                    "actual_bytes": artifact.actual_bytes,
                    "actual_blocks": artifact.actual_blocks,
                    "element_capacity": artifact.element_capacity,
                    "expected_elements": artifact.expected_elements,
                    "finding_count": len(audit.findings),
                }
            )


def write_markdown(path: Path, audit: QuantManifestAudit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quant Manifest Audit",
        "",
        f"- Status: {audit.status.upper()}",
        f"- Artifacts: {audit.artifact_count}",
        f"- Findings: {len(audit.findings)}",
        "",
        "## Artifacts",
    ]
    for artifact in audit.artifacts:
        lines.append(
            f"- {artifact.format} {artifact.path}: {artifact.actual_blocks} blocks, "
            f"{artifact.element_capacity} element capacity"
        )
    if not audit.artifacts:
        lines.append("- none")
    lines.extend(["", "## Findings"])
    lines.extend(f"- {finding.kind}: {finding.detail}" for finding in audit.findings)
    if not audit.findings:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, audit: QuantManifestAudit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {"name": "quant_manifest_audit", "tests": "1", "failures": "1" if audit.findings else "0"},
    )
    case = ET.SubElement(suite, "testcase", {"name": "manifest"})
    if audit.findings:
        failure = ET.SubElement(case, "failure", {"message": audit.findings[0].detail})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in audit.findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_manifest(args.manifest, root=args.root)
    if args.output:
        write_json(args.output, audit)
    else:
        print(json.dumps(audit_to_json(audit), indent=2, sort_keys=True))
    if args.csv:
        write_csv(args.csv, audit)
    if args.markdown:
        write_markdown(args.markdown, audit)
    if args.junit:
        write_junit(args.junit, audit)
    return 1 if args.fail_on_findings and audit.findings else 0


if __name__ == "__main__":
    sys.exit(main())
