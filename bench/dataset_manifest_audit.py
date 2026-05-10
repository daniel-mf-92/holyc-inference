#!/usr/bin/env python3
"""Audit local eval curation manifests before publishing packed datasets.

The audit is offline-only. It verifies that curation manifests retain source
provenance, that recorded file digests still match local artifacts, and that an
optional HCEval pack manifest is consistent with the curated JSONL manifest.
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


@dataclass(frozen=True)
class ManifestFinding:
    severity: str
    kind: str
    scope: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(base: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def append_required_field_findings(
    findings: list[ManifestFinding],
    manifest: dict[str, Any],
    fields: list[str],
) -> None:
    for field in fields:
        value = manifest.get(field)
        if value is None or value == "" or value == [] or value == {}:
            findings.append(
                ManifestFinding("error", "missing_required_field", field, f"manifest field {field!r} is empty")
            )


def audit_file_digest(
    findings: list[ManifestFinding],
    *,
    path: Path | None,
    expected_sha256: str | None,
    kind: str,
    missing_kind: str,
) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path) if path else "", "expected_sha256": expected_sha256 or ""}
    if path is None:
        findings.append(ManifestFinding("error", missing_kind, kind, f"{kind} path is missing"))
        return info
    if not path.exists():
        findings.append(ManifestFinding("error", missing_kind, str(path), f"{kind} file does not exist"))
        return info
    actual = file_sha256(path)
    info["actual_sha256"] = actual
    info["bytes"] = path.stat().st_size
    if expected_sha256 and actual != expected_sha256:
        findings.append(
            ManifestFinding(
                "error",
                f"{kind}_sha256_mismatch",
                str(path),
                f"expected {expected_sha256}, got {actual}",
            )
        )
    return info


def canonical_jsonl_sha256(path: Path) -> str:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def jsonl_record_metadata(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object row")
            rows.append(row)

    dataset_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    dataset_split_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "")
        split = str(row.get("split") or "")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
        split_group = dataset_split_counts.setdefault(dataset, {})
        split_group[split] = split_group.get(split, 0) + 1

    return {
        "record_count": len(rows),
        "record_ids": [str(row.get("record_id") or row.get("id") or "") for row in rows],
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "dataset_split_counts": {
            dataset: dict(sorted(split_counts.items()))
            for dataset, split_counts in sorted(dataset_split_counts.items())
        },
    }


def append_manifest_count_findings(
    findings: list[ManifestFinding],
    *,
    manifest: dict[str, Any],
    output_info: dict[str, Any],
) -> None:
    actual_record_count = output_info.get("actual_record_count")
    if actual_record_count is not None and manifest.get("record_count") != actual_record_count:
        findings.append(
            ManifestFinding(
                "error",
                "curated_record_count_mismatch",
                "record_count",
                f"manifest record_count {manifest.get('record_count')} does not match curated rows {actual_record_count}",
            )
        )

    for field in ["dataset_counts", "split_counts", "dataset_split_counts"]:
        expected = manifest.get(field)
        actual = output_info.get(f"actual_{field}")
        if expected and actual is not None and expected != actual:
            findings.append(
                ManifestFinding(
                    "error",
                    f"curated_{field}_mismatch",
                    field,
                    f"manifest {field} does not match curated JSONL rows",
                )
            )

    selected_ids = list(manifest.get("selected_record_ids") or [])
    actual_ids = list(output_info.get("actual_record_ids") or [])
    if selected_ids and actual_ids and selected_ids != actual_ids:
        findings.append(
            ManifestFinding(
                "error",
                "curated_record_id_order_mismatch",
                "selected_record_ids",
                "manifest selected_record_ids do not match curated JSONL record order",
            )
        )


def audit_curated_output(
    findings: list[ManifestFinding],
    *,
    path: Path | None,
    expected_normalized_sha256: str | None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path) if path else "",
        "expected_normalized_sha256": expected_normalized_sha256 or "",
    }
    if path is None:
        findings.append(
            ManifestFinding("error", "missing_curated_output", "curated_output", "curated output path is missing")
        )
        return info
    if not path.exists():
        findings.append(ManifestFinding("error", "missing_curated_output", str(path), "curated output file does not exist"))
        return info
    info["file_sha256"] = file_sha256(path)
    info["bytes"] = path.stat().st_size
    try:
        normalized = canonical_jsonl_sha256(path)
        metadata = jsonl_record_metadata(path)
    except ValueError as exc:
        findings.append(ManifestFinding("error", "curated_output_read_error", str(path), str(exc)))
        return info
    info["actual_normalized_sha256"] = normalized
    info["actual_record_count"] = metadata["record_count"]
    info["actual_record_ids"] = metadata["record_ids"]
    info["actual_dataset_counts"] = metadata["dataset_counts"]
    info["actual_split_counts"] = metadata["split_counts"]
    info["actual_dataset_split_counts"] = metadata["dataset_split_counts"]
    if expected_normalized_sha256 and normalized != expected_normalized_sha256:
        findings.append(
            ManifestFinding(
                "error",
                "curated_output_normalized_sha256_mismatch",
                str(path),
                f"expected {expected_normalized_sha256}, got {normalized}",
            )
        )
    return info


def audit_pack_manifest(
    findings: list[ManifestFinding],
    *,
    curation_manifest: dict[str, Any],
    pack_manifest: dict[str, Any],
    pack_path: Path | None,
) -> dict[str, Any]:
    pack_info = audit_file_digest(
        findings,
        path=pack_path,
        expected_sha256=pack_manifest.get("binary_sha256"),
        kind="pack_output",
        missing_kind="missing_pack_output",
    )

    if pack_manifest.get("source_sha256") != curation_manifest.get("normalized_sha256"):
        findings.append(
            ManifestFinding(
                "error",
                "pack_source_sha256_mismatch",
                "pack_manifest",
                "pack manifest source_sha256 does not match curation normalized_sha256",
            )
        )
    if pack_manifest.get("record_count") != curation_manifest.get("record_count"):
        findings.append(
            ManifestFinding(
                "error",
                "pack_record_count_mismatch",
                "pack_manifest",
                "pack manifest record_count does not match curation manifest",
            )
        )

    selected_ids = list(curation_manifest.get("selected_record_ids") or [])
    pack_ids = [record.get("record_id") for record in pack_manifest.get("records", []) if isinstance(record, dict)]
    if selected_ids and pack_ids and pack_ids != selected_ids:
        findings.append(
            ManifestFinding(
                "error",
                "pack_record_id_order_mismatch",
                "pack_manifest",
                "pack manifest records do not match selected_record_ids order",
            )
        )

    return {
        "format": pack_manifest.get("format", ""),
        "record_count": pack_manifest.get("record_count"),
        "records": pack_ids,
        "source_sha256": pack_manifest.get("source_sha256", ""),
        "binary_sha256": pack_manifest.get("binary_sha256", ""),
        "pack_output": pack_info,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    findings: list[ManifestFinding] = []
    manifest_path = args.manifest.resolve()
    root = args.root.resolve()

    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(ManifestFinding("error", "manifest_read_error", str(manifest_path), str(exc)))
        manifest = {}

    required = ["format", "source_name", "source_version", "license", "source", "output", "normalized_sha256"]
    if args.require_pack_manifest:
        required.append("pack_output")
    append_required_field_findings(findings, manifest, required)

    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    source_path = resolve_manifest_path(root, source.get("path"))
    output_path = resolve_manifest_path(root, manifest.get("output"))
    source_info = audit_file_digest(
        findings,
        path=source_path,
        expected_sha256=source.get("sha256"),
        kind="source",
        missing_kind="missing_source",
    )
    output_info = audit_curated_output(
        findings,
        path=output_path,
        expected_normalized_sha256=manifest.get("normalized_sha256"),
    )
    append_manifest_count_findings(findings, manifest=manifest, output_info=output_info)

    pack_info: dict[str, Any] | None = None
    pack_manifest_path = args.pack_manifest
    if pack_manifest_path is not None:
        pack_manifest_path = pack_manifest_path.resolve()
    elif manifest.get("pack_output"):
        candidate = resolve_manifest_path(root, manifest.get("pack_output"))
        if candidate is not None:
            pack_manifest_path = candidate.with_suffix(candidate.suffix + ".manifest.json")

    if pack_manifest_path is not None and pack_manifest_path.exists():
        try:
            pack_manifest = load_json(pack_manifest_path)
            pack_path = resolve_manifest_path(root, pack_manifest.get("output") or manifest.get("pack_output"))
            pack_info = audit_pack_manifest(
                findings,
                curation_manifest=manifest,
                pack_manifest=pack_manifest,
                pack_path=pack_path,
            )
            pack_info["manifest_path"] = str(pack_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(ManifestFinding("error", "pack_manifest_read_error", str(pack_manifest_path), str(exc)))
    elif args.require_pack_manifest:
        findings.append(
            ManifestFinding(
                "error",
                "missing_pack_manifest",
                str(pack_manifest_path or ""),
                "pack manifest is required but was not found",
            )
        )

    status = "fail" if findings and args.fail_on_findings else "pass"
    return {
        "created_at": iso_now(),
        "status": status,
        "manifest_path": str(manifest_path),
        "format": manifest.get("format", ""),
        "source_name": manifest.get("source_name", ""),
        "source_version": manifest.get("source_version", ""),
        "license": manifest.get("license", ""),
        "record_count": manifest.get("record_count", 0),
        "dataset_counts": manifest.get("dataset_counts", {}),
        "split_counts": manifest.get("split_counts", {}),
        "source": source_info,
        "curated_output": output_info,
        "pack_manifest": pack_info,
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "kind", "scope", "detail"])
        writer.writeheader()
        writer.writerows(findings)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Manifest Audit",
        "",
        f"- status: {report['status']}",
        f"- manifest: `{report['manifest_path']}`",
        f"- records: {report['record_count']}",
        f"- findings: {len(report['findings'])}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(f"- {finding['severity']} {finding['kind']} `{finding['scope']}`: {finding['detail']}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = report["findings"] if report["status"] == "fail" else []
    suite = ET.Element(
        "testsuite",
        {
            "name": "dataset_manifest_audit",
            "tests": "1",
            "failures": "1" if failures else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "manifest_consistency"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": f"{len(failures)} manifest audit finding(s)"})
        failure.text = "\n".join(f"{item['kind']}: {item['detail']}" for item in failures)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Curation manifest JSON")
    parser.add_argument("--pack-manifest", type=Path, help="Optional HCEval pack manifest JSON")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root for relative manifest paths")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--csv", type=Path, help="Output findings CSV")
    parser.add_argument("--markdown", type=Path, help="Output Markdown summary")
    parser.add_argument("--junit", type=Path, help="Output JUnit XML")
    parser.add_argument("--require-pack-manifest", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    write_json(args.output, report)
    if args.csv:
        write_csv(args.csv, report["findings"])
    if args.markdown:
        write_markdown(args.markdown, report)
    if args.junit:
        write_junit(args.junit, report)
    print(f"dataset_manifest_audit_status={report['status']} findings={len(report['findings'])}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
