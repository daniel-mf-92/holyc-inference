#!/usr/bin/env python3
"""Audit offline eval dataset manifests for source/license policy.

This is a local-only curation gate. It reads manifests produced by
dataset_curate.py, checks declared source and license metadata against an
explicit policy, and writes CI-friendly JSON/Markdown/CSV/JUnit artifacts. It
never fetches remote datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PLACEHOLDER_VALUES = {"", "unknown", "todo", "tbd", "n/a", "na", "none", "unspecified"}
DEFAULT_DENIED_LICENSE_TERMS = {
    "all rights reserved",
    "non-commercial",
    "noncommercial",
    "no redistribution",
    "no-redistribution",
    "proprietary",
    "research only",
    "research-only",
}


@dataclass(frozen=True)
class LicenseFinding:
    source: str
    severity: str
    kind: str
    detail: str


@dataclass(frozen=True)
class LicenseArtifact:
    source: str
    status: str
    source_name: str
    source_version: str
    license: str
    normalized_license: str
    source_url: str
    source_url_scheme: str
    source_url_host: str
    record_count: int
    dataset_count: int
    split_count: int
    findings: list[LicenseFinding]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize(value: Any) -> str:
    return " ".join(clean(value).casefold().replace("_", "-").split())


def is_placeholder(value: Any) -> bool:
    return normalize(value) in PLACEHOLDER_VALUES


def normalized_set(values: Iterable[str]) -> set[str]:
    return {normalize(value) for value in values if normalize(value)}


def source_url_parts(value: Any) -> tuple[str, str, str]:
    text = clean(value)
    if not text:
        return "", "", ""
    parsed = urllib.parse.urlparse(text)
    return parsed.scheme.casefold(), (parsed.hostname or "").rstrip(".").casefold(), parsed.path


def finding(path: Path, severity: str, kind: str, detail: str) -> LicenseFinding:
    return LicenseFinding(str(path), severity, kind, detail)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def has_denied_term(normalized_license: str, denied_terms: set[str]) -> str:
    for term in sorted(denied_terms):
        if term and term in normalized_license:
            return term
    return ""


def audit_manifest(path: Path, args: argparse.Namespace) -> LicenseArtifact:
    findings: list[LicenseFinding] = []
    try:
        manifest = load_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(finding(path, "error", "manifest_read_error", str(exc)))
        return LicenseArtifact(str(path), "fail", "", "", "", "", "", "", "", 0, 0, 0, findings)

    source_name = clean(manifest.get("source_name"))
    source_version = clean(manifest.get("source_version"))
    license_text = clean(manifest.get("license"))
    normalized_license = normalize(license_text)
    source_url = clean(manifest.get("source_url"))
    scheme, host, _path = source_url_parts(source_url)
    record_count = int(manifest.get("record_count") or 0)
    dataset_count = len(manifest.get("dataset_counts") or {})
    split_count = len(manifest.get("split_counts") or {})

    required_fields = ["source_name", "source_version", "license"]
    if args.require_source_url:
        required_fields.append("source_url")
    for field in required_fields:
        if is_placeholder(manifest.get(field)):
            findings.append(finding(path, "error", f"missing_{field}", f"manifest field {field!r} is empty or placeholder"))

    if args.min_records is not None and record_count < args.min_records:
        findings.append(
            finding(path, "error", "record_count_below_min", f"record_count {record_count} is below {args.min_records}")
        )

    if args.allow_license and normalized_license not in args.allow_license:
        findings.append(
            finding(
                path,
                "error",
                "license_not_allowed",
                f"license {license_text!r} is not in the configured allowlist",
            )
        )

    denied_term = has_denied_term(normalized_license, args.deny_license)
    if denied_term:
        findings.append(
            finding(path, "error", "license_denied", f"license {license_text!r} matched denied term {denied_term!r}")
        )

    if source_url:
        if scheme not in args.allow_source_url_scheme:
            findings.append(
                finding(path, "error", "source_url_scheme_not_allowed", f"source_url scheme {scheme!r} is not allowed")
            )
        if args.allow_source_url_host and host not in args.allow_source_url_host:
            findings.append(
                finding(path, "error", "source_url_host_not_allowed", f"source_url host {host!r} is not allowed")
            )

    status = "fail" if findings and args.fail_on_findings else "pass"
    return LicenseArtifact(
        source=str(path),
        status=status,
        source_name=source_name,
        source_version=source_version,
        license=license_text,
        normalized_license=normalized_license,
        source_url=source_url,
        source_url_scheme=scheme,
        source_url_host=host,
        record_count=record_count,
        dataset_count=dataset_count,
        split_count=split_count,
        findings=findings,
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = [audit_manifest(path.resolve(), args) for path in args.input]
    findings = [finding for artifact in artifacts for finding in artifact.findings]
    status = "fail" if any(artifact.status == "fail" for artifact in artifacts) else "pass"
    return {
        "created_at": iso_now(),
        "status": status,
        "artifact_count": len(artifacts),
        "finding_count": len(findings),
        "allow_license": sorted(args.allow_license),
        "deny_license": sorted(args.deny_license),
        "allow_source_url_scheme": sorted(args.allow_source_url_scheme),
        "allow_source_url_host": sorted(args.allow_source_url_host),
        "artifacts": [
            {
                **asdict(artifact),
                "findings": [asdict(item) for item in artifact.findings],
            }
            for artifact in artifacts
        ],
        "findings": [asdict(item) for item in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset License Audit",
        "",
        f"- status: {report['status']}",
        f"- artifacts: {report['artifact_count']}",
        f"- findings: {report['finding_count']}",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in report["artifacts"]:
        lines.append(
            f"- {artifact['status']} `{artifact['source']}` license={artifact['license']!r} records={artifact['record_count']}"
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for item in report["findings"]:
            lines.append(f"- {item['severity']} {item['kind']} `{item['source']}`: {item['detail']}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "status",
        "source_name",
        "source_version",
        "license",
        "normalized_license",
        "source_url",
        "source_url_scheme",
        "source_url_host",
        "record_count",
        "dataset_count",
        "split_count",
        "finding_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for artifact in report["artifacts"]:
            writer.writerow({name: artifact.get(name, "") for name in fieldnames[:-1]} | {"finding_count": len(artifact["findings"])})


def write_findings_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "severity", "kind", "detail"])
        writer.writeheader()
        writer.writerows(report["findings"])


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "dataset_license_audit",
            "tests": str(max(1, report["artifact_count"])),
            "failures": str(report["finding_count"] if report["status"] == "fail" else 0),
            "errors": "0",
        },
    )
    for artifact in report["artifacts"] or [{"source": "dataset_license_audit", "findings": []}]:
        case = ET.SubElement(suite, "testcase", {"name": artifact["source"]})
        if report["status"] == "fail":
            failures = artifact.get("findings", [])
            if failures:
                failure = ET.SubElement(case, "failure", {"message": f"{len(failures)} license audit finding(s)"})
                failure.text = "\n".join(f"{item['kind']}: {item['detail']}" for item in failures)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Curation manifest JSON; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results/datasets"))
    parser.add_argument("--output-stem", default="dataset_license_audit_latest")
    parser.add_argument("--allow-license", action="append", default=[], help="Allowed normalized license value; repeatable")
    parser.add_argument(
        "--deny-license",
        action="append",
        default=sorted(DEFAULT_DENIED_LICENSE_TERMS),
        help="Denied normalized license substring; repeatable",
    )
    parser.add_argument(
        "--allow-source-url-scheme",
        action="append",
        default=["https"],
        help="Allowed source_url scheme; repeatable",
    )
    parser.add_argument("--allow-source-url-host", action="append", default=[], help="Allowed source_url host; repeatable")
    parser.add_argument("--require-source-url", action="store_true")
    parser.add_argument("--min-records", type=int)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args(argv)
    args.allow_license = normalized_set(args.allow_license)
    args.deny_license = normalized_set(args.deny_license)
    args.allow_source_url_scheme = normalized_set(args.allow_source_url_scheme)
    args.allow_source_url_host = normalized_set(args.allow_source_url_host)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    output_dir = args.output_dir
    output_stem = args.output_stem
    write_json(output_dir / f"{output_stem}.json", report)
    write_markdown(output_dir / f"{output_stem}.md", report)
    write_csv(output_dir / f"{output_stem}.csv", report)
    write_findings_csv(output_dir / f"{output_stem}_findings.csv", report)
    write_junit(output_dir / f"{output_stem}_junit.xml", report)
    print(f"dataset_license_audit_status={report['status']} findings={report['finding_count']}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
