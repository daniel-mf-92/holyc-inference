#!/usr/bin/env python3
"""Audit QEMU benchmark artifacts for input provenance consistency.

This host-side tool reads saved benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
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

import qemu_prompt_bench


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    status: str
    prompt_source: str
    prompt_count: int | None
    prompt_suite_sha256: str
    prompt_live_checked: bool
    image_path: str
    image_exists_recorded: bool | None
    image_live_checked: bool
    qemu_args_files: int
    qemu_args_live_checked: int
    findings: int


@dataclass(frozen=True)
class Finding:
    source: str
    severity: str
    kind: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "artifact root must be a JSON object"
    return payload, ""


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def artifact_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def check_prompt_suite(path: Path, payload: dict[str, Any], findings: list[Finding], args: argparse.Namespace) -> bool:
    suite = payload.get("prompt_suite")
    if not isinstance(suite, dict):
        findings.append(Finding(str(path), "error", "missing_prompt_suite", "prompt_suite", "prompt_suite object is absent"))
        return False

    prompt_source = as_text(suite.get("source"))
    if not prompt_source:
        findings.append(Finding(str(path), "error", "missing_prompt_source", "prompt_suite.source", "prompt suite source is absent"))
        return False

    prompt_path = artifact_path(Path.cwd(), prompt_source)
    if not prompt_path.exists():
        if args.require_live_inputs:
            findings.append(Finding(str(path), "error", "prompt_source_missing", "prompt_suite.source", f"{prompt_source} does not exist"))
        return False

    try:
        cases = qemu_prompt_bench.load_prompt_cases(prompt_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(Finding(str(path), "error", "prompt_source_unreadable", "prompt_suite.source", str(exc)))
        return False

    expected = qemu_prompt_bench.prompt_suite_metadata(prompt_path, cases)
    for field in (
        "prompt_count",
        "suite_sha256",
        "prompt_bytes_total",
        "prompt_bytes_min",
        "prompt_bytes_max",
        "expected_token_prompts",
        "expected_tokens_total",
    ):
        if suite.get(field) != expected.get(field):
            findings.append(
                Finding(
                    str(path),
                    "error",
                    "prompt_suite_drift",
                    f"prompt_suite.{field}",
                    f"stored={suite.get(field)!r} expected={expected.get(field)!r}",
                )
            )
    return True


def check_file_metadata(
    artifact: Path,
    metadata: Any,
    field: str,
    findings: list[Finding],
    args: argparse.Namespace,
) -> bool:
    if not isinstance(metadata, dict):
        findings.append(Finding(str(artifact), "error", "metadata_type", field, "file metadata must be an object"))
        return False

    recorded_path = as_text(metadata.get("path"))
    if not recorded_path:
        findings.append(Finding(str(artifact), "error", "missing_path", f"{field}.path", "metadata path is absent"))
        return False

    recorded_exists = as_bool(metadata.get("exists"))
    if recorded_exists is None:
        findings.append(Finding(str(artifact), "error", "missing_exists", f"{field}.exists", "metadata exists flag must be boolean"))

    if recorded_exists is True:
        if as_int(metadata.get("size_bytes")) is None:
            findings.append(Finding(str(artifact), "error", "missing_size", f"{field}.size_bytes", "existing file metadata must include size_bytes"))
        if as_int(metadata.get("mtime_ns")) is None:
            findings.append(Finding(str(artifact), "error", "missing_mtime", f"{field}.mtime_ns", "existing file metadata must include mtime_ns"))

    live_path = artifact_path(Path.cwd(), recorded_path)
    if not live_path.exists():
        if args.require_live_inputs and recorded_exists is True:
            findings.append(Finding(str(artifact), "error", "live_input_missing", f"{field}.path", f"{recorded_path} does not exist"))
        return False

    live_stat = live_path.stat()
    if recorded_exists is False and args.require_live_inputs:
        findings.append(Finding(str(artifact), "error", "recorded_missing_live_present", f"{field}.exists", f"{recorded_path} exists now but artifact recorded exists=false"))
    if as_int(metadata.get("size_bytes")) not in (None, live_stat.st_size):
        findings.append(
            Finding(str(artifact), "error", "size_drift", f"{field}.size_bytes", f"stored={metadata.get('size_bytes')} expected={live_stat.st_size}")
        )

    stored_sha = as_text(metadata.get("sha256"))
    if stored_sha:
        live_sha = qemu_prompt_bench.file_sha256(live_path)
        if stored_sha != live_sha:
            findings.append(Finding(str(artifact), "error", "sha256_drift", f"{field}.sha256", f"stored={stored_sha} expected={live_sha}"))
    elif recorded_exists is True and args.require_file_sha256:
        findings.append(Finding(str(artifact), "error", "missing_sha256", f"{field}.sha256", "existing file metadata must include sha256"))

    return True


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactRecord | None, list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return None, [Finding(str(path), "error", "load_error", "", error)]

    findings: list[Finding] = []
    prompt_live_checked = check_prompt_suite(path, payload, findings, args)

    image = payload.get("image")
    image_live_checked = False
    if image in (None, {}):
        if args.require_image_metadata:
            findings.append(Finding(str(path), "error", "missing_image_metadata", "image", "image metadata is absent"))
    else:
        image_live_checked = check_file_metadata(path, image, "image", findings, args)

    qemu_args_files = payload.get("qemu_args_files")
    if qemu_args_files in (None, ""):
        qemu_args_files = []
    if not isinstance(qemu_args_files, list):
        findings.append(Finding(str(path), "error", "qemu_args_files_type", "qemu_args_files", "qemu_args_files must be a list"))
        qemu_args_files = []

    qemu_args_live_checked = 0
    for index, metadata in enumerate(qemu_args_files):
        if check_file_metadata(path, metadata, f"qemu_args_files[{index}]", findings, args):
            qemu_args_live_checked += 1

    suite = payload.get("prompt_suite") if isinstance(payload.get("prompt_suite"), dict) else {}
    image_meta = image if isinstance(image, dict) else {}
    record = ArtifactRecord(
        source=str(path),
        status=as_text(payload.get("status")) or "unknown",
        prompt_source=as_text(suite.get("source")),
        prompt_count=as_int(suite.get("prompt_count")),
        prompt_suite_sha256=as_text(suite.get("suite_sha256")),
        prompt_live_checked=prompt_live_checked,
        image_path=as_text(image_meta.get("path")),
        image_exists_recorded=as_bool(image_meta.get("exists")),
        image_live_checked=image_live_checked,
        qemu_args_files=len(qemu_args_files),
        qemu_args_live_checked=qemu_args_live_checked,
        findings=len(findings),
    )
    return record, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ArtifactRecord], list[Finding]]:
    records: list[ArtifactRecord] = []
    findings: list[Finding] = []
    seen = 0
    for path in iter_input_files(paths, args.pattern):
        seen += 1
        record, artifact_findings = audit_artifact(path, args)
        if record is not None:
            records.append(record)
        findings.extend(artifact_findings)
    if seen == 0:
        findings.append(Finding("", "error", "no_inputs", "inputs", "no benchmark artifacts matched"))
    return records, findings


def write_outputs(records: list[ArtifactRecord], findings: list[Finding], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "pass" if not any(finding.severity == "error" for finding in findings) else "fail"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "summary": {
            "artifacts": len(records),
            "findings": len(findings),
            "prompt_live_checked": sum(1 for record in records if record.prompt_live_checked),
            "qemu_args_live_checked": sum(record.qemu_args_live_checked for record in records),
        },
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (args.output_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(ArtifactRecord.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    with (args.output_dir / f"{stem}_findings.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(Finding.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))

    lines = [
        "# QEMU Input Provenance Audit",
        "",
        f"Status: {status}",
        f"Artifacts: {len(records)}",
        f"Findings: {len(findings)}",
    ]
    if findings:
        lines.extend(["", "## Findings", ""])
        for finding in findings:
            lines.append(f"- {finding.severity}: {finding.kind} {finding.field} - {finding.detail}")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    suite = ET.Element("testsuite", name="qemu_input_provenance_audit", tests=str(max(1, len(records))), failures=str(sum(1 for finding in findings if finding.severity == "error")))
    if not records:
        case = ET.SubElement(suite, "testcase", name="inputs")
        failure = ET.SubElement(case, "failure", message="no benchmark artifacts matched")
        failure.text = "no benchmark artifacts matched"
    for record in records:
        case = ET.SubElement(suite, "testcase", name=record.source)
        record_findings = [finding for finding in findings if finding.source == record.source and finding.severity == "error"]
        if record_findings:
            failure = ET.SubElement(case, "failure", message=f"{len(record_findings)} provenance findings")
            failure.text = "\n".join(f"{finding.kind}: {finding.field}: {finding.detail}" for finding in record_findings)
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)
    with (args.output_dir / f"{stem}_junit.xml").open("ab") as handle:
        handle.write(b"\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob pattern used when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_input_provenance_audit_latest")
    parser.add_argument("--require-live-inputs", action="store_true", help="fail when recorded live inputs are missing or recorded as absent")
    parser.add_argument("--require-file-sha256", action="store_true", help="require sha256 metadata for existing image and QEMU args files")
    parser.add_argument("--require-image-metadata", action="store_true", help="fail when artifact image metadata is absent")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    records, findings = audit(args.inputs, args)
    write_outputs(records, findings, args)
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
