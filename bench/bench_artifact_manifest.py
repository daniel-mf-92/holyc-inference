#!/usr/bin/env python3
"""Build a deterministic manifest for benchmark artifacts.

The manifest points CI and local tooling at the latest benchmark artifact for
each profile/model/quantization/prompt-suite key while retaining a compact
history. It is host-side only and never launches QEMU.
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
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_result_index


@dataclass(frozen=True)
class ManifestArtifact:
    key: str
    source: str
    artifact_type: str
    status: str
    generated_at: str
    profile: str
    model: str
    quantization: str
    prompt_suite_sha256: str
    measured_runs: int
    warmup_runs: int
    median_tok_per_s: float | None
    max_memory_bytes: int | None
    telemetry_status: str
    telemetry_findings: list[str]
    command_airgap_status: str
    commit: str
    current_commit: str
    current_commit_match: bool | None
    commit_status: str
    commit_findings: list[str]
    sha256: str
    bytes: int


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_key(summary: bench_result_index.ArtifactSummary) -> str:
    parts = (
        summary.profile or "-",
        summary.model or "-",
        summary.quantization or "-",
        summary.prompt_suite_sha256 or "no-suite",
    )
    return "/".join(parts)


def to_manifest_artifact(summary: bench_result_index.ArtifactSummary) -> ManifestArtifact:
    path = Path(summary.source)
    return ManifestArtifact(
        key=artifact_key(summary),
        source=summary.source,
        artifact_type=summary.artifact_type,
        status=summary.status,
        generated_at=summary.generated_at,
        profile=summary.profile,
        model=summary.model,
        quantization=summary.quantization,
        prompt_suite_sha256=summary.prompt_suite_sha256,
        measured_runs=summary.measured_runs,
        warmup_runs=summary.warmup_runs,
        median_tok_per_s=summary.median_tok_per_s,
        max_memory_bytes=summary.max_memory_bytes,
        telemetry_status=summary.telemetry_status,
        telemetry_findings=summary.telemetry_findings,
        command_airgap_status=summary.command_airgap_status,
        commit=summary.commit,
        current_commit=summary.current_commit,
        current_commit_match=summary.current_commit_match,
        commit_status=summary.commit_status,
        commit_findings=summary.commit_findings,
        sha256=file_sha256(path),
        bytes=path.stat().st_size,
    )


def latest_artifacts(artifacts: Iterable[ManifestArtifact]) -> list[ManifestArtifact]:
    latest: dict[str, ManifestArtifact] = {}
    for artifact in artifacts:
        current = latest.get(artifact.key)
        if current is None or (artifact.generated_at, artifact.source) > (
            current.generated_at,
            current.source,
        ):
            latest[artifact.key] = artifact
    return sorted(latest.values(), key=lambda item: item.key)


def manifest_status(artifacts: list[ManifestArtifact]) -> str:
    if not artifacts:
        return "fail"
    if any(
        item.status == "fail"
        or item.command_airgap_status == "fail"
        or item.telemetry_status == "fail"
        or item.commit_status == "fail"
        for item in artifacts
    ):
        return "fail"
    return "pass"


def format_value(value: object) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(report: dict[str, object]) -> str:
    latest = report["latest_artifacts"]
    assert isinstance(latest, list)
    lines = [
        "# Benchmark Artifact Manifest",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Latest keys: {len(latest)}",
        f"History artifacts: {report['history_artifacts']}",
        "",
    ]
    if latest:
        lines.extend(
            [
                "| Key | Status | Air-gap | Telemetry | Commit | Runs | Warmups | Median tok/s | Max memory bytes | SHA256 | Source |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for artifact in latest:
            lines.append(
                "| {key} | {status} | {command_airgap_status} | {telemetry_status} | {commit_status}:{commit} | {measured_runs} | "
                "{warmup_runs} | {median_tok_per_s} | {max_memory_bytes} | {sha256} | "
                "{source} |".format(
                    **{key: format_value(value) for key, value in artifact.items()}
                )
            )
    else:
        lines.append("No supported benchmark artifacts found.")
    return "\n".join(lines) + "\n"


def write_csv(artifacts: list[ManifestArtifact], path: Path) -> None:
    fields = [
        "key",
        "source",
        "artifact_type",
        "status",
        "generated_at",
        "profile",
        "model",
        "quantization",
        "prompt_suite_sha256",
        "measured_runs",
        "warmup_runs",
        "median_tok_per_s",
        "max_memory_bytes",
        "telemetry_status",
        "telemetry_findings",
        "command_airgap_status",
        "commit",
        "current_commit",
        "current_commit_match",
        "commit_status",
        "commit_findings",
        "sha256",
        "bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for artifact in artifacts:
            row = asdict(artifact)
            row["telemetry_findings"] = json.dumps(artifact.telemetry_findings, separators=(",", ":"))
            row["commit_findings"] = json.dumps(artifact.commit_findings, separators=(",", ":"))
            writer.writerow({field: row[field] for field in fields})


def junit_report(report: dict[str, object]) -> str:
    history = [row for row in report["history"] if isinstance(row, dict)]
    failed_artifacts = [row for row in history if row.get("status") == "fail"]
    airgap_failures = [row for row in history if row.get("command_airgap_status") == "fail"]
    telemetry_failures = [row for row in history if row.get("telemetry_status") == "fail"]
    commit_failures = [row for row in history if row.get("commit_status") == "fail"]
    missing_latest = not report["latest_artifacts"]
    failures = (
        int(bool(failed_artifacts))
        + int(bool(airgap_failures))
        + int(bool(telemetry_failures))
        + int(bool(commit_failures))
        + int(missing_latest)
    )

    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_bench_artifact_manifest",
            "tests": "5",
            "failures": str(failures),
        },
    )

    artifact_case = ET.SubElement(suite, "testcase", {"name": "artifact_status"})
    if failed_artifacts:
        failure = ET.SubElement(
            artifact_case,
            "failure",
            {
                "type": "benchmark_artifact_failure",
                "message": f"{len(failed_artifacts)} benchmark artifact(s) failed",
            },
        )
        failure.text = "\n".join(str(row.get("source", "")) for row in failed_artifacts)

    airgap_case = ET.SubElement(suite, "testcase", {"name": "airgap_status"})
    if airgap_failures:
        failure = ET.SubElement(
            airgap_case,
            "failure",
            {
                "type": "airgap_violation",
                "message": f"{len(airgap_failures)} benchmark artifact(s) violated air-gap policy",
            },
        )
        failure.text = "\n".join(str(row.get("source", "")) for row in airgap_failures)

    telemetry_case = ET.SubElement(suite, "testcase", {"name": "telemetry_coverage"})
    if telemetry_failures:
        failure = ET.SubElement(
            telemetry_case,
            "failure",
            {
                "type": "benchmark_telemetry_missing",
                "message": f"{len(telemetry_failures)} benchmark artifact(s) are missing telemetry",
            },
        )
        failure.text = "\n".join(
            f"{row.get('source', '')}: {json.dumps(row.get('telemetry_findings', []), separators=(',', ':'))}"
            for row in telemetry_failures
        )

    latest_case = ET.SubElement(suite, "testcase", {"name": "latest_artifacts_present"})
    if missing_latest:
        failure = ET.SubElement(
            latest_case,
            "failure",
            {
                "type": "manifest_empty",
                "message": "manifest contains no latest benchmark artifacts",
            },
        )
        failure.text = "No supported benchmark artifacts found."

    commit_case = ET.SubElement(suite, "testcase", {"name": "commit_metadata"})
    if commit_failures:
        failure = ET.SubElement(
            commit_case,
            "failure",
            {
                "type": "benchmark_commit_metadata_failure",
                "message": f"{len(commit_failures)} benchmark artifact(s) have inconsistent commit metadata",
            },
        )
        failure.text = "\n".join(
            f"{row.get('source', '')}: {json.dumps(row.get('commit_findings', []), separators=(',', ':'))}"
            for row in commit_failures
        )

    return ET.tostring(suite, encoding="unicode") + "\n"


def write_manifest(
    summaries: list[bench_result_index.ArtifactSummary], output_dir: Path
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = sorted(
        (to_manifest_artifact(summary) for summary in summaries),
        key=lambda item: (item.key, item.generated_at, item.source),
    )
    latest = latest_artifacts(history)
    report = {
        "generated_at": iso_now(),
        "status": manifest_status(history),
        "history_artifacts": len(history),
        "latest_artifacts": [asdict(artifact) for artifact in latest],
        "history": [asdict(artifact) for artifact in history],
    }

    json_path = output_dir / "bench_artifact_manifest_latest.json"
    md_path = output_dir / "bench_artifact_manifest_latest.md"
    csv_path = output_dir / "bench_artifact_manifest_latest.csv"
    junit_path = output_dir / "bench_artifact_manifest_junit_latest.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(latest, csv_path)
    junit_path.write_text(junit_report(report), encoding="utf-8")
    return json_path, str(report["status"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Benchmark report file or directory; defaults to bench/results",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument(
        "--fail-on-airgap",
        action="store_true",
        help="Return non-zero if any manifest artifact records a QEMU air-gap violation",
    )
    parser.add_argument(
        "--fail-on-stale-commit",
        action="store_true",
        help="Return non-zero if any manifest artifact commit differs from the current git commit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or [Path("bench/results")]
    try:
        summaries = bench_result_index.load_summaries(inputs)
        output, status = write_manifest(summaries, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote_json={output}")
    print(f"status={status}")
    print(f"artifacts={len(summaries)}")
    if args.fail_on_airgap and status == "fail":
        return 1
    if args.fail_on_stale_commit and any(summary.current_commit_match is False for summary in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
