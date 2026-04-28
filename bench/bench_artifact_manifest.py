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
    generated_age_seconds: int | None
    profile: str
    model: str
    quantization: str
    prompt_suite_sha256: str
    command_sha256: str
    environment_sha256: str
    host_platform: str
    host_machine: str
    qemu_version: str
    qemu_bin: str
    prompts: int | None
    measured_runs: int
    warmup_runs: int
    median_tok_per_s: float | None
    wall_tok_per_s_median: float | None
    ttft_us_p95: float | None
    host_overhead_pct_median: float | None
    host_child_cpu_us_median: float | None
    host_child_cpu_pct_median: float | None
    host_child_tok_per_cpu_s_median: float | None
    host_child_peak_rss_bytes_max: int | None
    us_per_token_median: float | None
    wall_us_per_token_median: float | None
    max_memory_bytes: int | None
    telemetry_status: str
    telemetry_findings: list[str]
    command_hash_status: str
    command_hash_findings: list[str]
    command_airgap_status: str
    commit: str
    current_commit: str
    current_commit_match: bool | None
    commit_status: str
    commit_findings: list[str]
    freshness_status: str
    freshness_findings: list[str]
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
        generated_age_seconds=summary.generated_age_seconds,
        profile=summary.profile,
        model=summary.model,
        quantization=summary.quantization,
        prompt_suite_sha256=summary.prompt_suite_sha256,
        command_sha256=summary.command_sha256,
        environment_sha256=summary.environment_sha256,
        host_platform=summary.host_platform,
        host_machine=summary.host_machine,
        qemu_version=summary.qemu_version,
        qemu_bin=summary.qemu_bin,
        prompts=summary.prompts,
        measured_runs=summary.measured_runs,
        warmup_runs=summary.warmup_runs,
        median_tok_per_s=summary.median_tok_per_s,
        wall_tok_per_s_median=summary.wall_tok_per_s_median,
        ttft_us_p95=summary.ttft_us_p95,
        host_overhead_pct_median=summary.host_overhead_pct_median,
        host_child_cpu_us_median=summary.host_child_cpu_us_median,
        host_child_cpu_pct_median=summary.host_child_cpu_pct_median,
        host_child_tok_per_cpu_s_median=summary.host_child_tok_per_cpu_s_median,
        host_child_peak_rss_bytes_max=summary.host_child_peak_rss_bytes_max,
        us_per_token_median=summary.us_per_token_median,
        wall_us_per_token_median=summary.wall_us_per_token_median,
        max_memory_bytes=summary.max_memory_bytes,
        telemetry_status=summary.telemetry_status,
        telemetry_findings=summary.telemetry_findings,
        command_hash_status=summary.command_hash_status,
        command_hash_findings=summary.command_hash_findings,
        command_airgap_status=summary.command_airgap_status,
        commit=summary.commit,
        current_commit=summary.current_commit,
        current_commit_match=summary.current_commit_match,
        commit_status=summary.commit_status,
        commit_findings=summary.commit_findings,
        freshness_status=summary.freshness_status,
        freshness_findings=summary.freshness_findings,
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
        or item.command_hash_status == "fail"
        or item.commit_status == "fail"
        or item.freshness_status == "fail"
        for item in artifacts
    ):
        return "fail"
    return "pass"


def has_airgap_failures(artifacts: Iterable[ManifestArtifact]) -> bool:
    return any(artifact.command_airgap_status == "fail" for artifact in artifacts)


def has_telemetry_failures(artifacts: Iterable[ManifestArtifact]) -> bool:
    return any(artifact.telemetry_status == "fail" for artifact in artifacts)


def has_commit_metadata_failures(artifacts: Iterable[ManifestArtifact]) -> bool:
    return any(artifact.commit_status == "fail" for artifact in artifacts)


def has_command_hash_metadata_failures(artifacts: Iterable[ManifestArtifact]) -> bool:
    return any(artifact.command_hash_status == "fail" for artifact in artifacts)


def has_stale_commits(artifacts: Iterable[ManifestArtifact]) -> bool:
    return any(artifact.current_commit_match is False for artifact in artifacts)


def has_stale_artifacts(artifacts: Iterable[ManifestArtifact]) -> bool:
    return any(artifact.freshness_status == "fail" for artifact in artifacts)


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
                "| Key | Status | Air-gap | Telemetry | Command Hash | Freshness | Commit | Host | QEMU | Prompts | Runs | Warmups | Age seconds | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Host child CPU us | Host child CPU % | Host child tok/CPU s | Max host child RSS bytes | Guest us/token | Wall us/token | Max memory bytes | Command SHA256 | Env SHA256 | Artifact SHA256 | Source |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for artifact in latest:
            values = {key: format_value(value) for key, value in artifact.items()}
            values["qemu"] = format_value(artifact.get("qemu_version") or artifact.get("qemu_bin"))
            lines.append(
                "| {key} | {status} | {command_airgap_status} | {telemetry_status} | {command_hash_status} | {freshness_status} | {commit_status}:{commit} | "
                "{host_platform}/{host_machine} | {qemu} | {prompts} | {measured_runs} | "
                "{warmup_runs} | {generated_age_seconds} | {median_tok_per_s} | {wall_tok_per_s_median} | "
                "{ttft_us_p95} | {host_overhead_pct_median} | {host_child_cpu_us_median} | "
                "{host_child_cpu_pct_median} | {host_child_tok_per_cpu_s_median} | "
                "{host_child_peak_rss_bytes_max} | {us_per_token_median} | {wall_us_per_token_median} | "
                "{max_memory_bytes} | {command_sha256} | {environment_sha256} | "
                "{sha256} | {source} |".format(**values)
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
        "generated_age_seconds",
        "profile",
        "model",
        "quantization",
        "prompt_suite_sha256",
        "command_sha256",
        "environment_sha256",
        "host_platform",
        "host_machine",
        "qemu_version",
        "qemu_bin",
        "prompts",
        "measured_runs",
        "warmup_runs",
        "median_tok_per_s",
        "wall_tok_per_s_median",
        "ttft_us_p95",
        "host_overhead_pct_median",
        "host_child_cpu_us_median",
        "host_child_cpu_pct_median",
        "host_child_tok_per_cpu_s_median",
        "host_child_peak_rss_bytes_max",
        "us_per_token_median",
        "wall_us_per_token_median",
        "max_memory_bytes",
        "telemetry_status",
        "telemetry_findings",
        "command_hash_status",
        "command_hash_findings",
        "command_airgap_status",
        "commit",
        "current_commit",
        "current_commit_match",
        "commit_status",
        "commit_findings",
        "freshness_status",
        "freshness_findings",
        "sha256",
        "bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for artifact in artifacts:
            row = asdict(artifact)
            row["telemetry_findings"] = json.dumps(artifact.telemetry_findings, separators=(",", ":"))
            row["command_hash_findings"] = json.dumps(artifact.command_hash_findings, separators=(",", ":"))
            row["commit_findings"] = json.dumps(artifact.commit_findings, separators=(",", ":"))
            row["freshness_findings"] = json.dumps(artifact.freshness_findings, separators=(",", ":"))
            writer.writerow({field: row[field] for field in fields})


def junit_report(report: dict[str, object]) -> str:
    history = [row for row in report["history"] if isinstance(row, dict)]
    failed_artifacts = [row for row in history if row.get("status") == "fail"]
    airgap_failures = [row for row in history if row.get("command_airgap_status") == "fail"]
    telemetry_failures = [row for row in history if row.get("telemetry_status") == "fail"]
    command_hash_failures = [row for row in history if row.get("command_hash_status") == "fail"]
    commit_failures = [row for row in history if row.get("commit_status") == "fail"]
    freshness_failures = [row for row in history if row.get("freshness_status") == "fail"]
    missing_latest = not report["latest_artifacts"]
    failures = (
        int(bool(failed_artifacts))
        + int(bool(airgap_failures))
        + int(bool(telemetry_failures))
        + int(bool(command_hash_failures))
        + int(bool(commit_failures))
        + int(bool(freshness_failures))
        + int(missing_latest)
    )

    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_bench_artifact_manifest",
            "tests": "7",
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

    command_hash_case = ET.SubElement(suite, "testcase", {"name": "command_hash_metadata"})
    if command_hash_failures:
        failure = ET.SubElement(
            command_hash_case,
            "failure",
            {
                "type": "benchmark_command_hash_metadata_failure",
                "message": f"{len(command_hash_failures)} benchmark artifact(s) have inconsistent command hashes",
            },
        )
        failure.text = "\n".join(
            f"{row.get('source', '')}: {json.dumps(row.get('command_hash_findings', []), separators=(',', ':'))}"
            for row in command_hash_failures
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

    freshness_case = ET.SubElement(suite, "testcase", {"name": "artifact_freshness"})
    if freshness_failures:
        failure = ET.SubElement(
            freshness_case,
            "failure",
            {
                "type": "benchmark_artifact_stale",
                "message": f"{len(freshness_failures)} benchmark artifact(s) failed freshness policy",
            },
        )
        failure.text = "\n".join(
            f"{row.get('source', '')}: {json.dumps(row.get('freshness_findings', []), separators=(',', ':'))}"
            for row in freshness_failures
        )

    return ET.tostring(suite, encoding="unicode") + "\n"


def write_manifest(
    summaries: list[bench_result_index.ArtifactSummary], output_dir: Path
) -> tuple[Path, str, list[ManifestArtifact]]:
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
    history_csv_path = output_dir / "bench_artifact_manifest_history_latest.csv"
    junit_path = output_dir / "bench_artifact_manifest_junit_latest.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(latest, csv_path)
    write_csv(history, history_csv_path)
    junit_path.write_text(junit_report(report), encoding="utf-8")
    return json_path, str(report["status"]), history


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
        "--fail-on-telemetry",
        action="store_true",
        help="Return non-zero if any manifest artifact is missing required benchmark telemetry",
    )
    parser.add_argument(
        "--fail-on-commit-metadata",
        action="store_true",
        help="Return non-zero if any manifest artifact has missing or inconsistent commit metadata",
    )
    parser.add_argument(
        "--fail-on-command-hash-metadata",
        action="store_true",
        help="Return non-zero if any manifest artifact has inconsistent command_sha256 metadata",
    )
    parser.add_argument(
        "--fail-on-stale-commit",
        action="store_true",
        help="Return non-zero if any manifest artifact commit differs from the current git commit",
    )
    parser.add_argument(
        "--max-artifact-age-hours",
        type=float,
        help="Mark artifacts as stale if generated_at is older than this many hours",
    )
    parser.add_argument(
        "--fail-on-stale-artifact",
        action="store_true",
        help="Return non-zero if any artifact exceeds --max-artifact-age-hours",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or [Path("bench/results")]
    max_artifact_age_seconds = None
    if args.max_artifact_age_hours is not None:
        if args.max_artifact_age_hours < 0:
            print("error: --max-artifact-age-hours must be non-negative", file=sys.stderr)
            return 2
        max_artifact_age_seconds = int(args.max_artifact_age_hours * 3600)
    elif args.fail_on_stale_artifact:
        print("error: --fail-on-stale-artifact requires --max-artifact-age-hours", file=sys.stderr)
        return 2
    try:
        summaries = bench_result_index.load_summaries(
            inputs,
            max_artifact_age_seconds=max_artifact_age_seconds,
        )
        output, status, artifacts = write_manifest(summaries, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote_json={output}")
    print(f"status={status}")
    print(f"artifacts={len(summaries)}")
    print(f"command_hash_failures={sum(1 for artifact in artifacts if artifact.command_hash_status == 'fail')}")
    print(f"freshness_failures={sum(1 for artifact in artifacts if artifact.freshness_status == 'fail')}")
    if args.fail_on_airgap and has_airgap_failures(artifacts):
        return 1
    if args.fail_on_telemetry and has_telemetry_failures(artifacts):
        return 1
    if args.fail_on_commit_metadata and has_commit_metadata_failures(artifacts):
        return 1
    if args.fail_on_command_hash_metadata and has_command_hash_metadata_failures(artifacts):
        return 1
    if args.fail_on_stale_commit and has_stale_commits(artifacts):
        return 1
    if args.fail_on_stale_artifact and has_stale_artifacts(artifacts):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
