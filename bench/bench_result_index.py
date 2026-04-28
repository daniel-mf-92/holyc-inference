#!/usr/bin/env python3
"""Index host-side QEMU benchmark result artifacts.

The indexer scans benchmark JSON reports, extracts comparable throughput and
memory summaries, and writes JSON/Markdown/CSV rollups under bench/results. It
does not launch QEMU; it only validates recorded commands for air-gap drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import airgap_audit


@dataclass(frozen=True)
class ArtifactSummary:
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
    prompts: int | None
    measured_runs: int
    warmup_runs: int
    median_tok_per_s: float | None
    wall_tok_per_s_median: float | None
    ttft_us_p95: float | None
    host_overhead_pct_median: float | None
    us_per_token_median: float | None
    wall_us_per_token_median: float | None
    max_memory_bytes: int | None
    telemetry_status: str
    telemetry_findings: list[str]
    command_hash_status: str
    command_hash_findings: list[str]
    command_airgap_status: str
    command_findings: list[str]
    commit: str
    current_commit: str
    current_commit_match: bool | None
    commit_status: str
    commit_findings: list[str]
    freshness_status: str
    freshness_findings: list[str]


@dataclass(frozen=True)
class PromptSuiteDrift:
    key: str
    hashes: list[str]
    sources: list[str]


@dataclass(frozen=True)
class CommandDrift:
    key: str
    hashes: list[str]
    sources: list[str]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def artifact_age_seconds(generated_at: str, now: datetime) -> int | None:
    generated = parse_iso_datetime(generated_at)
    if generated is None:
        return None
    return max(0, int((now - generated).total_seconds()))


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def first_present(row: dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return str(value)
    return default


def iter_report_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(
                child
                for child in path.rglob("*.json")
                if child.is_file() and looks_like_supported_report(child)
            )
        elif path.is_file() and looks_like_supported_report(path):
            yield path


def looks_like_supported_report(path: Path) -> bool:
    name = path.name
    return (
        (name.startswith("qemu_prompt_bench") or name.startswith("bench_matrix"))
        and not name.startswith("bench_result_index")
    )


def command_status(command: Any) -> tuple[str, list[str]]:
    normalized = airgap_audit.normalize_command(command)
    if normalized is None or not airgap_audit.qemu_like(normalized):
        return "not-qemu", []

    violations = airgap_audit.command_violations(normalized)
    return ("fail" if violations else "pass"), violations


def aggregate_command_status(commands: Iterable[tuple[str, Any]]) -> tuple[str, list[str]]:
    statuses: list[str] = []
    findings: list[str] = []
    for label, command in commands:
        status, command_findings = command_status(command)
        statuses.append(status)
        findings.extend(f"{label}: {finding}" for finding in command_findings)

    if any(status == "fail" for status in statuses):
        return "fail", findings
    if any(status == "pass" for status in statuses):
        return "pass", findings
    return "not-qemu", findings


def qemu_run_failed(row: dict[str, Any]) -> bool:
    returncode = parse_int(row.get("returncode"))
    return (returncode is not None and returncode != 0) or parse_bool(row.get("timed_out"))


def qemu_report_status(report: dict[str, Any], runs: list[dict[str, Any]], warmups: list[dict[str, Any]]) -> str:
    status = str(report.get("status", "unknown"))
    if any(qemu_run_failed(row) for row in runs + warmups):
        return "fail"
    if report.get("variability_findings"):
        return "fail"
    return status


def summary_float(report: dict[str, Any], summaries: list[dict[str, Any]], key: str) -> float | None:
    suite_summary = report.get("suite_summary") if isinstance(report.get("suite_summary"), dict) else {}
    value = parse_float(suite_summary.get(key))
    if value is not None:
        return value
    values = [parsed for row in summaries if (parsed := parse_float(row.get(key))) is not None]
    return statistics.median(values) if values else None


def telemetry_status(
    artifact_type: str,
    prompts: int | None,
    measured_runs: int,
    median_tok_per_s: float | None,
) -> tuple[str, list[str]]:
    findings: list[str] = []
    if prompts is not None and prompts <= 0:
        findings.append(f"non-positive prompt count: {prompts}")
    if measured_runs <= 0:
        findings.append(f"non-positive measured run count: {measured_runs}")
    if median_tok_per_s is None:
        findings.append("missing median tok/s")
    elif median_tok_per_s <= 0:
        findings.append(f"non-positive median tok/s: {median_tok_per_s}")
    return ("fail" if findings else "pass"), [f"{artifact_type}: {finding}" for finding in findings]


def commit_status(artifact_type: str, commit_values: Iterable[Any]) -> tuple[str, str, list[str]]:
    commits = sorted({str(value) for value in commit_values if value})
    if not commits:
        return "unknown", "", [f"{artifact_type}: missing commit metadata"]
    if len(commits) > 1:
        return "fail", ",".join(commits), [f"{artifact_type}: mixed commits: {','.join(commits)}"]
    return "pass", commits[0], []


def command_hash_status(artifact_type: str, hash_values: Iterable[Any]) -> tuple[str, str, list[str]]:
    hashes = sorted({str(value) for value in hash_values if value})
    if not hashes:
        return "unknown", "", [f"{artifact_type}: missing command_sha256 metadata"]
    if len(hashes) > 1:
        return "fail", ",".join(hashes), [f"{artifact_type}: mixed command_sha256 values: {','.join(hashes)}"]
    return "pass", hashes[0], []


def current_commit_match(commit: str, current_commit: str) -> bool | None:
    if not commit or current_commit == "unknown":
        return None
    return commit == current_commit


def freshness_status(
    artifact_type: str,
    generated_at: str,
    generated_age_seconds: int | None,
    max_artifact_age_seconds: int | None,
) -> tuple[str, list[str]]:
    if max_artifact_age_seconds is None:
        return "unchecked", []
    if generated_age_seconds is None:
        return "fail", [f"{artifact_type}: missing or invalid generated_at: {generated_at or '-'}"]
    if generated_age_seconds > max_artifact_age_seconds:
        return (
            "fail",
            [
                f"{artifact_type}: artifact age {generated_age_seconds}s exceeds "
                f"{max_artifact_age_seconds}s"
            ],
        )
    return "pass", []


def summarize_qemu_report(
    path: Path,
    report: dict[str, Any],
    current_commit: str,
    now: datetime,
    max_artifact_age_seconds: int | None,
) -> ArtifactSummary:
    runs = [row for row in report.get("benchmarks", []) if isinstance(row, dict)]
    warmups = [row for row in report.get("warmups", []) if isinstance(row, dict)]
    summaries = [row for row in report.get("summaries", []) if isinstance(row, dict)]
    suite_summary = report.get("suite_summary") if isinstance(report.get("suite_summary"), dict) else {}
    first_run = runs[0] if runs else {}
    prompt_suite = report.get("prompt_suite") if isinstance(report.get("prompt_suite"), dict) else {}

    tok_values = [
        value
        for row in summaries
        if (value := parse_float(row.get("tok_per_s_median"))) is not None
    ]
    memory_values = [
        value
        for row in summaries
        if (value := parse_int(row.get("memory_bytes_max"))) is not None
    ]
    airgap_status, findings = aggregate_command_status(
        [(f"measured[{index}]", row.get("command")) for index, row in enumerate(runs)]
        + [(f"warmup[{index}]", row.get("command")) for index, row in enumerate(warmups)]
    )

    prompts = parse_int(prompt_suite.get("prompt_count"))
    if prompts is None:
        prompts = parse_int(suite_summary.get("prompts"))
    median_tok_per_s = parse_float(suite_summary.get("tok_per_s_median"))
    if median_tok_per_s is None and tok_values:
        median_tok_per_s = statistics.median(tok_values)
    max_memory_bytes = parse_int(suite_summary.get("memory_bytes_max"))
    if max_memory_bytes is None and memory_values:
        max_memory_bytes = max(memory_values)
    telem_status, telem_findings = telemetry_status(
        "qemu_prompt",
        prompts,
        len(runs),
        median_tok_per_s,
    )
    commit_state, commit, commit_findings = commit_status(
        "qemu_prompt",
        [row.get("commit") for row in runs + warmups if isinstance(row, dict)],
    )
    command_hash_state, command_sha256, command_hash_findings = command_hash_status(
        "qemu_prompt",
        [report.get("command_sha256")]
        + [row.get("command_sha256") for row in runs + warmups if isinstance(row, dict)],
    )
    if command_hash_state == "fail":
        findings.extend(command_hash_findings)
    generated_at = str(report.get("generated_at", ""))
    generated_age = artifact_age_seconds(generated_at, now)
    fresh_state, fresh_findings = freshness_status(
        "qemu_prompt",
        generated_at,
        generated_age,
        max_artifact_age_seconds,
    )

    return ArtifactSummary(
        source=str(path),
        artifact_type="qemu_prompt",
        status=qemu_report_status(report, runs, warmups),
        generated_at=generated_at,
        generated_age_seconds=generated_age,
        profile=first_present(first_run, ("profile",), str(report.get("profile", ""))),
        model=first_present(first_run, ("model",), str(report.get("model", ""))),
        quantization=first_present(first_run, ("quantization",), str(report.get("quantization", ""))),
        prompt_suite_sha256=str(prompt_suite.get("suite_sha256", "")),
        command_sha256=command_sha256,
        prompts=prompts,
        measured_runs=len(runs),
        warmup_runs=len(warmups),
        median_tok_per_s=median_tok_per_s,
        wall_tok_per_s_median=summary_float(report, summaries, "wall_tok_per_s_median"),
        ttft_us_p95=summary_float(report, summaries, "ttft_us_p95"),
        host_overhead_pct_median=summary_float(report, summaries, "host_overhead_pct_median"),
        us_per_token_median=summary_float(report, summaries, "us_per_token_median"),
        wall_us_per_token_median=summary_float(report, summaries, "wall_us_per_token_median"),
        max_memory_bytes=max_memory_bytes,
        telemetry_status=telem_status,
        telemetry_findings=telem_findings,
        command_hash_status=command_hash_state,
        command_hash_findings=command_hash_findings,
        command_airgap_status=airgap_status,
        command_findings=findings,
        commit=commit,
        current_commit=current_commit,
        current_commit_match=current_commit_match(commit, current_commit),
        commit_status=commit_state,
        commit_findings=commit_findings,
        freshness_status=fresh_state,
        freshness_findings=fresh_findings,
    )


def summarize_matrix_report(
    path: Path,
    report: dict[str, Any],
    current_commit: str,
    now: datetime,
    max_artifact_age_seconds: int | None,
) -> list[ArtifactSummary]:
    cells = [row for row in report.get("cells", []) if isinstance(row, dict)]
    summaries: list[ArtifactSummary] = []
    generated_at = str(report.get("generated_at", ""))
    generated_age = artifact_age_seconds(generated_at, now)
    fresh_state, fresh_findings = freshness_status(
        "bench_matrix_cell",
        generated_at,
        generated_age,
        max_artifact_age_seconds,
    )
    for cell in cells:
        airgap_status, findings = command_status(cell.get("command"))
        prompts = parse_int(cell.get("prompts"))
        measured_runs = parse_int(cell.get("measured_runs")) or 0
        median_tok_per_s = parse_float(cell.get("median_tok_per_s"))
        telem_status, telem_findings = telemetry_status(
            "bench_matrix_cell",
            prompts,
            measured_runs,
            median_tok_per_s,
        )
        commit_state, commit, commit_findings = commit_status(
            "bench_matrix_cell",
            [cell.get("commit")],
        )
        command_hash_state, command_sha256, command_hash_findings = command_hash_status(
            "bench_matrix_cell",
            [cell.get("command_sha256")],
        )
        if command_hash_state == "fail":
            findings.extend(command_hash_findings)
        summaries.append(
            ArtifactSummary(
                source=str(path),
                artifact_type="bench_matrix_cell",
                status=str(cell.get("status", report.get("status", "unknown"))),
                generated_at=generated_at,
                generated_age_seconds=generated_age,
                profile=str(cell.get("profile", "")),
                model=str(cell.get("model", "")),
                quantization=str(cell.get("quantization", "")),
                prompt_suite_sha256=str(cell.get("prompt_suite_sha256", "")),
                command_sha256=command_sha256,
                prompts=prompts,
                measured_runs=measured_runs,
                warmup_runs=parse_int(cell.get("warmup_runs")) or 0,
                median_tok_per_s=median_tok_per_s,
                wall_tok_per_s_median=parse_float(cell.get("wall_tok_per_s_median")),
                ttft_us_p95=parse_float(cell.get("ttft_us_p95")),
                host_overhead_pct_median=parse_float(cell.get("host_overhead_pct_median")),
                us_per_token_median=parse_float(cell.get("us_per_token_median")),
                wall_us_per_token_median=parse_float(cell.get("wall_us_per_token_median")),
                max_memory_bytes=parse_int(cell.get("max_memory_bytes")),
                telemetry_status=telem_status,
                telemetry_findings=telem_findings,
                command_hash_status=command_hash_state,
                command_hash_findings=command_hash_findings,
                command_airgap_status=airgap_status,
                command_findings=findings,
                commit=commit,
                current_commit=current_commit,
                current_commit_match=current_commit_match(commit, current_commit),
                commit_status=commit_state,
                commit_findings=commit_findings,
                freshness_status=fresh_state,
                freshness_findings=fresh_findings,
            )
        )
    return summaries


def load_summaries(
    paths: Iterable[Path],
    max_artifact_age_seconds: int | None = None,
    now: datetime | None = None,
) -> list[ArtifactSummary]:
    summaries: list[ArtifactSummary] = []
    current_commit = git_commit(Path.cwd())
    effective_now = now or datetime.now(timezone.utc)
    for path in sorted(set(iter_report_files(paths))):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            continue
        if isinstance(report.get("benchmarks"), list):
            summaries.append(
                summarize_qemu_report(path, report, current_commit, effective_now, max_artifact_age_seconds)
            )
        elif isinstance(report.get("cells"), list):
            summaries.extend(
                summarize_matrix_report(path, report, current_commit, effective_now, max_artifact_age_seconds)
            )
    return sorted(
        summaries,
        key=lambda item: (
            item.generated_at,
            item.artifact_type,
            item.profile,
            item.model,
            item.quantization,
            item.source,
        ),
    )


def index_status(summaries: list[ArtifactSummary]) -> str:
    if any(summary.command_airgap_status == "fail" for summary in summaries):
        return "fail"
    if any(summary.telemetry_status == "fail" for summary in summaries):
        return "fail"
    if any(summary.commit_status == "fail" for summary in summaries):
        return "fail"
    if any(summary.command_hash_status == "fail" for summary in summaries):
        return "fail"
    if any(summary.freshness_status == "fail" for summary in summaries):
        return "fail"
    if any(summary.status == "fail" for summary in summaries):
        return "fail"
    return "pass"


def has_airgap_failures(summaries: list[ArtifactSummary]) -> bool:
    return any(summary.command_airgap_status == "fail" for summary in summaries)


def has_telemetry_failures(summaries: list[ArtifactSummary]) -> bool:
    return any(summary.telemetry_status == "fail" for summary in summaries)


def has_commit_metadata_failures(summaries: list[ArtifactSummary]) -> bool:
    return any(summary.commit_status == "fail" for summary in summaries)


def has_command_hash_metadata_failures(summaries: list[ArtifactSummary]) -> bool:
    return any(summary.command_hash_status == "fail" for summary in summaries)


def has_freshness_failures(summaries: list[ArtifactSummary]) -> bool:
    return any(summary.freshness_status == "fail" for summary in summaries)


def commit_drift(summaries: list[ArtifactSummary]) -> list[ArtifactSummary]:
    return [summary for summary in summaries if summary.current_commit_match is False]


def prompt_suite_drift(summaries: list[ArtifactSummary]) -> list[PromptSuiteDrift]:
    by_key: dict[str, dict[str, set[str]]] = {}
    for summary in summaries:
        if not summary.prompt_suite_sha256:
            continue
        key = "/".join(
            (
                summary.profile or "-",
                summary.model or "-",
                summary.quantization or "-",
            )
        )
        by_key.setdefault(key, {}).setdefault(summary.prompt_suite_sha256, set()).add(summary.source)

    findings: list[PromptSuiteDrift] = []
    for key, hash_sources in sorted(by_key.items()):
        if len(hash_sources) <= 1:
            continue
        findings.append(
            PromptSuiteDrift(
                key=key,
                hashes=sorted(hash_sources),
                sources=sorted(source for sources in hash_sources.values() for source in sources),
            )
        )
    return findings


def command_drift(summaries: list[ArtifactSummary]) -> list[CommandDrift]:
    by_key: dict[str, dict[str, set[str]]] = {}
    for summary in summaries:
        if not summary.command_sha256:
            continue
        key = "/".join(
            (
                summary.profile or "-",
                summary.model or "-",
                summary.quantization or "-",
                summary.prompt_suite_sha256 or "no-suite",
            )
        )
        by_key.setdefault(key, {}).setdefault(summary.command_sha256, set()).add(summary.source)

    findings: list[CommandDrift] = []
    for key, hash_sources in sorted(by_key.items()):
        if len(hash_sources) <= 1:
            continue
        findings.append(
            CommandDrift(
                key=key,
                hashes=sorted(hash_sources),
                sources=sorted(source for sources in hash_sources.values() for source in sources),
            )
        )
    return findings


def format_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Result Index",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Artifacts: {len(report['artifacts'])}",
        f"Command drift: {len(report['command_drift'])}",
        "",
    ]
    if report["artifacts"]:
        lines.extend(
            [
                "| Type | Status | Air-gap | Telemetry | Freshness | Commit | Profile | Model | Quant | Prompt suite | Command SHA256 | Prompts | Runs | Warmups | Age seconds | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Guest us/token | Wall us/token | Max memory bytes | Source |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for artifact in report["artifacts"]:
            lines.append(
                "| {artifact_type} | {status} | {command_airgap_status} | {telemetry_status} | {freshness_status} | {commit_status}:{commit} | {profile} | {model} | "
                "{quantization} | {prompt_suite_sha256} | {command_sha256} | {prompts} | {measured_runs} | {warmup_runs} | "
                "{generated_age_seconds} | {median_tok_per_s} | {wall_tok_per_s_median} | {ttft_us_p95} | "
                "{host_overhead_pct_median} | {us_per_token_median} | {wall_us_per_token_median} | "
                "{max_memory_bytes} | {source} |".format(
                    **{key: format_value(value) for key, value in artifact.items()}
                )
            )
    else:
        lines.append("No supported benchmark artifacts found.")

    if report["prompt_suite_drift"]:
        lines.extend(
            [
                "",
                "## Prompt Suite Drift",
                "",
                "| Profile/Model/Quant | Hashes | Sources |",
                "| --- | ---: | ---: |",
            ]
        )
        for finding in report["prompt_suite_drift"]:
            lines.append(
                "| {key} | {hashes} | {sources} |".format(
                    key=finding["key"],
                    hashes=len(finding["hashes"]),
                    sources=len(finding["sources"]),
                )
            )
    else:
        lines.extend(["", "Prompt suite drift: none detected."])

    if report["command_drift"]:
        lines.extend(
            [
                "",
                "## Command Drift",
                "",
                "| Profile/Model/Quant/Prompt suite | Command hashes | Sources |",
                "| --- | ---: | ---: |",
            ]
        )
        for finding in report["command_drift"]:
            lines.append(
                "| {key} | {hashes} | {sources} |".format(
                    key=finding["key"],
                    hashes=len(finding["hashes"]),
                    sources=len(finding["sources"]),
                )
            )
    else:
        lines.extend(["", "Command drift: none detected."])
    return "\n".join(lines) + "\n"


def junit_report(report: dict[str, Any]) -> str:
    artifacts = [row for row in report["artifacts"] if isinstance(row, dict)]
    drift = [row for row in report.get("prompt_suite_drift", []) if isinstance(row, dict)]
    command_drift_rows = [row for row in report.get("command_drift", []) if isinstance(row, dict)]
    failed_artifacts = [row for row in artifacts if row.get("status") == "fail"]
    airgap_failures = [row for row in artifacts if row.get("command_airgap_status") == "fail"]
    telemetry_failures = [row for row in artifacts if row.get("telemetry_status") == "fail"]
    commit_failures = [row for row in artifacts if row.get("commit_status") == "fail"]
    command_hash_failures = [row for row in artifacts if row.get("command_hash_status") == "fail"]
    freshness_failures = [row for row in artifacts if row.get("freshness_status") == "fail"]
    failures = (
        int(bool(failed_artifacts))
        + int(bool(airgap_failures))
        + int(bool(telemetry_failures))
        + int(bool(commit_failures))
        + int(bool(command_hash_failures))
        + int(bool(freshness_failures))
        + int(bool(drift))
        + int(bool(command_drift_rows))
    )

    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_bench_result_index",
            "tests": "8",
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
                "message": f"{len(airgap_failures)} benchmark command(s) violated air-gap policy",
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
                "message": f"{len(telemetry_failures)} benchmark artifact(s) are missing required telemetry",
            },
        )
        failure.text = "\n".join(
            f"{row.get('source', '')}: {json.dumps(row.get('telemetry_findings', []), separators=(',', ':'))}"
            for row in telemetry_failures
        )

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

    drift_case = ET.SubElement(suite, "testcase", {"name": "prompt_suite_drift"})
    if drift:
        failure = ET.SubElement(
            drift_case,
            "failure",
            {
                "type": "prompt_suite_drift",
                "message": f"{len(drift)} comparable benchmark key(s) have prompt-suite drift",
            },
        )
        failure.text = "\n".join(str(row.get("key", "")) for row in drift)

    command_drift_case = ET.SubElement(suite, "testcase", {"name": "command_drift"})
    if command_drift_rows:
        failure = ET.SubElement(
            command_drift_case,
            "failure",
            {
                "type": "command_drift",
                "message": f"{len(command_drift_rows)} comparable benchmark key(s) have command-hash drift",
            },
        )
        failure.text = "\n".join(str(row.get("key", "")) for row in command_drift_rows)

    return ET.tostring(suite, encoding="unicode") + "\n"


def write_csv(summaries: list[ArtifactSummary], path: Path) -> None:
    fields = [
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
        "prompts",
        "measured_runs",
        "warmup_runs",
        "median_tok_per_s",
        "wall_tok_per_s_median",
        "ttft_us_p95",
        "host_overhead_pct_median",
        "us_per_token_median",
        "wall_us_per_token_median",
        "max_memory_bytes",
        "telemetry_status",
        "telemetry_findings",
        "command_hash_status",
        "command_hash_findings",
        "command_airgap_status",
        "command_findings",
        "commit",
        "current_commit",
        "current_commit_match",
        "commit_status",
        "commit_findings",
        "freshness_status",
        "freshness_findings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            row = asdict(summary)
            row["telemetry_findings"] = json.dumps(summary.telemetry_findings, separators=(",", ":"))
            row["command_hash_findings"] = json.dumps(summary.command_hash_findings, separators=(",", ":"))
            row["command_findings"] = json.dumps(summary.command_findings, separators=(",", ":"))
            row["commit_findings"] = json.dumps(summary.commit_findings, separators=(",", ":"))
            row["freshness_findings"] = json.dumps(summary.freshness_findings, separators=(",", ":"))
            writer.writerow({field: row[field] for field in fields})


def write_drift_csv(findings: list[PromptSuiteDrift], path: Path) -> None:
    fields = ["key", "hash_count", "source_count", "hashes", "sources"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    "key": finding.key,
                    "hash_count": len(finding.hashes),
                    "source_count": len(finding.sources),
                    "hashes": json.dumps(finding.hashes, separators=(",", ":")),
                    "sources": json.dumps(finding.sources, separators=(",", ":")),
                }
            )


def write_command_drift_csv(findings: list[CommandDrift], path: Path) -> None:
    fields = ["key", "hash_count", "source_count", "hashes", "sources"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    "key": finding.key,
                    "hash_count": len(finding.hashes),
                    "source_count": len(finding.sources),
                    "hashes": json.dumps(finding.hashes, separators=(",", ":")),
                    "sources": json.dumps(finding.sources, separators=(",", ":")),
                }
            )


def write_report(summaries: list[ArtifactSummary], output_dir: Path) -> tuple[Path, list[PromptSuiteDrift], list[CommandDrift]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    drift = prompt_suite_drift(summaries)
    command_drift_findings = command_drift(summaries)
    report = {
        "generated_at": iso_now(),
        "status": index_status(summaries),
        "artifacts": [asdict(summary) for summary in summaries],
        "prompt_suite_drift": [asdict(finding) for finding in drift],
        "command_drift": [asdict(finding) for finding in command_drift_findings],
    }
    json_path = output_dir / "bench_result_index_latest.json"
    md_path = output_dir / "bench_result_index_latest.md"
    csv_path = output_dir / "bench_result_index_latest.csv"
    drift_csv_path = output_dir / "bench_result_index_prompt_suite_drift_latest.csv"
    command_drift_csv_path = output_dir / "bench_result_index_command_drift_latest.csv"
    junit_path = output_dir / "bench_result_index_junit_latest.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    junit_path.write_text(junit_report(report), encoding="utf-8")
    write_csv(summaries, csv_path)
    write_drift_csv(drift, drift_csv_path)
    write_command_drift_csv(command_drift_findings, command_drift_csv_path)
    return json_path, drift, command_drift_findings


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
        help="Return non-zero if any indexed command violates air-gap policy",
    )
    parser.add_argument(
        "--fail-on-telemetry",
        action="store_true",
        help="Return non-zero if any indexed artifact is missing required benchmark telemetry",
    )
    parser.add_argument(
        "--fail-on-commit-metadata",
        action="store_true",
        help="Return non-zero if any artifact has missing or inconsistent commit metadata",
    )
    parser.add_argument(
        "--fail-on-command-hash-metadata",
        action="store_true",
        help="Return non-zero if any artifact has inconsistent command_sha256 metadata",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Return non-zero if comparable artifacts use different prompt-suite hashes",
    )
    parser.add_argument(
        "--fail-on-command-drift",
        action="store_true",
        help="Return non-zero if comparable artifacts use different command_sha256 hashes",
    )
    parser.add_argument(
        "--fail-on-stale-commit",
        action="store_true",
        help="Return non-zero if any artifact commit differs from the current git commit",
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
        summaries = load_summaries(inputs, max_artifact_age_seconds=max_artifact_age_seconds)
        output, drift, command_drift_findings = write_report(summaries, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    status = index_status(summaries)
    print(f"wrote_json={output}")
    print(f"status={status}")
    print(f"artifacts={len(summaries)}")
    print(f"prompt_suite_drift={len(drift)}")
    print(f"command_drift={len(command_drift_findings)}")
    print(f"freshness_failures={sum(1 for summary in summaries if summary.freshness_status == 'fail')}")
    if args.fail_on_airgap and has_airgap_failures(summaries):
        return 1
    if args.fail_on_telemetry and has_telemetry_failures(summaries):
        return 1
    if args.fail_on_commit_metadata and has_commit_metadata_failures(summaries):
        return 1
    if args.fail_on_command_hash_metadata and has_command_hash_metadata_failures(summaries):
        return 1
    if args.fail_on_drift and drift:
        return 1
    if args.fail_on_command_drift and command_drift_findings:
        return 1
    if args.fail_on_stale_commit and commit_drift(summaries):
        return 1
    if args.fail_on_stale_artifact and has_freshness_failures(summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
