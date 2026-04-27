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
import sys
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
    profile: str
    model: str
    quantization: str
    prompt_suite_sha256: str
    prompts: int | None
    measured_runs: int
    warmup_runs: int
    median_tok_per_s: float | None
    max_memory_bytes: int | None
    command_airgap_status: str
    command_findings: list[str]


@dataclass(frozen=True)
class PromptSuiteDrift:
    key: str
    hashes: list[str]
    sources: list[str]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def summarize_qemu_report(path: Path, report: dict[str, Any]) -> ArtifactSummary:
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
    command = first_run.get("command") or (warmups[0].get("command") if warmups else None)
    airgap_status, findings = command_status(command)

    return ArtifactSummary(
        source=str(path),
        artifact_type="qemu_prompt",
        status=qemu_report_status(report, runs, warmups),
        generated_at=str(report.get("generated_at", "")),
        profile=first_present(first_run, ("profile",), str(report.get("profile", ""))),
        model=first_present(first_run, ("model",), str(report.get("model", ""))),
        quantization=first_present(first_run, ("quantization",), str(report.get("quantization", ""))),
        prompt_suite_sha256=str(prompt_suite.get("suite_sha256", "")),
        prompts=parse_int(prompt_suite.get("prompt_count")) or parse_int(suite_summary.get("prompts")),
        measured_runs=len(runs),
        warmup_runs=len(warmups),
        median_tok_per_s=parse_float(suite_summary.get("tok_per_s_median"))
        or (statistics.median(tok_values) if tok_values else None),
        max_memory_bytes=parse_int(suite_summary.get("memory_bytes_max"))
        or (max(memory_values) if memory_values else None),
        command_airgap_status=airgap_status,
        command_findings=findings,
    )


def summarize_matrix_report(path: Path, report: dict[str, Any]) -> list[ArtifactSummary]:
    cells = [row for row in report.get("cells", []) if isinstance(row, dict)]
    summaries: list[ArtifactSummary] = []
    for cell in cells:
        airgap_status, findings = command_status(cell.get("command"))
        summaries.append(
            ArtifactSummary(
                source=str(path),
                artifact_type="bench_matrix_cell",
                status=str(cell.get("status", report.get("status", "unknown"))),
                generated_at=str(report.get("generated_at", "")),
                profile=str(cell.get("profile", "")),
                model=str(cell.get("model", "")),
                quantization=str(cell.get("quantization", "")),
                prompt_suite_sha256=str(cell.get("prompt_suite_sha256", "")),
                prompts=parse_int(cell.get("prompts")),
                measured_runs=parse_int(cell.get("measured_runs")) or 0,
                warmup_runs=parse_int(cell.get("warmup_runs")) or 0,
                median_tok_per_s=parse_float(cell.get("median_tok_per_s")),
                max_memory_bytes=parse_int(cell.get("max_memory_bytes")),
                command_airgap_status=airgap_status,
                command_findings=findings,
            )
        )
    return summaries


def load_summaries(paths: Iterable[Path]) -> list[ArtifactSummary]:
    summaries: list[ArtifactSummary] = []
    for path in sorted(set(iter_report_files(paths))):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            continue
        if isinstance(report.get("benchmarks"), list):
            summaries.append(summarize_qemu_report(path, report))
        elif isinstance(report.get("cells"), list):
            summaries.extend(summarize_matrix_report(path, report))
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
    if any(summary.status == "fail" for summary in summaries):
        return "fail"
    return "pass"


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
        "",
    ]
    if report["artifacts"]:
        lines.extend(
            [
                "| Type | Status | Air-gap | Profile | Model | Quant | Prompts | Runs | Warmups | Median tok/s | Max memory bytes | Source |",
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for artifact in report["artifacts"]:
            lines.append(
                "| {artifact_type} | {status} | {command_airgap_status} | {profile} | {model} | "
                "{quantization} | {prompts} | {measured_runs} | {warmup_runs} | "
                "{median_tok_per_s} | {max_memory_bytes} | {source} |".format(
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
    return "\n".join(lines) + "\n"


def write_csv(summaries: list[ArtifactSummary], path: Path) -> None:
    fields = [
        "source",
        "artifact_type",
        "status",
        "generated_at",
        "profile",
        "model",
        "quantization",
        "prompt_suite_sha256",
        "prompts",
        "measured_runs",
        "warmup_runs",
        "median_tok_per_s",
        "max_memory_bytes",
        "command_airgap_status",
        "command_findings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            row = asdict(summary)
            row["command_findings"] = json.dumps(summary.command_findings, separators=(",", ":"))
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


def write_report(summaries: list[ArtifactSummary], output_dir: Path) -> tuple[Path, list[PromptSuiteDrift]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    drift = prompt_suite_drift(summaries)
    report = {
        "generated_at": iso_now(),
        "status": index_status(summaries),
        "artifacts": [asdict(summary) for summary in summaries],
        "prompt_suite_drift": [asdict(finding) for finding in drift],
    }
    json_path = output_dir / "bench_result_index_latest.json"
    md_path = output_dir / "bench_result_index_latest.md"
    csv_path = output_dir / "bench_result_index_latest.csv"
    drift_csv_path = output_dir / "bench_result_index_prompt_suite_drift_latest.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(summaries, csv_path)
    write_drift_csv(drift, drift_csv_path)
    return json_path, drift


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
        "--fail-on-drift",
        action="store_true",
        help="Return non-zero if comparable artifacts use different prompt-suite hashes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or [Path("bench/results")]
    try:
        summaries = load_summaries(inputs)
        output, drift = write_report(summaries, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    status = index_status(summaries)
    print(f"wrote_json={output}")
    print(f"status={status}")
    print(f"artifacts={len(summaries)}")
    print(f"prompt_suite_drift={len(drift)}")
    if args.fail_on_airgap and status == "fail":
        return 1
    if args.fail_on_drift and drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
