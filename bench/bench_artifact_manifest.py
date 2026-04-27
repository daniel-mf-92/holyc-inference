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
    command_airgap_status: str
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
        command_airgap_status=summary.command_airgap_status,
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
    if any(item.status == "fail" or item.command_airgap_status == "fail" for item in artifacts):
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
                "| Key | Status | Air-gap | Runs | Warmups | Median tok/s | Max memory bytes | SHA256 | Source |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for artifact in latest:
            lines.append(
                "| {key} | {status} | {command_airgap_status} | {measured_runs} | "
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
        "command_airgap_status",
        "sha256",
        "bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for artifact in artifacts:
            row = asdict(artifact)
            writer.writerow({field: row[field] for field in fields})


def write_manifest(summaries: list[bench_result_index.ArtifactSummary], output_dir: Path) -> Path:
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
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(latest, csv_path)
    return json_path


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or [Path("bench/results")]
    try:
        summaries = bench_result_index.load_summaries(inputs)
        output = write_manifest(summaries, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    status = "fail" if any(
        summary.status == "fail" or summary.command_airgap_status == "fail" for summary in summaries
    ) else "pass"
    print(f"wrote_json={output}")
    print(f"status={status}")
    print(f"artifacts={len(summaries)}")
    if args.fail_on_airgap and status == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
