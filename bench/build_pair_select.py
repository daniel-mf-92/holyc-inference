#!/usr/bin/env python3
"""Select comparable baseline/candidate QEMU benchmark artifact pairs.

This host-side tool reads benchmark index JSON/CSV artifacts and emits the
latest two comparable QEMU prompt benchmark reports per config. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class IndexedArtifact:
    source: str
    artifact_type: str
    status: str
    generated_at: str
    commit: str
    profile: str
    model: str
    quantization: str
    prompt_suite_sha256: str
    command_sha256: str
    launch_plan_sha256: str
    environment_sha256: str
    measured_runs: int
    median_tok_per_s: float | None
    wall_tok_per_s_median: float | None
    max_memory_bytes: int | None

    @property
    def config_key(self) -> str:
        return "/".join(
            (
                self.profile or "-",
                self.model or "-",
                self.quantization or "-",
                self.prompt_suite_sha256 or "no-suite",
                self.command_sha256 or "no-command",
                self.launch_plan_sha256 or "no-launch-plan",
                self.environment_sha256 or "no-environment",
            )
        )


@dataclass(frozen=True)
class BuildPair:
    key: str
    baseline_build: str
    candidate_build: str
    baseline_source: str
    candidate_source: str
    baseline_commit: str
    candidate_commit: str
    baseline_generated_at: str
    candidate_generated_at: str
    baseline_measured_runs: int
    candidate_measured_runs: int
    baseline_median_tok_per_s: float | None
    candidate_median_tok_per_s: float | None
    median_tok_per_s_delta_pct: float | None
    baseline_wall_tok_per_s_median: float | None
    candidate_wall_tok_per_s_median: float | None
    wall_tok_per_s_delta_pct: float | None
    baseline_max_memory_bytes: int | None
    candidate_max_memory_bytes: int | None
    max_memory_delta_pct: float | None
    build_compare_args: list[str]


@dataclass(frozen=True)
class Finding:
    key: str
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def text_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value) if value not in (None, "") else ""


def percent_delta(baseline: float | int | None, candidate: float | int | None) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    return round((float(candidate) - float(baseline)) * 100.0 / float(baseline), 6)


def parse_artifact(row: dict[str, Any]) -> IndexedArtifact:
    return IndexedArtifact(
        source=text_value(row, "source"),
        artifact_type=text_value(row, "artifact_type"),
        status=text_value(row, "status"),
        generated_at=text_value(row, "generated_at"),
        commit=text_value(row, "commit"),
        profile=text_value(row, "profile"),
        model=text_value(row, "model"),
        quantization=text_value(row, "quantization"),
        prompt_suite_sha256=text_value(row, "prompt_suite_sha256"),
        command_sha256=text_value(row, "command_sha256"),
        launch_plan_sha256=text_value(row, "launch_plan_sha256"),
        environment_sha256=text_value(row, "environment_sha256"),
        measured_runs=parse_int(row.get("measured_runs")) or 0,
        median_tok_per_s=parse_float(row.get("median_tok_per_s")),
        wall_tok_per_s_median=parse_float(row.get("wall_tok_per_s_median")),
        max_memory_bytes=parse_int(row.get("max_memory_bytes")),
    )


def flatten_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    for key in ("artifacts", "rows", "results", "latest_comparable"):
        rows = payload.get(key)
        if isinstance(rows, list):
            for item in rows:
                if isinstance(item, dict):
                    yield item
            return
    yield payload


def load_artifacts(path: Path) -> list[IndexedArtifact]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]]
    if suffix == ".json":
        rows = list(flatten_json_payload(json.loads(path.read_text(encoding="utf-8"))))
    elif suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.extend(flatten_json_payload(json.loads(stripped)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"{path}: unsupported input suffix {path.suffix}")
    return [parse_artifact(row) for row in rows]


def artifact_sort_key(artifact: IndexedArtifact) -> tuple[str, str, str]:
    return (artifact.generated_at, artifact.commit, artifact.source)


def eligible_artifacts(
    artifacts: Iterable[IndexedArtifact],
    *,
    min_measured_runs: int,
    require_wall_tok_per_s: bool,
) -> tuple[list[IndexedArtifact], list[Finding]]:
    eligible: list[IndexedArtifact] = []
    findings: list[Finding] = []
    for artifact in artifacts:
        key = artifact.config_key
        if artifact.artifact_type and artifact.artifact_type != "qemu_prompt":
            findings.append(Finding(key, "info", "skipped_type", f"skipped artifact_type={artifact.artifact_type!r}"))
            continue
        if artifact.status and artifact.status.lower() not in {"pass", "ok", "success"}:
            findings.append(Finding(key, "warning", "skipped_status", f"skipped status={artifact.status!r}"))
            continue
        if not artifact.source:
            findings.append(Finding(key, "error", "missing_source", "source path is absent"))
            continue
        if not artifact.generated_at:
            findings.append(Finding(key, "error", "missing_generated_at", "generated_at is absent"))
            continue
        if not artifact.commit:
            findings.append(Finding(key, "error", "missing_commit", "commit is absent"))
            continue
        if artifact.measured_runs < min_measured_runs:
            findings.append(
                Finding(
                    key,
                    "warning",
                    "insufficient_runs",
                    f"measured_runs={artifact.measured_runs} below required {min_measured_runs}",
                )
            )
            continue
        if artifact.median_tok_per_s is None:
            findings.append(Finding(key, "error", "missing_tok_per_s", "median_tok_per_s is absent"))
            continue
        if require_wall_tok_per_s and artifact.wall_tok_per_s_median is None:
            findings.append(Finding(key, "error", "missing_wall_tok_per_s", "wall_tok_per_s_median is absent"))
            continue
        eligible.append(artifact)
    return eligible, findings


def build_label(prefix: str, artifact: IndexedArtifact) -> str:
    short_commit = artifact.commit[:12] if artifact.commit else "unknown"
    date = artifact.generated_at.replace(":", "").replace("-", "")
    date = date.split("T", 1)[0] if date else "undated"
    return f"{prefix}-{date}-{short_commit}"


def select_pairs(
    artifacts: list[IndexedArtifact],
    *,
    min_measured_runs: int = 1,
    require_wall_tok_per_s: bool = False,
    allow_same_commit: bool = False,
) -> tuple[list[BuildPair], list[Finding]]:
    eligible, findings = eligible_artifacts(
        artifacts,
        min_measured_runs=min_measured_runs,
        require_wall_tok_per_s=require_wall_tok_per_s,
    )
    grouped: dict[str, list[IndexedArtifact]] = {}
    for artifact in eligible:
        grouped.setdefault(artifact.config_key, []).append(artifact)

    pairs: list[BuildPair] = []
    for key, rows in sorted(grouped.items()):
        latest_by_commit: list[IndexedArtifact] = []
        seen_commits: set[str] = set()
        for artifact in sorted(rows, key=artifact_sort_key, reverse=True):
            if allow_same_commit or artifact.commit not in seen_commits:
                latest_by_commit.append(artifact)
                seen_commits.add(artifact.commit)
            if len(latest_by_commit) == 2:
                break
        if len(latest_by_commit) < 2:
            findings.append(Finding(key, "warning", "insufficient_history", "fewer than two comparable artifacts"))
            continue
        candidate, baseline = latest_by_commit[0], latest_by_commit[1]
        baseline_build = build_label("base", baseline)
        candidate_build = build_label("head", candidate)
        pairs.append(
            BuildPair(
                key=key,
                baseline_build=baseline_build,
                candidate_build=candidate_build,
                baseline_source=baseline.source,
                candidate_source=candidate.source,
                baseline_commit=baseline.commit,
                candidate_commit=candidate.commit,
                baseline_generated_at=baseline.generated_at,
                candidate_generated_at=candidate.generated_at,
                baseline_measured_runs=baseline.measured_runs,
                candidate_measured_runs=candidate.measured_runs,
                baseline_median_tok_per_s=baseline.median_tok_per_s,
                candidate_median_tok_per_s=candidate.median_tok_per_s,
                median_tok_per_s_delta_pct=percent_delta(baseline.median_tok_per_s, candidate.median_tok_per_s),
                baseline_wall_tok_per_s_median=baseline.wall_tok_per_s_median,
                candidate_wall_tok_per_s_median=candidate.wall_tok_per_s_median,
                wall_tok_per_s_delta_pct=percent_delta(baseline.wall_tok_per_s_median, candidate.wall_tok_per_s_median),
                baseline_max_memory_bytes=baseline.max_memory_bytes,
                candidate_max_memory_bytes=candidate.max_memory_bytes,
                max_memory_delta_pct=percent_delta(baseline.max_memory_bytes, candidate.max_memory_bytes),
                build_compare_args=[
                    "--input",
                    f"{baseline_build}={baseline.source}",
                    "--input",
                    f"{candidate_build}={candidate.source}",
                ],
            )
        )
    return pairs, findings


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[BuildPair]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(BuildPair.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["build_compare_args"] = shlex.join(payload["build_compare_args"])
            writer.writerow(payload)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Build Pair Select",
        "",
        f"Status: {payload['status']}",
        f"Pairs: {len(payload['pairs'])}",
        f"Findings: {len(payload['findings'])}",
        "",
        "| key | baseline | candidate | tok/s delta % | wall tok/s delta % | memory delta % |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for pair in payload["pairs"]:
        lines.append(
            "| {key} | {baseline_build} | {candidate_build} | {median_tok_per_s_delta_pct} | "
            "{wall_tok_per_s_delta_pct} | {max_memory_delta_pct} |".format(**pair)
        )
    if payload["findings"]:
        lines.extend(["", "## Findings", "", "| severity | kind | key | detail |", "| --- | --- | --- | --- |"])
        for finding in payload["findings"]:
            lines.append(
                f"| {finding['severity']} | {finding['kind']} | {finding['key']} | {finding['detail']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    failures = [finding for finding in payload["findings"] if finding["severity"] == "error"]
    tests = max(1, len(payload["pairs"]) + len(failures))
    suite = ET.Element("testsuite", name="holyc_build_pair_select", tests=str(tests), failures=str(len(failures)))
    for pair in payload["pairs"]:
        ET.SubElement(suite, "testcase", name=pair["key"], classname="build_pair_select")
    for finding in failures:
        case = ET.SubElement(suite, "testcase", name=finding["kind"], classname="build_pair_select")
        failure = ET.SubElement(case, "failure", message=finding["detail"])
        failure.text = json.dumps(finding, sort_keys=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="bench result index JSON/JSONL/CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="build_pair_select_latest")
    parser.add_argument("--min-measured-runs", type=int, default=1)
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument("--require-wall-tok-per-s", action="store_true")
    parser.add_argument("--allow-same-commit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts: list[IndexedArtifact] = []
    for path in args.inputs:
        artifacts.extend(load_artifacts(path))

    pairs, findings = select_pairs(
        artifacts,
        min_measured_runs=args.min_measured_runs,
        require_wall_tok_per_s=args.require_wall_tok_per_s,
        allow_same_commit=args.allow_same_commit,
    )
    if len(pairs) < args.min_pairs:
        findings.append(
            Finding(
                key="",
                severity="error",
                kind="min_pairs",
                detail=f"selected {len(pairs)} pairs below required {args.min_pairs}",
            )
        )

    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "input_count": len(args.inputs),
        "artifact_count": len(artifacts),
        "eligible_pair_count": len(pairs),
        "min_measured_runs": args.min_measured_runs,
        "pairs": [asdict(pair) for pair in pairs],
        "findings": [asdict(finding) for finding in findings],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.output_stem}.json", payload)
    write_csv(args.output_dir / f"{args.output_stem}.csv", pairs)
    write_markdown(args.output_dir / f"{args.output_stem}.md", payload)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", payload)
    print(json.dumps({"status": status, "pairs": len(pairs), "findings": len(findings)}, sort_keys=True))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
