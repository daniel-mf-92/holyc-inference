#!/usr/bin/env python3
"""Compare two saved QEMU benchmark matrix plans without launching QEMU.

This host-side tool reads JSON artifacts produced by qemu_benchmark_matrix.py
and reports launch-plan drift before benchmark VM time is spent. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildDiff:
    build: str
    profile: str
    model: str
    quantization: str
    baseline_present: bool
    candidate_present: bool
    baseline_command_sha256: str
    candidate_command_sha256: str
    baseline_launch_plan_sha256: str
    candidate_launch_plan_sha256: str
    baseline_launch_count: int
    candidate_launch_count: int
    delta_launch_count: int
    status: str


@dataclass(frozen=True)
class LaunchDiff:
    build: str
    profile: str
    model: str
    quantization: str
    phase: str
    prompt_id: str
    prompt_sha256: str
    iteration: int
    baseline_count: int
    candidate_count: int
    delta: int
    status: str


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: matrix artifact root must be a JSON object")
    return payload


def object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def text_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def int_value(value: Any) -> int:
    if isinstance(value, bool) or value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        text_value(row, "build"),
        text_value(row, "profile"),
        text_value(row, "model"),
        text_value(row, "quantization"),
    )


def launch_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, int]:
    return (
        text_value(row, "build"),
        text_value(row, "profile"),
        text_value(row, "model"),
        text_value(row, "quantization"),
        text_value(row, "phase"),
        text_value(row, "prompt_id"),
        text_value(row, "prompt_sha256"),
        int_value(row.get("iteration")),
    )


def count_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, ...] | tuple[str, str, str, str, str, str, str, int], int]:
    counts: dict[tuple[str, ...] | tuple[str, str, str, str, str, str, str, int], int] = {}
    for row in rows:
        key = launch_key(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {build_key(row): row for row in rows}


def compare_matrix_plans(
    baseline_path: Path,
    candidate_path: Path,
    *,
    allow_command_hash_drift: bool = False,
    allow_launch_drift: bool = False,
) -> dict[str, Any]:
    baseline = load_json_object(baseline_path)
    candidate = load_json_object(candidate_path)

    baseline_builds = build_map(object_list(baseline.get("builds")))
    candidate_builds = build_map(object_list(candidate.get("builds")))
    baseline_launches = count_by_key(object_list(baseline.get("launches")))
    candidate_launches = count_by_key(object_list(candidate.get("launches")))
    findings: list[Finding] = []

    build_diffs: list[BuildDiff] = []
    for key in sorted(set(baseline_builds) | set(candidate_builds)):
        base = baseline_builds.get(key, {})
        cand = candidate_builds.get(key, {})
        base_present = bool(base)
        cand_present = bool(cand)
        base_command = text_value(base, "command_sha256")
        cand_command = text_value(cand, "command_sha256")
        base_plan = text_value(base, "launch_plan_sha256")
        cand_plan = text_value(cand, "launch_plan_sha256")
        base_count = int_value(base.get("launch_count"))
        cand_count = int_value(cand.get("launch_count"))
        status = "pass"

        if not base_present:
            status = "extra"
            findings.append(Finding("error", "extra_build", key[0], "candidate adds a matrix build absent from baseline"))
        elif not cand_present:
            status = "missing"
            findings.append(Finding("error", "missing_build", key[0], "candidate omits a matrix build present in baseline"))
        elif base_count != cand_count:
            status = "changed"
            findings.append(
                Finding(
                    "error",
                    "build_launch_count_drift",
                    key[0],
                    f"launch_count changed from {base_count} to {cand_count}",
                )
            )
        elif base_plan and cand_plan and base_plan != cand_plan:
            status = "changed"
            findings.append(Finding("error", "build_launch_plan_hash_drift", key[0], "launch_plan_sha256 changed"))

        if (
            base_present
            and cand_present
            and base_command
            and cand_command
            and base_command != cand_command
            and not allow_command_hash_drift
        ):
            status = "changed"
            findings.append(Finding("error", "build_command_hash_drift", key[0], "command_sha256 changed"))

        build_diffs.append(
            BuildDiff(
                build=key[0],
                profile=key[1],
                model=key[2],
                quantization=key[3],
                baseline_present=base_present,
                candidate_present=cand_present,
                baseline_command_sha256=base_command,
                candidate_command_sha256=cand_command,
                baseline_launch_plan_sha256=base_plan,
                candidate_launch_plan_sha256=cand_plan,
                baseline_launch_count=base_count,
                candidate_launch_count=cand_count,
                delta_launch_count=cand_count - base_count,
                status=status,
            )
        )

    launch_diffs: list[LaunchDiff] = []
    for key in sorted(set(baseline_launches) | set(candidate_launches)):
        base_count = baseline_launches.get(key, 0)
        cand_count = candidate_launches.get(key, 0)
        status = "pass" if base_count == cand_count else "changed"
        if status != "pass" and not allow_launch_drift:
            kind = "extra_launch" if base_count == 0 else "missing_launch" if cand_count == 0 else "launch_count_drift"
            findings.append(
                Finding(
                    "error",
                    kind,
                    key[5],
                    f"{key[0]}/{key[4]}/iter {key[7]} count changed from {base_count} to {cand_count}",
                )
            )
        launch_diffs.append(
            LaunchDiff(
                build=key[0],
                profile=key[1],
                model=key[2],
                quantization=key[3],
                phase=key[4],
                prompt_id=key[5],
                prompt_sha256=key[6],
                iteration=key[7],
                baseline_count=base_count,
                candidate_count=cand_count,
                delta=cand_count - base_count,
                status=status,
            )
        )

    if allow_launch_drift:
        findings = [finding for finding in findings if not finding.kind.endswith("_launch") and finding.kind != "launch_count_drift"]

    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "summary": {
            "baseline_builds": len(baseline_builds),
            "candidate_builds": len(candidate_builds),
            "build_diff_rows": len(build_diffs),
            "baseline_launch_keys": len(baseline_launches),
            "candidate_launch_keys": len(candidate_launches),
            "launch_diff_rows": len(launch_diffs),
            "changed_build_rows": sum(1 for row in build_diffs if row.status != "pass"),
            "changed_launch_rows": sum(1 for row in launch_diffs if row.status != "pass"),
            "findings": len(findings),
        },
        "build_diffs": [asdict(row) for row in build_diffs],
        "launch_diffs": [asdict(row) for row in launch_diffs],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Matrix Plan Diff",
        "",
        f"- Status: {report['status']}",
        f"- Baseline builds: {summary['baseline_builds']}",
        f"- Candidate builds: {summary['candidate_builds']}",
        f"- Changed build rows: {summary['changed_build_rows']}",
        f"- Changed launch rows: {summary['changed_launch_rows']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Build | Profile | Model | Quantization | Baseline launches | Candidate launches | Status |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in report["build_diffs"]:
        lines.append(
            "| {build} | {profile} | {model} | {quantization} | {baseline_launch_count} | {candidate_launch_count} | {status} |".format(
                **row
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append("- {kind}: {field} ({detail})".format(**finding))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_matrix_plan_diff",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding['kind']}:{finding['field']}"})
            failure = ET.SubElement(case, "failure", {"message": finding["detail"]})
            failure.text = finding["detail"]
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_matrix_plan_diff"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def write_report(report: dict[str, Any], output_dir: Path, output_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / output_stem
    write_json(output_base.with_suffix(".json"), report)
    write_csv(output_base.with_suffix(".csv"), report["build_diffs"], list(BuildDiff.__dataclass_fields__))
    write_csv(
        output_base.with_name(f"{output_base.name}_launches.csv"),
        report["launch_diffs"],
        list(LaunchDiff.__dataclass_fields__),
    )
    write_markdown(output_base.with_suffix(".md"), report)
    write_junit(output_base.with_name(f"{output_base.name}_junit.xml"), report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline qemu_benchmark_matrix JSON artifact")
    parser.add_argument("candidate", type=Path, help="Candidate qemu_benchmark_matrix JSON artifact")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_matrix_plan_diff_latest")
    parser.add_argument("--allow-command-hash-drift", action="store_true")
    parser.add_argument("--allow-launch-drift", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_matrix_plans(
            args.baseline,
            args.candidate,
            allow_command_hash_drift=args.allow_command_hash_drift,
            allow_launch_drift=args.allow_launch_drift,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 2
    write_report(report, args.output_dir, args.output_stem)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
