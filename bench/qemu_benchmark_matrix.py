#!/usr/bin/env python3
"""Create an air-gapped QEMU benchmark matrix plan without launching QEMU.

The matrix expands one prompt suite across one or more build images and writes a
replayable plan under bench/results. Every generated command is built through
qemu_prompt_bench.build_command(), which injects `-nic none` and rejects network
arguments before any artifact is written.
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
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_prompt_bench


@dataclass(frozen=True)
class MatrixBuild:
    build: str
    image: str
    qemu_bin: str
    profile: str
    model: str
    quantization: str
    command_sha256: str
    command_argc: int
    command_airgap_ok: bool
    command_airgap_violations: tuple[str, ...]
    prompt_count: int
    warmup: int
    repeat: int
    launch_count: int
    launch_plan_sha256: str
    command: list[str]


@dataclass(frozen=True)
class MatrixLaunch:
    build: str
    profile: str
    model: str
    quantization: str
    command_sha256: str
    launch_index: int
    phase: str
    prompt_index: int
    prompt_id: str
    prompt_sha256: str
    prompt_bytes: int
    expected_tokens: int | None
    iteration: int


@dataclass(frozen=True)
class MatrixFinding:
    severity: str
    kind: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("matrix root must be a JSON object")
    return payload


def string_value(row: dict[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def int_value(row: dict[str, Any], key: str, default: int) -> int:
    value = row.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer")
    if parsed < 0:
        raise ValueError(f"{key} must be non-negative")
    return parsed


def string_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON array of strings")
    return list(value)


def matrix_build_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    builds = payload.get("builds")
    if not isinstance(builds, list) or not builds:
        raise ValueError("matrix must contain a non-empty builds array")
    rows = []
    for index, row in enumerate(builds, 1):
        if not isinstance(row, dict):
            raise ValueError(f"builds[{index}] must be an object")
        rows.append(row)
    return rows


def build_matrix_report(matrix_path: Path, output_stem: str) -> dict[str, Any]:
    payload = load_matrix(matrix_path)
    prompts_path = Path(string_value(payload, "prompts"))
    if not prompts_path.is_absolute():
        prompts_path = (matrix_path.parent / prompts_path).resolve()
    cases = qemu_prompt_bench.load_prompt_cases(prompts_path)
    prompt_suite = qemu_prompt_bench.prompt_suite_metadata(prompts_path, cases)

    default_qemu_bin = string_value(payload, "qemu_bin", "qemu-system-x86_64")
    default_profile = string_value(payload, "profile", "benchmark-matrix")
    default_model = string_value(payload, "model", "unknown")
    default_quantization = string_value(payload, "quantization", "unknown")
    default_warmup = int_value(payload, "warmup", 1)
    default_repeat = int_value(payload, "repeat", 3)
    default_qemu_args = string_list(payload.get("qemu_args"), field="qemu_args")

    builds: list[MatrixBuild] = []
    launches: list[MatrixLaunch] = []
    findings: list[MatrixFinding] = []

    for row in matrix_build_rows(payload):
        build_name = string_value(row, "build")
        if not build_name:
            raise ValueError("build rows must include a non-empty build")
        image = Path(string_value(row, "image"))
        if not image.is_absolute():
            image = (matrix_path.parent / image).resolve()
        qemu_bin = string_value(row, "qemu_bin", default_qemu_bin)
        profile = string_value(row, "profile", default_profile)
        model = string_value(row, "model", default_model)
        quantization = string_value(row, "quantization", default_quantization)
        warmup = int_value(row, "warmup", default_warmup)
        repeat = int_value(row, "repeat", default_repeat)
        qemu_args = default_qemu_args + string_list(row.get("qemu_args"), field=f"{build_name}.qemu_args")

        command = qemu_prompt_bench.build_command(qemu_bin, image, qemu_args)
        command_airgap = qemu_prompt_bench.command_airgap_metadata(command)
        plan = qemu_prompt_bench.dry_run_launch_plan(cases, warmup=warmup, repeat=repeat)
        plan_sha256 = qemu_prompt_bench.launch_plan_hash(plan)
        command_sha256 = qemu_prompt_bench.command_hash(command)

        if not command_airgap["ok"]:
            findings.append(
                MatrixFinding(
                    "error",
                    "command_airgap",
                    build_name,
                    "; ".join(command_airgap["violations"]),
                )
            )

        builds.append(
            MatrixBuild(
                build=build_name,
                image=str(image),
                qemu_bin=qemu_bin,
                profile=profile,
                model=model,
                quantization=quantization,
                command_sha256=command_sha256,
                command_argc=len(command),
                command_airgap_ok=bool(command_airgap["ok"]),
                command_airgap_violations=tuple(str(item) for item in command_airgap["violations"]),
                prompt_count=len(cases),
                warmup=warmup,
                repeat=repeat,
                launch_count=len(plan),
                launch_plan_sha256=plan_sha256,
                command=command,
            )
        )
        for plan_row in plan:
            launches.append(
                MatrixLaunch(
                    build=build_name,
                    profile=profile,
                    model=model,
                    quantization=quantization,
                    command_sha256=command_sha256,
                    launch_index=int(plan_row["launch_index"]),
                    phase=str(plan_row["phase"]),
                    prompt_index=int(plan_row["prompt_index"]),
                    prompt_id=str(plan_row["prompt_id"]),
                    prompt_sha256=str(plan_row["prompt_sha256"]),
                    prompt_bytes=int(plan_row["prompt_bytes"]),
                    expected_tokens=plan_row["expected_tokens"],
                    iteration=int(plan_row["iteration"]),
                )
            )

    matrix_identity = {
        "matrix_source": str(matrix_path),
        "prompt_suite_sha256": prompt_suite["suite_sha256"],
        "builds": [
            {
                "build": build.build,
                "command_sha256": build.command_sha256,
                "launch_plan_sha256": build.launch_plan_sha256,
            }
            for build in builds
        ],
    }
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "matrix_sha256": json_sha256(matrix_identity),
        "matrix_source": str(matrix_path),
        "output_stem": output_stem,
        "prompt_suite": prompt_suite,
        "summary": {
            "builds": len(builds),
            "prompts": len(cases),
            "launches": len(launches),
            "airgap_ok_builds": sum(1 for build in builds if build.command_airgap_ok),
            "findings": len(findings),
        },
        "builds": [asdict(build) for build in builds],
        "launches": [asdict(launch) for launch in launches],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_builds_csv(path: Path, builds: list[dict[str, Any]]) -> None:
    fieldnames = [field for field in MatrixBuild.__dataclass_fields__ if field != "command"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for build in builds:
            row = dict(build)
            row.pop("command", None)
            row["command_airgap_violations"] = "; ".join(row["command_airgap_violations"])
            writer.writerow(row)


def write_launches_csv(path: Path, launches: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MatrixLaunch.__dataclass_fields__))
        writer.writeheader()
        for launch in launches:
            writer.writerow(launch)


def write_commands_jsonl(path: Path, builds: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for build in builds:
            handle.write(
                json.dumps(
                    {
                        "build": build["build"],
                        "command_sha256": build["command_sha256"],
                        "argv": build["command"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Benchmark Matrix",
        "",
        f"- Status: {report['status']}",
        f"- Builds: {summary['builds']}",
        f"- Prompts: {summary['prompts']}",
        f"- Launches: {summary['launches']}",
        f"- Air-gap OK builds: {summary['airgap_ok_builds']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Build | Profile | Model | Quantization | Launches | Air-gap | Command |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for build in report["builds"]:
        lines.append(
            "| {build} | {profile} | {model} | {quantization} | {launch_count} | {command_airgap_ok} | {command_sha256} |".format(
                **build
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
            "name": "holyc_qemu_benchmark_matrix",
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
        ET.SubElement(suite, "testcase", {"name": "qemu_benchmark_matrix"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def write_report(report: dict[str, Any], output_dir: Path, output_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / output_stem
    write_json(output_base.with_suffix(".json"), report)
    write_builds_csv(output_base.with_suffix(".csv"), report["builds"])
    write_launches_csv(output_base.with_name(f"{output_base.name}_launches.csv"), report["launches"])
    write_commands_jsonl(output_base.with_name(f"{output_base.name}_commands.jsonl"), report["builds"])
    write_markdown(output_base.with_suffix(".md"), report)
    write_junit(output_base.with_name(f"{output_base.name}_junit.xml"), report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="Local JSON benchmark matrix")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_benchmark_matrix_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_matrix_report(args.matrix, args.output_stem)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    write_report(report, args.output_dir, args.output_stem)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
