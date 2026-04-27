#!/usr/bin/env python3
"""Audit benchmark artifacts for explicit TempleOS guest air-gap settings.

This host-side tool scans JSON/JSONL benchmark reports for recorded QEMU
commands. Any command that looks like a QEMU launch must include `-nic none` or
`-nic=none`, and must not include legacy `-net` networking, `-netdev`, or known
virtual NIC devices.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


NETWORK_DEVICE_MARKERS = (
    "e1000",
    "eepro100",
    "i825",
    "ne2k",
    "pcnet",
    "rtl8139",
    "spapr-vlan",
    "sunhme",
    "sungem",
    "tulip",
    "usb-net",
    "virtio-net",
    "vmxnet",
    "xen_nic",
)
RESULT_KEYS = ("benchmarks", "warmups", "results", "runs", "rows")


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    reason: str
    command: list[str]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def is_network_device_arg(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in NETWORK_DEVICE_MARKERS)


def qemu_like(command: list[str]) -> bool:
    executable = Path(command[0]).name.lower() if command else ""
    return "qemu-system" in executable or any(arg.startswith("-drive") for arg in command)


def has_explicit_nic_none(command: list[str]) -> bool:
    for index, arg in enumerate(command):
        if arg == "-nic" and index + 1 < len(command) and command[index + 1] == "none":
            return True
        if arg == "-nic=none":
            return True
    return False


def command_violations(command: list[str]) -> list[str]:
    if not qemu_like(command):
        return []

    violations: list[str] = []
    if not has_explicit_nic_none(command):
        violations.append("missing explicit `-nic none`")

    index = 0
    while index < len(command):
        arg = command[index]
        next_arg = command[index + 1] if index + 1 < len(command) else ""

        if arg == "-nic":
            if next_arg != "none":
                violations.append(f"non-air-gapped `-nic {next_arg}`")
            index += 2
            continue
        if arg.startswith("-nic=") and arg != "-nic=none":
            violations.append(f"non-air-gapped `{arg}`")

        if arg == "-net":
            if next_arg == "none":
                violations.append("legacy `-net none` present; use `-nic none` in benchmark artifacts")
            else:
                violations.append(f"networking `-net {next_arg}`")
            index += 2
            continue
        if arg.startswith("-net="):
            if arg == "-net=none":
                violations.append("legacy `-net=none` present; use `-nic none` in benchmark artifacts")
            else:
                violations.append(f"networking `{arg}`")

        if arg == "-netdev" or arg.startswith("-netdev"):
            violations.append(f"network backend `{arg}`")
        if arg == "-device" and is_network_device_arg(next_arg):
            violations.append(f"network device `{next_arg}`")
        if arg.startswith("-device=") and is_network_device_arg(arg):
            violations.append(f"network device `{arg}`")

        index += 1

    return violations


def flatten_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(payload, dict):
        return

    yielded = False
    for key in RESULT_KEYS:
        nested = payload.get(key)
        if isinstance(nested, list):
            yielded = True
            for item in nested:
                if isinstance(item, dict):
                    merged = {k: v for k, v in payload.items() if k not in RESULT_KEYS}
                    merged.update(item)
                    yield merged

    if not yielded:
        yield payload


def load_json_records(path: Path) -> Iterable[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    yield from flatten_json_payload(payload)


def load_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            yield from flatten_json_payload(payload)


def iter_input_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in {".json", ".jsonl"}
            )
        elif path.suffix.lower() in {".json", ".jsonl"}:
            yield path


def normalize_command(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, str):
        return shlex.split(value)
    return None


def audit(paths: Iterable[Path]) -> tuple[int, list[Finding]]:
    commands_checked = 0
    findings: list[Finding] = []
    for path in sorted(iter_input_files(paths)):
        loader = load_jsonl_records if path.suffix.lower() == ".jsonl" else load_json_records
        for row_number, row in enumerate(loader(path), 1):
            command = normalize_command(row.get("command"))
            if command is None:
                continue
            violations = command_violations(command)
            if not qemu_like(command):
                continue
            commands_checked += 1
            for violation in violations:
                findings.append(
                    Finding(source=str(path), row=row_number, reason=violation, command=command)
                )
    return commands_checked, findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Benchmark artifact file or directory; defaults to bench/results",
    )
    parser.add_argument("--output", type=Path, default=Path("bench/results/airgap_audit_latest.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or [Path("bench/results")]
    commands_checked, findings = audit(inputs)
    report = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "commands_checked": commands_checked,
        "findings": [asdict(finding) for finding in findings],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_json={args.output}")
    print(f"status={report['status']}")
    print(f"commands_checked={commands_checked}")
    if findings:
        for finding in findings[:20]:
            print(f"{finding.source}:{finding.row}: {finding.reason}", file=sys.stderr)
        if len(findings) > 20:
            print(f"... {len(findings) - 20} more findings", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
