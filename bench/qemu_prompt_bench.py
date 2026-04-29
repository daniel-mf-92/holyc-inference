#!/usr/bin/env python3
"""Host-side QEMU prompt benchmark runner for the HolyC inference engine.

The runner launches QEMU once per prompt, captures serial output, extracts token
timing records, and writes normalized results under bench/results. Networking is
always disabled with `-nic none`, and conflicting network flags are rejected.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import resource
except ImportError:  # pragma: no cover - resource is Unix-only.
    resource = None


DEFAULT_TIMEOUT_SECONDS = 300.0
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
RESULT_LINE_RE = re.compile(r"(?:BENCH_RESULT|bench_result)\s*[:=]\s*(\{.*\})")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,\s]+)")
PROC_PIDTASKINFO = 4


class DarwinProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


try:
    DARWIN_LIBC = ctypes.CDLL("libc.dylib") if sys.platform == "darwin" else None
except OSError:
    DARWIN_LIBC = None


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    prompt: str
    expected_tokens: int | None = None


@dataclass(frozen=True)
class BenchRun:
    benchmark: str
    profile: str
    model: str
    quantization: str
    phase: str
    launch_index: int
    prompt: str
    prompt_sha256: str
    guest_prompt_sha256: str | None
    guest_prompt_sha256_match: bool | None
    prompt_bytes: int
    guest_prompt_bytes: int | None
    guest_prompt_bytes_match: bool | None
    iteration: int
    commit: str
    timestamp: str
    tokens: int | None
    expected_tokens: int | None
    expected_tokens_match: bool | None
    elapsed_us: int
    wall_elapsed_us: int
    timeout_seconds: float
    wall_timeout_pct: float | None
    host_overhead_us: int
    host_overhead_pct: float | None
    host_child_user_cpu_us: int | None
    host_child_system_cpu_us: int | None
    host_child_cpu_us: int | None
    host_child_cpu_pct: float | None
    host_child_tok_per_cpu_s: float | None
    host_child_peak_rss_bytes: int | None
    ttft_us: int | None
    tok_per_s: float | None
    wall_tok_per_s: float | None
    prompt_bytes_per_s: float | None
    wall_prompt_bytes_per_s: float | None
    tokens_per_prompt_byte: float | None
    us_per_token: float | None
    wall_us_per_token: float | None
    memory_bytes: int | None
    memory_bytes_per_token: float | None
    stdout_bytes: int
    stderr_bytes: int
    serial_output_bytes: int
    serial_output_lines: int
    returncode: int
    timed_out: bool
    exit_class: str
    failure_reason: str | None
    command: list[str]
    command_sha256: str
    command_airgap_ok: bool
    command_has_explicit_nic_none: bool
    command_has_legacy_net_none: bool
    command_airgap_violations: tuple[str, ...]
    stdout_tail: str
    stderr_tail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def command_hash(command: list[str]) -> str:
    encoded = json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_file_metadata(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": None,
        "mtime_ns": None,
        "sha256": None,
    }
    if not path.exists():
        return metadata
    stat = path.stat()
    metadata["size_bytes"] = stat.st_size
    metadata["mtime_ns"] = stat.st_mtime_ns
    if include_sha256:
        metadata["sha256"] = file_sha256(path)
    return metadata


def qemu_args_files_metadata(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [input_file_metadata(path, include_sha256=True) for path in paths]


def prompt_bytes(prompt: str) -> int:
    return len(prompt.encode("utf-8"))


def text_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def text_lines(text: str) -> int:
    return len(text.splitlines())


def prompt_suite_hash_parts(payload: list[dict[str, Any]]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_suite_hash(cases: list[PromptCase]) -> str:
    return prompt_suite_hash_parts(
        [
            {
                "prompt_id": case.prompt_id,
                "sha256": prompt_hash(case.prompt),
                "bytes": prompt_bytes(case.prompt),
            }
            for case in cases
        ]
    )


def prompt_suite_metadata(source: Path, cases: list[PromptCase]) -> dict[str, Any]:
    byte_counts = [prompt_bytes(case.prompt) for case in cases]
    return {
        "source": str(source),
        "prompt_count": len(cases),
        "suite_sha256": prompt_suite_hash(cases),
        "prompt_bytes_total": sum(byte_counts),
        "prompt_bytes_min": min(byte_counts) if byte_counts else None,
        "prompt_bytes_max": max(byte_counts) if byte_counts else None,
        "expected_token_prompts": sum(1 for case in cases if case.expected_tokens is not None),
        "expected_tokens_total": sum(case.expected_tokens or 0 for case in cases),
    }


def dry_run_launch_plan(cases: list[PromptCase], *, warmup: int, repeat: int) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    launch_index = 1
    for phase, iterations in (("warmup", warmup), ("measured", repeat)):
        for prompt_index, case in enumerate(cases, 1):
            for iteration in range(1, iterations + 1):
                plan.append(
                    {
                        "launch_index": launch_index,
                        "phase": phase,
                        "prompt_index": prompt_index,
                        "prompt_id": case.prompt_id,
                        "prompt_sha256": prompt_hash(case.prompt),
                        "prompt_bytes": prompt_bytes(case.prompt),
                        "expected_tokens": case.expected_tokens,
                        "iteration": iteration,
                    }
                )
                launch_index += 1
    return plan


def launch_plan_hash(plan: list[dict[str, Any]]) -> str:
    encoded = json.dumps(plan, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def launch_sequence_from_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "launch_index": row.get("launch_index"),
            "phase": row.get("phase"),
            "prompt_id": row.get("prompt_id"),
            "prompt_sha256": row.get("prompt_sha256"),
            "prompt_bytes": row.get("prompt_bytes"),
            "expected_tokens": row.get("expected_tokens"),
            "iteration": row.get("iteration"),
        }
        for row in sorted(plan, key=lambda item: int(item.get("launch_index") or 0))
    ]


def launch_sequence_from_runs(runs: list[BenchRun]) -> list[dict[str, Any]]:
    return [
        {
            "launch_index": run.launch_index,
            "phase": run.phase,
            "prompt_id": run.prompt,
            "prompt_sha256": run.prompt_sha256,
            "prompt_bytes": run.prompt_bytes,
            "expected_tokens": run.expected_tokens,
            "iteration": run.iteration,
        }
        for run in sorted(runs, key=lambda item: item.launch_index)
    ]


def launch_sequence_hash(sequence: list[dict[str, Any]]) -> str:
    encoded = json.dumps(sequence, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def launch_sequence_integrity(
    expected_sequence: list[dict[str, Any]],
    observed_sequence: list[dict[str, Any]],
) -> dict[str, Any]:
    matched = sum(1 for expected, observed in zip(expected_sequence, observed_sequence) if expected == observed)
    mismatched = sum(1 for expected, observed in zip(expected_sequence, observed_sequence) if expected != observed)
    missing = max(0, len(expected_sequence) - len(observed_sequence))
    extra = max(0, len(observed_sequence) - len(expected_sequence))
    expected_hash = launch_sequence_hash(expected_sequence)
    observed_hash = launch_sequence_hash(observed_sequence)
    return {
        "expected_launches": len(expected_sequence),
        "observed_launches": len(observed_sequence),
        "matched_launches": matched,
        "mismatched_launches": mismatched,
        "missing_launches": missing,
        "extra_launches": extra,
        "expected_launch_sequence_sha256": expected_hash,
        "observed_launch_sequence_sha256": observed_hash,
        "launch_sequence_match": expected_hash == observed_hash,
    }


def launch_sequence_findings(integrity: dict[str, Any]) -> list[dict[str, Any]]:
    if integrity.get("launch_sequence_match") is True:
        return []
    return [
        {
            "scope": "launch_sequence",
            "launch_index": "",
            "prompt": "",
            "iteration": "",
            "metric": "launch_sequence_match",
            "value": integrity.get("observed_launch_sequence_sha256"),
            "limit": integrity.get("expected_launch_sequence_sha256"),
        }
    ]


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


def qemu_version_line(qemu_bin: str) -> str | None:
    """Return a stable first-line QEMU version when the binary is discoverable."""
    qemu_path = shutil.which(qemu_bin)
    if qemu_path is None:
        return None
    if not Path(qemu_bin).name.startswith("qemu-"):
        return None
    try:
        result = subprocess.run(
            [qemu_path, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def host_environment(qemu_bin: str) -> dict[str, Any]:
    qemu_path = shutil.which(qemu_bin)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "qemu_bin": qemu_bin,
        "qemu_path": qemu_path,
        "qemu_version": qemu_version_line(qemu_bin),
    }


def load_prompt_cases(path: Path) -> list[PromptCase]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("prompts", payload) if isinstance(payload, dict) else payload
        return [prompt_case_from_row(row, index) for index, row in enumerate(rows)]
    if suffix == ".jsonl":
        cases = []
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                stripped = line.strip()
                if stripped:
                    cases.append(prompt_case_from_row(json.loads(stripped), index))
        return cases

    text = path.read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in text.split("\n---\n") if chunk.strip()]
    if not chunks and text.strip():
        chunks = [text.strip()]
    return [PromptCase(prompt_id=f"prompt-{index + 1}", prompt=chunk) for index, chunk in enumerate(chunks)]


def prompt_case_from_row(row: Any, index: int) -> PromptCase:
    if isinstance(row, str):
        return PromptCase(prompt_id=f"prompt-{index + 1}", prompt=row)
    if not isinstance(row, dict):
        raise ValueError(f"prompt row {index + 1} must be a string or object")

    prompt = row.get("prompt") or row.get("text") or row.get("input")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"prompt row {index + 1} is missing non-empty prompt text")
    prompt_id = str(row.get("id") or row.get("prompt_id") or f"prompt-{index + 1}")
    expected_tokens = parse_expected_tokens(row.get("expected_tokens", row.get("expected_generated_tokens")))
    return PromptCase(prompt_id=prompt_id, prompt=prompt, expected_tokens=expected_tokens)


def parse_expected_tokens(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("expected_tokens must be a non-negative integer")
    if parsed < 0:
        raise ValueError("expected_tokens must be a non-negative integer")
    return parsed


def is_network_device_arg(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in NETWORK_DEVICE_MARKERS)


def has_explicit_nic_none(args: list[str]) -> bool:
    for index, arg in enumerate(args):
        if arg == "-nic" and index + 1 < len(args) and args[index + 1] == "none":
            return True
        if arg == "-nic=none":
            return True
    return False


def has_legacy_net_none(args: list[str]) -> bool:
    for index, arg in enumerate(args):
        if arg == "-net" and index + 1 < len(args) and args[index + 1] == "none":
            return True
        if arg == "-net=none":
            return True
    return False


def command_airgap_violations(args: list[str]) -> list[str]:
    violations: list[str] = []
    if not has_explicit_nic_none(args):
        violations.append("missing explicit `-nic none`")

    index = 0
    while index < len(args):
        arg = args[index]
        next_arg = args[index + 1] if index + 1 < len(args) else ""

        if arg == "-nic":
            if next_arg != "none":
                violations.append(f"non-air-gapped `-nic {next_arg}`")
            index += 2
            continue
        if arg.startswith("-nic=") and arg != "-nic=none":
            violations.append(f"non-air-gapped `{arg}`")

        if arg == "-net":
            if next_arg != "none":
                violations.append(f"networking `-net {next_arg}`")
            index += 2
            continue
        if arg.startswith("-net=") and arg != "-net=none":
            violations.append(f"networking `{arg}`")

        if arg == "-netdev" or arg.startswith("-netdev"):
            violations.append(f"network backend `{arg}`")
        if arg == "-device" and is_network_device_arg(next_arg):
            violations.append(f"network device `{next_arg}`")
        if arg.startswith("-device=") and is_network_device_arg(arg):
            violations.append(f"network device `{arg}`")

        index += 1

    return violations


def command_airgap_metadata(args: list[str]) -> dict[str, Any]:
    violations = command_airgap_violations(args)
    return {
        "ok": not violations,
        "explicit_nic_none": has_explicit_nic_none(args),
        "legacy_net_none": has_legacy_net_none(args),
        "violations": violations,
    }


def reject_network_args(args: list[str]) -> None:
    index = 0
    while index < len(args):
        arg = args[index]
        next_arg = args[index + 1] if index + 1 < len(args) else ""

        if arg == "-nic":
            if next_arg != "none":
                raise ValueError("-nic arguments must be exactly `-nic none`")
            index += 2
            continue
        if arg.startswith("-nic=") and arg != "-nic=none":
            raise ValueError("-nic arguments must disable networking")

        if arg == "-net":
            if next_arg != "none":
                raise ValueError("-net arguments must be exactly `-net none`")
            index += 2
            continue
        if arg.startswith("-net=") and arg != "-net=none":
            raise ValueError("-net arguments must disable networking")

        if arg == "-netdev" or arg.startswith("-netdev"):
            raise ValueError("-netdev is not allowed for air-gapped benchmark runs")
        if arg == "-device" and is_network_device_arg(next_arg):
            raise ValueError(f"network device is not allowed: {next_arg}")
        if arg.startswith("-device=") and is_network_device_arg(arg):
            raise ValueError(f"network device is not allowed: {arg}")

        index += 1


def load_qemu_args_file(path: Path) -> list[str]:
    """Load extra QEMU arguments from a local JSON string array or text file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError(f"{path} must contain a JSON array of strings")
        return list(payload)
    return shlex.split(text, comments=True, posix=True)


def load_qemu_args_files(paths: Iterable[Path]) -> list[str]:
    args: list[str] = []
    for path in paths:
        args.extend(load_qemu_args_file(path))
    return args


def build_command(qemu_bin: str, image: Path, qemu_args: list[str]) -> list[str]:
    reject_network_args(qemu_args)
    command = [
        qemu_bin,
        "-nic",
        "none",
        "-serial",
        "stdio",
        "-display",
        "none",
        "-drive",
        f"file={image},format=raw,if=ide",
    ]
    command.extend(qemu_args)
    return command


def tail_text(text: str, limit: int = 4096) -> str:
    return text[-limit:]


def parse_bench_payload(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = RESULT_LINE_RE.search(stripped)
        candidate = match.group(1) if match else stripped
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(payload, dict):
                    return payload

    kv_payload: dict[str, Any] = {}
    for key, value in KV_RE.findall(text):
        kv_payload[key] = value
    return kv_payload


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_tokens(payload: dict[str, Any]) -> int | None:
    for key in ("tokens", "generated_tokens", "decode_tokens", "total_tokens"):
        parsed = parse_int(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def extract_guest_prompt_sha256(payload: dict[str, Any]) -> str | None:
    for key in ("prompt_sha256", "input_sha256", "prompt_hash", "input_hash"):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            return text
    return None


def extract_guest_prompt_bytes(payload: dict[str, Any]) -> int | None:
    for key in ("prompt_bytes", "input_bytes", "prompt_size_bytes", "input_size_bytes"):
        parsed = parse_int(payload.get(key))
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def extract_elapsed_us(payload: dict[str, Any], wall_elapsed_us: int) -> int:
    for key in ("elapsed_us", "elapsed_usec", "duration_us", "total_us"):
        parsed = parse_int(payload.get(key))
        if parsed is not None and parsed > 0:
            return parsed
    elapsed_ms = parse_float(payload.get("elapsed_ms") or payload.get("duration_ms"))
    if elapsed_ms is not None and elapsed_ms > 0:
        return max(1, int(elapsed_ms * 1000.0))
    elapsed_s = parse_float(payload.get("elapsed_s") or payload.get("duration_s"))
    if elapsed_s is not None and elapsed_s > 0:
        return max(1, int(elapsed_s * 1_000_000.0))
    return wall_elapsed_us


def extract_tok_per_s(payload: dict[str, Any], tokens: int | None, elapsed_us: int) -> float | None:
    direct = parse_float(payload.get("tok_per_s") or payload.get("tokens_per_second"))
    if direct is not None:
        return direct
    milli = parse_float(payload.get("tok_per_s_milli"))
    if milli is not None:
        return milli / 1000.0
    if tokens is not None and elapsed_us > 0:
        return tokens * 1_000_000.0 / elapsed_us
    return None


def derived_tok_per_s(tokens: int | None, elapsed_us: int) -> float | None:
    if tokens is None or elapsed_us <= 0:
        return None
    return tokens * 1_000_000.0 / elapsed_us


def derived_us_per_token(tokens: int | None, elapsed_us: int) -> float | None:
    if tokens is None or tokens <= 0 or elapsed_us <= 0:
        return None
    return elapsed_us / tokens


def derived_bytes_per_s(byte_count: int, elapsed_us: int) -> float | None:
    if byte_count <= 0 or elapsed_us <= 0:
        return None
    return byte_count * 1_000_000.0 / elapsed_us


def derived_tokens_per_prompt_byte(tokens: int | None, byte_count: int) -> float | None:
    if tokens is None or tokens < 0 or byte_count <= 0:
        return None
    return tokens / byte_count


def derived_memory_bytes_per_token(memory_bytes: int | None, tokens: int | None) -> float | None:
    if memory_bytes is None or memory_bytes < 0 or tokens is None or tokens <= 0:
        return None
    return memory_bytes / tokens


def host_overhead_pct(host_overhead_us: int, elapsed_us: int) -> float | None:
    if elapsed_us <= 0:
        return None
    return host_overhead_us * 100.0 / elapsed_us


def wall_timeout_pct(wall_elapsed_us: int, timeout_seconds: float) -> float | None:
    if timeout_seconds <= 0:
        return None
    return wall_elapsed_us * 100.0 / (timeout_seconds * 1_000_000.0)


def classify_exit(returncode: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "ok"
    if returncode == 127:
        return "launch_error"
    return "nonzero_exit"


def failure_reason(returncode: int, timed_out: bool) -> str | None:
    exit_class = classify_exit(returncode, timed_out)
    if exit_class == "ok":
        return None
    if exit_class == "timeout":
        return "timeout"
    return f"returncode_{returncode}"


def child_rusage() -> tuple[float, float] | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (usage.ru_utime, usage.ru_stime)


def child_cpu_delta_us(before: tuple[float, float] | None, after: tuple[float, float] | None) -> tuple[int, int] | None:
    if before is None or after is None:
        return None
    user_us = max(0, int((after[0] - before[0]) * 1_000_000.0))
    system_us = max(0, int((after[1] - before[1]) * 1_000_000.0))
    return (user_us, system_us)


def cpu_pct(cpu_us: int | None, wall_elapsed_us: int) -> float | None:
    if cpu_us is None or wall_elapsed_us <= 0:
        return None
    return cpu_us * 100.0 / wall_elapsed_us


def tokens_per_cpu_s(tokens: int | None, cpu_us: int | None) -> float | None:
    if tokens is None or tokens <= 0 or cpu_us is None or cpu_us <= 0:
        return None
    return tokens * 1_000_000.0 / cpu_us


def ok_run_pct(ok_runs: int, total_runs: int) -> float | None:
    if total_runs <= 0:
        return None
    return ok_runs * 100.0 / total_runs


def prompt_integrity_summary(runs: list[BenchRun]) -> dict[str, int]:
    sha_records = [run for run in runs if run.guest_prompt_sha256_match is not None]
    byte_records = [run for run in runs if run.guest_prompt_bytes_match is not None]
    return {
        "guest_prompt_sha256_records": len(sha_records),
        "guest_prompt_sha256_matches": sum(1 for run in sha_records if run.guest_prompt_sha256_match is True),
        "guest_prompt_sha256_mismatches": sum(1 for run in sha_records if run.guest_prompt_sha256_match is False),
        "guest_prompt_bytes_records": len(byte_records),
        "guest_prompt_bytes_matches": sum(1 for run in byte_records if run.guest_prompt_bytes_match is True),
        "guest_prompt_bytes_mismatches": sum(1 for run in byte_records if run.guest_prompt_bytes_match is False),
    }


def process_rss_bytes(pid: int) -> int | None:
    """Return current resident set size for a direct child process when available."""
    if sys.platform == "darwin":
        info = DarwinProcTaskInfo()
        proc_pidinfo = getattr(DARWIN_LIBC, "proc_pidinfo", None) if DARWIN_LIBC is not None else None
        if proc_pidinfo is not None:
            size = proc_pidinfo(
                ctypes.c_int(pid),
                ctypes.c_int(PROC_PIDTASKINFO),
                ctypes.c_uint64(0),
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if size == ctypes.sizeof(info):
                return int(info.pti_resident_size)

    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (OSError, ValueError):
        pass

    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    stripped = result.stdout.strip()
    if not stripped:
        return None
    try:
        return int(stripped.splitlines()[-1].strip()) * 1024
    except ValueError:
        return None


def run_command_with_rss_sample(
    command: list[str],
    *,
    prompt: str,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, str, str, bool, int | None]:
    """Run a child process while sampling direct-child RSS from the host."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    peak_rss = process_rss_bytes(process.pid)
    stop_sampling = threading.Event()
    lock = threading.Lock()

    def sample_rss() -> None:
        nonlocal peak_rss
        while not stop_sampling.wait(0.01):
            rss = process_rss_bytes(process.pid)
            if rss is None:
                if process.poll() is not None:
                    break
                continue
            with lock:
                peak_rss = rss if peak_rss is None else max(peak_rss, rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
        if exc.stdout:
            stdout = exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", errors="replace")
        if exc.stderr:
            stderr = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", errors="replace")
    finally:
        final_rss = process_rss_bytes(process.pid)
        with lock:
            if final_rss is not None:
                peak_rss = final_rss if peak_rss is None else max(peak_rss, final_rss)
        stop_sampling.set()
        sampler.join(timeout=1.0)

    with lock:
        sampled_peak_rss = peak_rss
    returncode = 124 if timed_out else (process.returncode if process.returncode is not None else 124)
    return returncode, stdout, stderr, timed_out, sampled_peak_rss


def extract_ttft_us(payload: dict[str, Any]) -> int | None:
    for key in ("ttft_us", "time_to_first_token_us", "first_token_us"):
        parsed = parse_int(payload.get(key))
        if parsed is not None and parsed >= 0:
            return parsed

    for key in ("ttft_ms", "time_to_first_token_ms", "first_token_ms"):
        parsed = parse_float(payload.get(key))
        if parsed is not None and parsed >= 0:
            return int(parsed * 1000.0)

    for key in ("ttft_s", "time_to_first_token_s", "first_token_s"):
        parsed = parse_float(payload.get(key))
        if parsed is not None and parsed >= 0:
            return int(parsed * 1_000_000.0)

    return None


def extract_memory_bytes(payload: dict[str, Any]) -> int | None:
    for key in ("memory_bytes", "max_rss_bytes", "rss_bytes", "peak_memory_bytes"):
        parsed = parse_int(payload.get(key))
        if parsed is not None and parsed >= 0:
            return parsed

    for key in ("memory_kib", "max_rss_kib", "rss_kib", "peak_memory_kib"):
        parsed = parse_int(payload.get(key))
        if parsed is not None and parsed >= 0:
            return parsed * 1024

    for key in ("memory_kb", "max_rss_kb", "rss_kb", "peak_memory_kb"):
        parsed = parse_int(payload.get(key))
        if parsed is not None and parsed >= 0:
            return parsed * 1000

    for key in ("memory_mib", "max_rss_mib", "rss_mib", "peak_memory_mib"):
        parsed = parse_float(payload.get(key))
        if parsed is not None and parsed >= 0:
            return int(parsed * 1024 * 1024)

    for key in ("memory_mb", "max_rss_mb", "rss_mb", "peak_memory_mb"):
        parsed = parse_float(payload.get(key))
        if parsed is not None and parsed >= 0:
            return int(parsed * 1000 * 1000)

    return None


def run_prompt(
    command: list[str],
    prompt_case: PromptCase,
    timeout: float,
    metadata: dict[str, str],
    *,
    phase: str = "measured",
    launch_index: int = 1,
    iteration: int = 1,
) -> BenchRun:
    started = time.monotonic_ns()
    child_cpu_before = child_rusage()
    env = os.environ.copy()
    env["HOLYC_BENCH_PROMPT"] = prompt_case.prompt
    env["HOLYC_BENCH_PROMPT_ID"] = prompt_case.prompt_id

    try:
        returncode, stdout, stderr, timed_out, host_child_peak_rss_bytes = run_command_with_rss_sample(
            command,
            prompt=prompt_case.prompt,
            env=env,
            timeout=timeout,
        )
    except OSError as exc:
        returncode = 127
        stdout = ""
        stderr = str(exc)
        timed_out = False
        host_child_peak_rss_bytes = None

    wall_elapsed_us = max(1, (time.monotonic_ns() - started) // 1000)
    child_cpu_after = child_rusage()
    child_cpu = child_cpu_delta_us(child_cpu_before, child_cpu_after)
    child_user_cpu_us = child_cpu[0] if child_cpu else None
    child_system_cpu_us = child_cpu[1] if child_cpu else None
    child_total_cpu_us = sum(child_cpu) if child_cpu else None
    payload = parse_bench_payload(stdout + "\n" + stderr)
    tokens = extract_tokens(payload)
    host_prompt_sha256 = prompt_hash(prompt_case.prompt)
    guest_prompt_sha256 = extract_guest_prompt_sha256(payload)
    guest_prompt_bytes = extract_guest_prompt_bytes(payload)
    elapsed_us = extract_elapsed_us(payload, wall_elapsed_us)
    host_overhead_us = wall_elapsed_us - elapsed_us
    ttft_us = extract_ttft_us(payload)
    tok_per_s = extract_tok_per_s(payload, tokens, elapsed_us)
    wall_tok_per_s = derived_tok_per_s(tokens, wall_elapsed_us)
    prompt_byte_count = prompt_bytes(prompt_case.prompt)
    prompt_bytes_per_s = derived_bytes_per_s(prompt_byte_count, elapsed_us)
    wall_prompt_bytes_per_s = derived_bytes_per_s(prompt_byte_count, wall_elapsed_us)
    tokens_per_prompt_byte = derived_tokens_per_prompt_byte(tokens, prompt_byte_count)
    us_per_token = derived_us_per_token(tokens, elapsed_us)
    wall_us_per_token = derived_us_per_token(tokens, wall_elapsed_us)
    memory_bytes = extract_memory_bytes(payload)
    memory_bytes_per_token = derived_memory_bytes_per_token(memory_bytes, tokens)
    stdout_bytes = text_bytes(stdout)
    stderr_bytes = text_bytes(stderr)

    return BenchRun(
        benchmark="qemu_prompt",
        profile=metadata["profile"],
        model=metadata["model"],
        quantization=metadata["quantization"],
        phase=phase,
        launch_index=launch_index,
        prompt=prompt_case.prompt_id,
        prompt_sha256=host_prompt_sha256,
        guest_prompt_sha256=guest_prompt_sha256,
        guest_prompt_sha256_match=None
        if guest_prompt_sha256 is None
        else guest_prompt_sha256 == host_prompt_sha256,
        prompt_bytes=prompt_byte_count,
        guest_prompt_bytes=guest_prompt_bytes,
        guest_prompt_bytes_match=None
        if guest_prompt_bytes is None
        else guest_prompt_bytes == prompt_byte_count,
        iteration=iteration,
        commit=metadata["commit"],
        timestamp=iso_now(),
        tokens=tokens,
        expected_tokens=prompt_case.expected_tokens,
        expected_tokens_match=None
        if prompt_case.expected_tokens is None or tokens is None
        else tokens == prompt_case.expected_tokens,
        elapsed_us=elapsed_us,
        wall_elapsed_us=wall_elapsed_us,
        timeout_seconds=timeout,
        wall_timeout_pct=wall_timeout_pct(wall_elapsed_us, timeout),
        host_overhead_us=host_overhead_us,
        host_overhead_pct=host_overhead_pct(host_overhead_us, elapsed_us),
        host_child_user_cpu_us=child_user_cpu_us,
        host_child_system_cpu_us=child_system_cpu_us,
        host_child_cpu_us=child_total_cpu_us,
        host_child_cpu_pct=cpu_pct(child_total_cpu_us, wall_elapsed_us),
        host_child_tok_per_cpu_s=tokens_per_cpu_s(tokens, child_total_cpu_us),
        host_child_peak_rss_bytes=host_child_peak_rss_bytes,
        ttft_us=ttft_us,
        tok_per_s=tok_per_s,
        wall_tok_per_s=wall_tok_per_s,
        prompt_bytes_per_s=prompt_bytes_per_s,
        wall_prompt_bytes_per_s=wall_prompt_bytes_per_s,
        tokens_per_prompt_byte=tokens_per_prompt_byte,
        us_per_token=us_per_token,
        wall_us_per_token=wall_us_per_token,
        memory_bytes=memory_bytes,
        memory_bytes_per_token=memory_bytes_per_token,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        serial_output_bytes=stdout_bytes + stderr_bytes,
        serial_output_lines=text_lines(stdout) + text_lines(stderr),
        returncode=returncode,
        timed_out=timed_out,
        exit_class=classify_exit(returncode, timed_out),
        failure_reason=failure_reason(returncode, timed_out),
        command=command,
        command_sha256=command_hash(command),
        command_airgap_ok=not command_airgap_violations(command),
        command_has_explicit_nic_none=has_explicit_nic_none(command),
        command_has_legacy_net_none=has_legacy_net_none(command),
        command_airgap_violations=tuple(command_airgap_violations(command)),
        stdout_tail=tail_text(stdout),
        stderr_tail=tail_text(stderr),
    )


def summarize_runs(runs: list[BenchRun]) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[BenchRun]] = {}
    for run in runs:
        by_prompt.setdefault(run.prompt, []).append(run)

    summaries: list[dict[str, Any]] = []
    for prompt_id, prompt_runs in sorted(by_prompt.items()):
        tok_values = [run.tok_per_s for run in prompt_runs if run.tok_per_s is not None]
        wall_tok_values = [run.wall_tok_per_s for run in prompt_runs if run.wall_tok_per_s is not None]
        prompt_bytes_per_s_values = [
            run.prompt_bytes_per_s for run in prompt_runs if run.prompt_bytes_per_s is not None
        ]
        wall_prompt_bytes_per_s_values = [
            run.wall_prompt_bytes_per_s
            for run in prompt_runs
            if run.wall_prompt_bytes_per_s is not None
        ]
        tokens_per_prompt_byte_values = [
            run.tokens_per_prompt_byte
            for run in prompt_runs
            if run.tokens_per_prompt_byte is not None
        ]
        us_per_token_values = [run.us_per_token for run in prompt_runs if run.us_per_token is not None]
        wall_us_per_token_values = [
            run.wall_us_per_token for run in prompt_runs if run.wall_us_per_token is not None
        ]
        elapsed_values = [run.elapsed_us for run in prompt_runs if run.elapsed_us > 0]
        host_overhead_values = [run.host_overhead_us for run in prompt_runs]
        host_overhead_pct_values = [
            run.host_overhead_pct for run in prompt_runs if run.host_overhead_pct is not None
        ]
        host_child_cpu_values = [
            run.host_child_cpu_us for run in prompt_runs if run.host_child_cpu_us is not None
        ]
        host_child_cpu_pct_values = [
            run.host_child_cpu_pct for run in prompt_runs if run.host_child_cpu_pct is not None
        ]
        host_child_tok_per_cpu_s_values = [
            run.host_child_tok_per_cpu_s
            for run in prompt_runs
            if run.host_child_tok_per_cpu_s is not None
        ]
        host_child_peak_rss_values = [
            run.host_child_peak_rss_bytes
            for run in prompt_runs
            if run.host_child_peak_rss_bytes is not None
        ]
        ttft_values = [run.ttft_us for run in prompt_runs if run.ttft_us is not None]
        memory_values = [run.memory_bytes for run in prompt_runs if run.memory_bytes is not None]
        memory_per_token_values = [
            run.memory_bytes_per_token
            for run in prompt_runs
            if run.memory_bytes_per_token is not None
        ]
        serial_output_values = [run.serial_output_bytes for run in prompt_runs]
        serial_output_line_values = [run.serial_output_lines for run in prompt_runs]
        ok_runs = [run for run in prompt_runs if run.returncode == 0 and not run.timed_out]
        timed_out_runs = [run for run in prompt_runs if run.timed_out]
        nonzero_exit_runs = [run for run in prompt_runs if run.returncode != 0]
        summaries.append(
            {
                "prompt": prompt_id,
                "prompt_bytes": prompt_runs[0].prompt_bytes,
                "runs": len(prompt_runs),
                "ok_runs": len(ok_runs),
                "failed_runs": len(prompt_runs) - len(ok_runs),
                "ok_run_pct": ok_run_pct(len(ok_runs), len(prompt_runs)),
                "timed_out_runs": len(timed_out_runs),
                "nonzero_exit_runs": len(nonzero_exit_runs),
                "tokens_median": statistics.median(
                    [run.tokens for run in prompt_runs if run.tokens is not None]
                )
                if any(run.tokens is not None for run in prompt_runs)
                else None,
                "elapsed_us_median": statistics.median(elapsed_values) if elapsed_values else None,
                "host_overhead_us_median": statistics.median(host_overhead_values)
                if host_overhead_values
                else None,
                "host_overhead_pct_median": statistics.median(host_overhead_pct_values)
                if host_overhead_pct_values
                else None,
                "host_child_cpu_us_median": statistics.median(host_child_cpu_values)
                if host_child_cpu_values
                else None,
                "host_child_cpu_pct_median": statistics.median(host_child_cpu_pct_values)
                if host_child_cpu_pct_values
                else None,
                "host_child_tok_per_cpu_s_median": statistics.median(host_child_tok_per_cpu_s_values)
                if host_child_tok_per_cpu_s_values
                else None,
                "host_child_peak_rss_bytes_max": max(host_child_peak_rss_values)
                if host_child_peak_rss_values
                else None,
                "ttft_us_median": statistics.median(ttft_values) if ttft_values else None,
                "ttft_us_p95": percentile([float(value) for value in ttft_values], 95.0),
                "tok_per_s_min": min(tok_values) if tok_values else None,
                "tok_per_s_p05": percentile(tok_values, 5.0),
                "tok_per_s_median": statistics.median(tok_values) if tok_values else None,
                "tok_per_s_stdev": sample_stdev(tok_values),
                "tok_per_s_cv_pct": coefficient_of_variation_pct(tok_values),
                "tok_per_s_iqr_pct": interquartile_range_pct(tok_values),
                "tok_per_s_p05_p95_spread_pct": percentile_spread_pct(tok_values, 5.0, 95.0),
                "tok_per_s_max": max(tok_values) if tok_values else None,
                "wall_tok_per_s_p05": percentile(wall_tok_values, 5.0),
                "wall_tok_per_s_median": statistics.median(wall_tok_values) if wall_tok_values else None,
                "wall_tok_per_s_p95": percentile(wall_tok_values, 95.0),
                "wall_tok_per_s_iqr_pct": interquartile_range_pct(wall_tok_values),
                "wall_tok_per_s_p05_p95_spread_pct": percentile_spread_pct(
                    wall_tok_values, 5.0, 95.0
                ),
                "prompt_bytes_per_s_median": statistics.median(prompt_bytes_per_s_values)
                if prompt_bytes_per_s_values
                else None,
                "wall_prompt_bytes_per_s_median": statistics.median(wall_prompt_bytes_per_s_values)
                if wall_prompt_bytes_per_s_values
                else None,
                "tokens_per_prompt_byte_median": statistics.median(tokens_per_prompt_byte_values)
                if tokens_per_prompt_byte_values
                else None,
                "us_per_token_median": statistics.median(us_per_token_values)
                if us_per_token_values
                else None,
                "us_per_token_p95": percentile(us_per_token_values, 95.0),
                "wall_us_per_token_median": statistics.median(wall_us_per_token_values)
                if wall_us_per_token_values
                else None,
                "wall_us_per_token_p95": percentile(wall_us_per_token_values, 95.0),
                "memory_bytes_max": max(memory_values) if memory_values else None,
                "memory_bytes_per_token_median": statistics.median(memory_per_token_values)
                if memory_per_token_values
                else None,
                "memory_bytes_per_token_max": max(memory_per_token_values)
                if memory_per_token_values
                else None,
                "serial_output_bytes_total": sum(serial_output_values),
                "serial_output_bytes_max": max(serial_output_values) if serial_output_values else None,
                "serial_output_lines_total": sum(serial_output_line_values),
                "serial_output_lines_max": max(serial_output_line_values)
                if serial_output_line_values
                else None,
                **prompt_integrity_summary(prompt_runs),
            }
        )
    return summaries


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)

    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def sample_stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def coefficient_of_variation_pct(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    return statistics.stdev(values) * 100.0 / mean


def percentile_spread_pct(values: list[float], low_pct: float, high_pct: float) -> float | None:
    if not values:
        return None
    low = percentile(values, low_pct)
    high = percentile(values, high_pct)
    median = statistics.median(values)
    if low is None or high is None or median == 0:
        return None
    return (high - low) * 100.0 / median


def interquartile_range_pct(values: list[float]) -> float | None:
    if not values:
        return None
    lower = percentile(values, 25.0)
    upper = percentile(values, 75.0)
    median = statistics.median(values)
    if lower is None or upper is None or median == 0:
        return None
    return (upper - lower) * 100.0 / median


def variability_findings(
    suite: dict[str, Any],
    summaries: list[dict[str, Any]],
    *,
    max_suite_cv_pct: float | None = None,
    max_prompt_cv_pct: float | None = None,
    max_suite_iqr_pct: float | None = None,
    max_prompt_iqr_pct: float | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    suite_cv = parse_float(suite.get("tok_per_s_cv_pct"))
    if max_suite_cv_pct is not None and suite_cv is not None and suite_cv > max_suite_cv_pct:
        findings.append(
            {
                "scope": "suite",
                "prompt": "",
                "metric": "tok_per_s_cv_pct",
                "value": suite_cv,
                "limit": max_suite_cv_pct,
            }
        )
    suite_iqr = parse_float(suite.get("tok_per_s_iqr_pct"))
    if max_suite_iqr_pct is not None and suite_iqr is not None and suite_iqr > max_suite_iqr_pct:
        findings.append(
            {
                "scope": "suite",
                "prompt": "",
                "metric": "tok_per_s_iqr_pct",
                "value": suite_iqr,
                "limit": max_suite_iqr_pct,
            }
        )

    if max_prompt_cv_pct is not None:
        for summary in summaries:
            prompt_cv = parse_float(summary.get("tok_per_s_cv_pct"))
            if prompt_cv is None or prompt_cv <= max_prompt_cv_pct:
                continue
            findings.append(
                {
                    "scope": "prompt",
                    "prompt": str(summary.get("prompt", "")),
                    "metric": "tok_per_s_cv_pct",
                    "value": prompt_cv,
                    "limit": max_prompt_cv_pct,
                }
            )
    if max_prompt_iqr_pct is not None:
        for summary in summaries:
            prompt_iqr = parse_float(summary.get("tok_per_s_iqr_pct"))
            if prompt_iqr is None or prompt_iqr <= max_prompt_iqr_pct:
                continue
            findings.append(
                {
                    "scope": "prompt",
                    "prompt": str(summary.get("prompt", "")),
                    "metric": "tok_per_s_iqr_pct",
                    "value": prompt_iqr,
                    "limit": max_prompt_iqr_pct,
                }
            )
    return findings


def telemetry_findings(
    runs: list[BenchRun],
    suite: dict[str, Any],
    *,
    require_tokens: bool = False,
    require_tok_per_s: bool = False,
    require_memory: bool = False,
    require_ttft_us: bool = False,
    min_tokens: int | None = None,
    min_total_tokens: int | None = None,
    min_tok_per_s: float | None = None,
    min_wall_tok_per_s: float | None = None,
    max_memory_bytes: int | None = None,
    max_ttft_us: int | None = None,
    max_host_overhead_us: int | None = None,
    max_host_overhead_pct: float | None = None,
    max_wall_timeout_pct: float | None = None,
    min_host_child_tok_per_cpu_s: float | None = None,
    min_tokens_per_prompt_byte: float | None = None,
    require_host_child_rss: bool = False,
    max_host_child_rss_bytes: int | None = None,
    max_memory_bytes_per_token: float | None = None,
    max_serial_output_bytes: int | None = None,
    max_serial_output_lines: int | None = None,
    require_guest_prompt_sha256_match: bool = False,
    require_guest_prompt_bytes_match: bool = False,
    require_expected_tokens_match: bool = False,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if min_total_tokens is not None:
        total_tokens = parse_int(suite.get("total_tokens"))
        if total_tokens is None or total_tokens < min_total_tokens:
            findings.append(
                {
                    "scope": "suite",
                    "launch_index": "",
                    "prompt": "",
                    "iteration": "",
                    "metric": "total_tokens",
                    "value": total_tokens,
                    "limit": min_total_tokens,
                }
            )
    for run in runs:
        base = {
            "scope": "measured_run",
            "launch_index": run.launch_index,
            "prompt": run.prompt,
            "iteration": run.iteration,
        }
        if require_tokens and run.tokens is None:
            findings.append({**base, "metric": "tokens", "value": None, "limit": "present"})
        if min_tokens is not None and (run.tokens is None or run.tokens < min_tokens):
            findings.append({**base, "metric": "tokens", "value": run.tokens, "limit": min_tokens})
        if (
            require_expected_tokens_match
            and run.expected_tokens is not None
            and run.expected_tokens_match is not True
        ):
            findings.append(
                {
                    **base,
                    "metric": "expected_tokens_match",
                    "value": run.tokens,
                    "limit": run.expected_tokens,
                }
            )
        if require_tok_per_s and run.tok_per_s is None:
            findings.append({**base, "metric": "tok_per_s", "value": None, "limit": "present"})
        if min_tok_per_s is not None and (run.tok_per_s is None or run.tok_per_s < min_tok_per_s):
            findings.append({**base, "metric": "tok_per_s", "value": run.tok_per_s, "limit": min_tok_per_s})
        if min_wall_tok_per_s is not None and (
            run.wall_tok_per_s is None or run.wall_tok_per_s < min_wall_tok_per_s
        ):
            findings.append(
                {
                    **base,
                    "metric": "wall_tok_per_s",
                    "value": run.wall_tok_per_s,
                    "limit": min_wall_tok_per_s,
                }
            )
        if require_memory and run.memory_bytes is None:
            findings.append({**base, "metric": "memory_bytes", "value": None, "limit": "present"})
        if max_memory_bytes is not None and (
            run.memory_bytes is None or run.memory_bytes > max_memory_bytes
        ):
            findings.append(
                {
                    **base,
                    "metric": "memory_bytes",
                    "value": run.memory_bytes,
                    "limit": max_memory_bytes,
                }
            )
        if require_ttft_us and run.ttft_us is None:
            findings.append({**base, "metric": "ttft_us", "value": None, "limit": "present"})
        if max_ttft_us is not None and (run.ttft_us is None or run.ttft_us > max_ttft_us):
            findings.append({**base, "metric": "ttft_us", "value": run.ttft_us, "limit": max_ttft_us})
        if max_host_overhead_us is not None and run.host_overhead_us > max_host_overhead_us:
            findings.append(
                {
                    **base,
                    "metric": "host_overhead_us",
                    "value": run.host_overhead_us,
                    "limit": max_host_overhead_us,
                }
            )
        if max_host_overhead_pct is not None and (
            run.host_overhead_pct is None or run.host_overhead_pct > max_host_overhead_pct
        ):
            findings.append(
                {
                    **base,
                    "metric": "host_overhead_pct",
                    "value": run.host_overhead_pct,
                    "limit": max_host_overhead_pct,
                }
            )
        if max_wall_timeout_pct is not None and (
            run.wall_timeout_pct is None or run.wall_timeout_pct > max_wall_timeout_pct
        ):
            findings.append(
                {
                    **base,
                    "metric": "wall_timeout_pct",
                    "value": run.wall_timeout_pct,
                    "limit": max_wall_timeout_pct,
                }
            )
        if min_host_child_tok_per_cpu_s is not None and (
            run.host_child_tok_per_cpu_s is None
            or run.host_child_tok_per_cpu_s < min_host_child_tok_per_cpu_s
        ):
            findings.append(
                {
                    **base,
                    "metric": "host_child_tok_per_cpu_s",
                    "value": run.host_child_tok_per_cpu_s,
                    "limit": min_host_child_tok_per_cpu_s,
                }
            )
        if min_tokens_per_prompt_byte is not None and (
            run.tokens_per_prompt_byte is None
            or run.tokens_per_prompt_byte < min_tokens_per_prompt_byte
        ):
            findings.append(
                {
                    **base,
                    "metric": "tokens_per_prompt_byte",
                    "value": run.tokens_per_prompt_byte,
                    "limit": min_tokens_per_prompt_byte,
                }
            )
        if require_host_child_rss and run.host_child_peak_rss_bytes is None:
            findings.append(
                {**base, "metric": "host_child_peak_rss_bytes", "value": None, "limit": "present"}
            )
        if max_host_child_rss_bytes is not None and (
            run.host_child_peak_rss_bytes is None
            or run.host_child_peak_rss_bytes > max_host_child_rss_bytes
        ):
            findings.append(
                {
                    **base,
                    "metric": "host_child_peak_rss_bytes",
                    "value": run.host_child_peak_rss_bytes,
                    "limit": max_host_child_rss_bytes,
                }
            )
        if max_memory_bytes_per_token is not None and (
            run.memory_bytes_per_token is None
            or run.memory_bytes_per_token > max_memory_bytes_per_token
        ):
            findings.append(
                {
                    **base,
                    "metric": "memory_bytes_per_token",
                    "value": run.memory_bytes_per_token,
                    "limit": max_memory_bytes_per_token,
                }
            )
        if max_serial_output_bytes is not None and run.serial_output_bytes > max_serial_output_bytes:
            findings.append(
                {
                    **base,
                    "metric": "serial_output_bytes",
                    "value": run.serial_output_bytes,
                    "limit": max_serial_output_bytes,
                }
            )
        if max_serial_output_lines is not None and run.serial_output_lines > max_serial_output_lines:
            findings.append(
                {
                    **base,
                    "metric": "serial_output_lines",
                    "value": run.serial_output_lines,
                    "limit": max_serial_output_lines,
                }
            )
        if require_guest_prompt_sha256_match and run.guest_prompt_sha256_match is not True:
            findings.append(
                {
                    **base,
                    "metric": "guest_prompt_sha256_match",
                    "value": run.guest_prompt_sha256,
                    "limit": run.prompt_sha256,
                }
            )
        if require_guest_prompt_bytes_match and run.guest_prompt_bytes_match is not True:
            findings.append(
                {
                    **base,
                    "metric": "guest_prompt_bytes_match",
                    "value": run.guest_prompt_bytes,
                    "limit": run.prompt_bytes,
                }
            )
    return findings


def suite_summary(runs: list[BenchRun]) -> dict[str, Any]:
    tok_values = [run.tok_per_s for run in runs if run.tok_per_s is not None]
    wall_tok_values = [run.wall_tok_per_s for run in runs if run.wall_tok_per_s is not None]
    prompt_bytes_per_s_values = [
        run.prompt_bytes_per_s for run in runs if run.prompt_bytes_per_s is not None
    ]
    wall_prompt_bytes_per_s_values = [
        run.wall_prompt_bytes_per_s for run in runs if run.wall_prompt_bytes_per_s is not None
    ]
    tokens_per_prompt_byte_values = [
        run.tokens_per_prompt_byte for run in runs if run.tokens_per_prompt_byte is not None
    ]
    us_per_token_values = [run.us_per_token for run in runs if run.us_per_token is not None]
    wall_us_per_token_values = [run.wall_us_per_token for run in runs if run.wall_us_per_token is not None]
    elapsed_values = [run.elapsed_us for run in runs if run.elapsed_us > 0]
    host_overhead_values = [run.host_overhead_us for run in runs]
    host_overhead_pct_values = [run.host_overhead_pct for run in runs if run.host_overhead_pct is not None]
    host_child_cpu_values = [run.host_child_cpu_us for run in runs if run.host_child_cpu_us is not None]
    host_child_cpu_pct_values = [run.host_child_cpu_pct for run in runs if run.host_child_cpu_pct is not None]
    host_child_tok_per_cpu_s_values = [
        run.host_child_tok_per_cpu_s for run in runs if run.host_child_tok_per_cpu_s is not None
    ]
    host_child_peak_rss_values = [
        run.host_child_peak_rss_bytes for run in runs if run.host_child_peak_rss_bytes is not None
    ]
    ttft_values = [run.ttft_us for run in runs if run.ttft_us is not None]
    memory_values = [run.memory_bytes for run in runs if run.memory_bytes is not None]
    memory_per_token_values = [
        run.memory_bytes_per_token for run in runs if run.memory_bytes_per_token is not None
    ]
    serial_output_values = [run.serial_output_bytes for run in runs]
    serial_output_line_values = [run.serial_output_lines for run in runs]
    token_values = [run.tokens for run in runs if run.tokens is not None]
    prompts = sorted({run.prompt for run in runs})
    ok_runs = [run for run in runs if run.returncode == 0 and not run.timed_out]
    timed_out_runs = [run for run in runs if run.timed_out]
    nonzero_exit_runs = [run for run in runs if run.returncode != 0]
    prompt_byte_values = [run.prompt_bytes for run in runs]

    return {
        "prompts": len(prompts),
        "runs": len(runs),
        "ok_runs": len(ok_runs),
        "failed_runs": len(runs) - len(ok_runs),
        "ok_run_pct": ok_run_pct(len(ok_runs), len(runs)),
        "timed_out_runs": len(timed_out_runs),
        "nonzero_exit_runs": len(nonzero_exit_runs),
        "measured_prompt_bytes_total": sum(prompt_byte_values) if prompt_byte_values else None,
        "prompt_bytes_min": min(prompt_byte_values) if prompt_byte_values else None,
        "prompt_bytes_max": max(prompt_byte_values) if prompt_byte_values else None,
        "total_tokens": sum(token_values) if token_values else None,
        "total_elapsed_us": sum(elapsed_values) if elapsed_values else None,
        "host_overhead_us_median": statistics.median(host_overhead_values) if host_overhead_values else None,
        "host_overhead_pct_median": statistics.median(host_overhead_pct_values)
        if host_overhead_pct_values
        else None,
        "host_child_cpu_us_median": statistics.median(host_child_cpu_values) if host_child_cpu_values else None,
        "host_child_cpu_pct_median": statistics.median(host_child_cpu_pct_values)
        if host_child_cpu_pct_values
        else None,
        "host_child_tok_per_cpu_s_median": statistics.median(host_child_tok_per_cpu_s_values)
        if host_child_tok_per_cpu_s_values
        else None,
        "host_child_peak_rss_bytes_max": max(host_child_peak_rss_values)
        if host_child_peak_rss_values
        else None,
        "ttft_us_median": statistics.median(ttft_values) if ttft_values else None,
        "ttft_us_p95": percentile([float(value) for value in ttft_values], 95.0),
        "tok_per_s_min": min(tok_values) if tok_values else None,
        "tok_per_s_p05": percentile(tok_values, 5.0),
        "tok_per_s_median": statistics.median(tok_values) if tok_values else None,
        "tok_per_s_stdev": sample_stdev(tok_values),
        "tok_per_s_cv_pct": coefficient_of_variation_pct(tok_values),
        "tok_per_s_iqr_pct": interquartile_range_pct(tok_values),
        "tok_per_s_p05_p95_spread_pct": percentile_spread_pct(tok_values, 5.0, 95.0),
        "tok_per_s_p95": percentile(tok_values, 95.0),
        "tok_per_s_max": max(tok_values) if tok_values else None,
        "wall_tok_per_s_p05": percentile(wall_tok_values, 5.0),
        "wall_tok_per_s_median": statistics.median(wall_tok_values) if wall_tok_values else None,
        "wall_tok_per_s_p95": percentile(wall_tok_values, 95.0),
        "wall_tok_per_s_iqr_pct": interquartile_range_pct(wall_tok_values),
        "wall_tok_per_s_p05_p95_spread_pct": percentile_spread_pct(wall_tok_values, 5.0, 95.0),
        "prompt_bytes_per_s_median": statistics.median(prompt_bytes_per_s_values)
        if prompt_bytes_per_s_values
        else None,
        "wall_prompt_bytes_per_s_median": statistics.median(wall_prompt_bytes_per_s_values)
        if wall_prompt_bytes_per_s_values
        else None,
        "tokens_per_prompt_byte_median": statistics.median(tokens_per_prompt_byte_values)
        if tokens_per_prompt_byte_values
        else None,
        "us_per_token_median": statistics.median(us_per_token_values) if us_per_token_values else None,
        "us_per_token_p95": percentile(us_per_token_values, 95.0),
        "wall_us_per_token_median": statistics.median(wall_us_per_token_values)
        if wall_us_per_token_values
        else None,
        "wall_us_per_token_p95": percentile(wall_us_per_token_values, 95.0),
        "elapsed_us_p95": percentile([float(value) for value in elapsed_values], 95.0),
        "memory_bytes_max": max(memory_values) if memory_values else None,
        "memory_bytes_per_token_median": statistics.median(memory_per_token_values)
        if memory_per_token_values
        else None,
        "memory_bytes_per_token_max": max(memory_per_token_values)
        if memory_per_token_values
        else None,
        "serial_output_bytes_total": sum(serial_output_values),
        "serial_output_bytes_max": max(serial_output_values) if serial_output_values else None,
        "serial_output_lines_total": sum(serial_output_line_values),
        "serial_output_lines_max": max(serial_output_line_values)
        if serial_output_line_values
        else None,
        **prompt_integrity_summary(runs),
    }


def phase_summaries(warmups: list[BenchRun], runs: list[BenchRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, phase_runs in (
        ("warmup", warmups),
        ("measured", runs),
        ("all", warmups + runs),
    ):
        summary = suite_summary(phase_runs)
        rows.append(
            {
                "phase": phase,
                "launches": len(phase_runs),
                **summary,
            }
        )
    return rows


def report_status(all_runs: list[BenchRun], findings: list[dict[str, Any]]) -> str:
    runs_ok = all(run.returncode == 0 and not run.timed_out for run in all_runs)
    return "pass" if runs_ok and not findings else "fail"


def markdown_report(report: dict[str, Any]) -> str:
    image = report.get("image") or {}
    qemu_args_files = report.get("qemu_args_files") if isinstance(report.get("qemu_args_files"), list) else []
    lines = [
        "# QEMU Prompt Benchmark",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Prompt suite: {report.get('prompt_suite', {}).get('suite_sha256', '-')}",
        f"Command SHA256: {report.get('command_sha256', '-')}",
        f"Command air-gap OK: {format_summary_value((report.get('command_airgap') or {}).get('ok'))}",
        f"Launch plan SHA256: {report.get('launch_plan_sha256', '-')}",
        f"Expected launch sequence SHA256: {report.get('expected_launch_sequence_sha256', '-')}",
        f"Observed launch sequence SHA256: {report.get('observed_launch_sequence_sha256', '-')}",
        f"Launch budget: {format_summary_value(report.get('max_launches'))}",
        f"Prompt count floor: {format_summary_value(report.get('min_prompt_count'))}",
        f"Total launches: {format_summary_value(report.get('planned_total_launches'))}",
        f"Warmup runs: {len(report['warmups'])}",
        f"Runs: {len(report['benchmarks'])}",
        "",
    ]
    if image or qemu_args_files:
        lines.extend(
            [
                "## Inputs",
                "",
                "| Image | Exists | Size bytes | SHA256 | QEMU args files |",
                "| --- | --- | ---: | --- | ---: |",
                "| {path} | {exists} | {size_bytes} | {sha256} | {args_files} |".format(
                    path=format_summary_value(image.get("path")),
                    exists=format_summary_value(image.get("exists")),
                    size_bytes=format_summary_value(image.get("size_bytes")),
                    sha256=format_summary_value(image.get("sha256")),
                    args_files=len(qemu_args_files),
                ),
                "",
            ]
        )
    suite = report.get("suite_summary") or {}
    if suite:
        lines.extend(
            [
                "## Suite Summary",
                "",
                "| Prompts | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| {prompts} | {runs} | {ok_runs} | {failed_runs} | {ok_run_pct} | {timed_out_runs} | {nonzero_exit_runs} | {measured_prompt_bytes_total} | {total_tokens} | {total_elapsed_us} | "
                "{host_overhead_us_median} | {host_overhead_pct_median} | {host_child_cpu_us_median} | {host_child_cpu_pct_median} | {host_child_tok_per_cpu_s_median} | {host_child_peak_rss_bytes_max} | {ttft_us_median} | {ttft_us_p95} | {tok_per_s_p05} | {tok_per_s_median} | {tok_per_s_p95} | "
                "{wall_tok_per_s_p05} | {wall_tok_per_s_median} | {wall_tok_per_s_p95} | {prompt_bytes_per_s_median} | {wall_prompt_bytes_per_s_median} | {tokens_per_prompt_byte_median} | {us_per_token_median} | {us_per_token_p95} | "
                "{wall_us_per_token_median} | {wall_us_per_token_p95} | {memory_bytes_max} | {memory_bytes_per_token_median} | {memory_bytes_per_token_max} |".format(
                    **{key: format_summary_value(value) for key, value in suite.items()}
                ),
                "",
                "| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
                "| {tok_per_s_stdev} | {tok_per_s_cv_pct} | {tok_per_s_iqr_pct} | {tok_per_s_p05_p95_spread_pct} | {wall_tok_per_s_iqr_pct} | {wall_tok_per_s_p05_p95_spread_pct} |".format(
                    **{key: format_summary_value(value) for key, value in suite.items()}
                ),
                "",
                "| Serial output bytes total | Serial output bytes max | Serial output lines total | Serial output lines max |",
                "| ---: | ---: | ---: | ---: |",
                "| {serial_output_bytes_total} | {serial_output_bytes_max} | {serial_output_lines_total} | {serial_output_lines_max} |".format(
                    **{key: format_summary_value(value) for key, value in suite.items()}
                ),
                "",
                "| Guest prompt SHA records | Guest prompt SHA matches | Guest prompt SHA mismatches | Guest prompt byte records | Guest prompt byte matches | Guest prompt byte mismatches |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
                "| {guest_prompt_sha256_records} | {guest_prompt_sha256_matches} | {guest_prompt_sha256_mismatches} | {guest_prompt_bytes_records} | {guest_prompt_bytes_matches} | {guest_prompt_bytes_mismatches} |".format(
                    **{key: format_summary_value(value) for key, value in suite.items()}
                ),
                "",
                "## Prompt Summary",
                "",
            ]
        )
    if report["summaries"]:
        lines.append(
            "| Prompt | Prompt bytes | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | tok/s IQR % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall tok/s IQR % | Wall P05-P95 spread % | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for summary in report["summaries"]:
            lines.append(
                "| {prompt} | {prompt_bytes} | {runs} | {ok_runs} | {failed_runs} | {ok_run_pct} | {timed_out_runs} | {nonzero_exit_runs} | {tokens_median} | {elapsed_us_median} | "
                "{host_overhead_us_median} | {host_overhead_pct_median} | {host_child_cpu_us_median} | {host_child_cpu_pct_median} | {host_child_tok_per_cpu_s_median} | {host_child_peak_rss_bytes_max} | {ttft_us_median} | {ttft_us_p95} | {tok_per_s_min} | {tok_per_s_p05} | {tok_per_s_median} | {tok_per_s_stdev} | {tok_per_s_cv_pct} | {tok_per_s_iqr_pct} | {tok_per_s_p05_p95_spread_pct} | "
                "{tok_per_s_max} | {wall_tok_per_s_p05} | {wall_tok_per_s_median} | {wall_tok_per_s_p95} | {wall_tok_per_s_iqr_pct} | {wall_tok_per_s_p05_p95_spread_pct} | {prompt_bytes_per_s_median} | {wall_prompt_bytes_per_s_median} | {tokens_per_prompt_byte_median} | {us_per_token_median} | {us_per_token_p95} | "
                "{wall_us_per_token_median} | {wall_us_per_token_p95} | {memory_bytes_max} | {memory_bytes_per_token_median} | {memory_bytes_per_token_max} |".format(
                    **{key: format_summary_value(value) for key, value in summary.items()}
                )
            )
        lines.extend(
            [
                "",
                "## Prompt Serial Output",
                "",
                "| Prompt | Serial output bytes total | Serial output bytes max | Serial output lines total | Serial output lines max |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for summary in report["summaries"]:
            lines.append(
                "| {prompt} | {serial_output_bytes_total} | {serial_output_bytes_max} | {serial_output_lines_total} | {serial_output_lines_max} |".format(
                    **{key: format_summary_value(value) for key, value in summary.items()}
                )
            )
    else:
        lines.append("No benchmark runs recorded.")

    launch_integrity = report.get("launch_sequence_integrity") or {}
    if launch_integrity:
        lines.extend(
            [
                "",
                "## Launch Sequence Integrity",
                "",
                "| Expected launches | Observed launches | Matched | Mismatched | Missing | Extra | Match |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                "| {expected_launches} | {observed_launches} | {matched_launches} | {mismatched_launches} | {missing_launches} | {extra_launches} | {launch_sequence_match} |".format(
                    **{key: format_summary_value(value) for key, value in launch_integrity.items()}
                ),
                "",
            ]
        )

    if report.get("variability_findings"):
        lines.extend(
            [
                "",
                "## Variability Gate Findings",
                "",
                "| Scope | Prompt | Metric | Value | Limit |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for finding in report["variability_findings"]:
            lines.append(
                "| {scope} | {prompt} | {metric} | {value} | {limit} |".format(
                    **{key: format_summary_value(value) for key, value in finding.items()}
                )
            )
    if report.get("telemetry_findings"):
        lines.extend(
            [
                "",
                "## Telemetry Gate Findings",
                "",
                "| Scope | Launch | Prompt | Iteration | Metric | Value | Limit |",
                "| --- | ---: | --- | ---: | --- | ---: | --- |",
            ]
        )
        for finding in report["telemetry_findings"]:
            lines.append(
                "| {scope} | {launch_index} | {prompt} | {iteration} | {metric} | {value} | {limit} |".format(
                    **{key: format_summary_value(value) for key, value in finding.items()}
                )
            )
    if report.get("launch_sequence_findings"):
        lines.extend(
            [
                "",
                "## Launch Sequence Findings",
                "",
                "| Metric | Value | Limit |",
                "| --- | --- | --- |",
            ]
        )
        for finding in report["launch_sequence_findings"]:
            lines.append(
                "| {metric} | {value} | {limit} |".format(
                    **{key: format_summary_value(value) for key, value in finding.items()}
                )
            )
    phase_rows = report.get("phase_summaries") if isinstance(report.get("phase_summaries"), list) else []
    if phase_rows:
        lines.extend(
            [
                "",
                "## Phase Summary",
                "",
                "| Phase | Launches | Prompts | OK | Failed | OK % | Timed out | Nonzero exit | Total tokens | Median tok/s | Median wall tok/s | Max memory bytes | Serial output bytes total | Serial output lines total |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in phase_rows:
            lines.append(
                "| {phase} | {launches} | {prompts} | {ok_runs} | {failed_runs} | {ok_run_pct} | "
                "{timed_out_runs} | {nonzero_exit_runs} | {total_tokens} | {tok_per_s_median} | "
                "{wall_tok_per_s_median} | {memory_bytes_max} | {serial_output_bytes_total} | {serial_output_lines_total} |".format(
                    **{key: format_summary_value(value) for key, value in row.items()}
                )
            )
    environment = report.get("environment") or {}
    if environment:
        lines.extend(
            [
                "",
                "## Environment",
                "",
                "| Platform | Machine | Python | CPU count | QEMU |",
                "| --- | --- | --- | ---: | --- |",
                "| {platform} | {machine} | {python} | {cpu_count} | {qemu} |".format(
                    platform=format_summary_value(environment.get("platform")),
                    machine=format_summary_value(environment.get("machine")),
                    python=format_summary_value(environment.get("python")),
                    cpu_count=format_summary_value(environment.get("cpu_count")),
                    qemu=format_summary_value(environment.get("qemu_version") or environment.get("qemu_bin")),
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def markdown_dry_run_report(report: dict[str, Any]) -> str:
    prompt_suite = report.get("prompt_suite") or {}
    command = report.get("command") or []
    environment = report.get("environment") or {}
    image = report.get("image") or {}
    qemu_args_files = report.get("qemu_args_files") if isinstance(report.get("qemu_args_files"), list) else []
    launch_plan = report.get("launch_plan") if isinstance(report.get("launch_plan"), list) else []
    lines = [
        "# QEMU Prompt Benchmark Dry Run",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Profile: {format_summary_value(report.get('profile'))}",
        f"Model: {format_summary_value(report.get('model'))}",
        f"Quantization: {format_summary_value(report.get('quantization'))}",
        f"Commit: {format_summary_value(report.get('commit'))}",
        f"Prompt suite: {prompt_suite.get('suite_sha256', '-')}",
        f"Command SHA256: {report.get('command_sha256', '-')}",
        f"Command air-gap OK: {format_summary_value((report.get('command_airgap') or {}).get('ok'))}",
        f"Launch plan SHA256: {report.get('launch_plan_sha256', '-')}",
        f"Expected launch sequence SHA256: {report.get('expected_launch_sequence_sha256', '-')}",
        f"Prompt count: {report['prompt_count']}",
        f"Prompt count floor: {format_summary_value(report.get('min_prompt_count'))}",
        f"Warmup launches: {report['planned_warmup_launches']}",
        f"Measured launches: {report['planned_measured_launches']}",
        f"Total launches: {report['planned_total_launches']}",
        "",
        "## Command",
        "",
        "```text",
        " ".join(command),
        "```",
    ]
    if image or qemu_args_files:
        lines.extend(
            [
                "",
                "## Inputs",
                "",
                "| Image | Exists | Size bytes | SHA256 | QEMU args files |",
                "| --- | --- | ---: | --- | ---: |",
                "| {path} | {exists} | {size_bytes} | {sha256} | {args_files} |".format(
                    path=format_summary_value(image.get("path")),
                    exists=format_summary_value(image.get("exists")),
                    size_bytes=format_summary_value(image.get("size_bytes")),
                    sha256=format_summary_value(image.get("sha256")),
                    args_files=len(qemu_args_files),
                ),
            ]
        )
        if qemu_args_files:
            lines.extend(
                [
                    "",
                    "| QEMU args file | Exists | Size bytes | SHA256 |",
                    "| --- | --- | ---: | --- |",
                ]
            )
            for item in qemu_args_files:
                lines.append(
                    "| {path} | {exists} | {size_bytes} | {sha256} |".format(
                        path=format_summary_value(item.get("path")),
                        exists=format_summary_value(item.get("exists")),
                        size_bytes=format_summary_value(item.get("size_bytes")),
                        sha256=format_summary_value(item.get("sha256")),
                    )
                )
    if launch_plan:
        lines.extend(
            [
                "",
                "## Launch Plan",
                "",
                "| Launch | Phase | Prompt index | Prompt | Iteration | Prompt bytes | Prompt SHA256 |",
                "| ---: | --- | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for row in launch_plan:
            lines.append(
                "| {launch_index} | {phase} | {prompt_index} | {prompt_id} | {iteration} | {prompt_bytes} | {prompt_sha256} |".format(
                    launch_index=format_summary_value(row.get("launch_index")),
                    phase=format_summary_value(row.get("phase")),
                    prompt_index=format_summary_value(row.get("prompt_index")),
                    prompt_id=format_summary_value(row.get("prompt_id")),
                    iteration=format_summary_value(row.get("iteration")),
                    prompt_bytes=format_summary_value(row.get("prompt_bytes")),
                    prompt_sha256=format_summary_value(row.get("prompt_sha256")),
                )
            )
    if environment:
        lines.extend(
            [
                "",
                "## Environment",
                "",
                "| Platform | Machine | Python | CPU count | QEMU |",
                "| --- | --- | --- | ---: | --- |",
                "| {platform} | {machine} | {python} | {cpu_count} | {qemu} |".format(
                    platform=format_summary_value(environment.get("platform")),
                    machine=format_summary_value(environment.get("machine")),
                    python=format_summary_value(environment.get("python")),
                    cpu_count=format_summary_value(environment.get("cpu_count")),
                    qemu=format_summary_value(environment.get("qemu_version") or environment.get("qemu_bin")),
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def format_summary_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def dry_run_payload(
    *,
    command: list[str],
    prompts_path: Path,
    prompts: list[PromptCase],
    warmup: int,
    repeat: int,
    profile: str = "default",
    model: str = "",
    quantization: str = "",
    commit: str = "unknown",
    max_launches: int | None = None,
    min_prompt_count: int | None = None,
    environment: dict[str, Any] | None = None,
    image: dict[str, Any] | None = None,
    qemu_args_files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt_count = len(prompts)
    planned_warmups = prompt_count * warmup
    planned_measured = prompt_count * repeat
    planned_total = planned_warmups + planned_measured
    plan = dry_run_launch_plan(prompts, warmup=warmup, repeat=repeat)
    launch_sequence = launch_sequence_from_plan(plan)
    return {
        "generated_at": iso_now(),
        "status": "planned",
        "command": command,
        "command_sha256": command_hash(command),
        "command_airgap": command_airgap_metadata(command),
        "profile": profile,
        "model": model,
        "quantization": quantization,
        "commit": commit,
        "launch_plan_sha256": launch_plan_hash(plan),
        "expected_launch_sequence_sha256": launch_sequence_hash(launch_sequence),
        "launch_plan": plan,
        "prompt_count": prompt_count,
        "prompt_suite": prompt_suite_metadata(prompts_path, prompts),
        "environment": environment or {},
        "image": image or {},
        "qemu_args_files": qemu_args_files or [],
        "warmup": warmup,
        "repeat": repeat,
        "max_launches": max_launches,
        "min_prompt_count": min_prompt_count,
        "planned_warmup_launches": planned_warmups,
        "planned_measured_launches": planned_measured,
        "planned_total_launches": planned_total,
    }


def validate_prompt_count(prompts: list[PromptCase], min_prompt_count: int | None) -> None:
    if min_prompt_count is None:
        return
    if len(prompts) < min_prompt_count:
        raise ValueError(
            f"prompt count ({len(prompts)}) is below --min-prompt-count ({min_prompt_count}); "
            "use a larger prompt suite or lower the coverage gate"
        )


def validate_launch_budget(
    prompts: list[PromptCase],
    *,
    warmup: int,
    repeat: int,
    max_launches: int | None,
) -> None:
    if max_launches is None:
        return
    planned = len(prompts) * (warmup + repeat)
    if planned > max_launches:
        raise ValueError(
            f"planned QEMU launches ({planned}) exceed --max-launches ({max_launches}); "
            "reduce prompts, --warmup, or --repeat"
        )


def write_dry_run_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "qemu_prompt_bench_dry_run_latest.json"
    latest_md = output_dir / "qemu_prompt_bench_dry_run_latest.md"
    latest_csv = output_dir / "qemu_prompt_bench_dry_run_latest.csv"
    latest_launch_csv = output_dir / "qemu_prompt_bench_dry_run_launches_latest.csv"
    latest_junit = output_dir / "qemu_prompt_bench_dry_run_junit_latest.xml"
    latest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_md.write_text(markdown_dry_run_report(report), encoding="utf-8")
    write_dry_run_csv_report(report, latest_csv)
    write_dry_run_launch_csv_report(report, latest_launch_csv)
    write_dry_run_junit_report(report, latest_junit)
    stamped = output_dir / f"qemu_prompt_bench_dry_run_{report['generated_at'].replace(':', '').replace('-', '')}.json"
    stamped.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return latest


def write_dry_run_csv_report(report: dict[str, Any], path: Path) -> None:
    prompt_suite = report.get("prompt_suite") or {}
    command = report.get("command") or []
    image = report.get("image") or {}
    qemu_args_files = report.get("qemu_args_files") if isinstance(report.get("qemu_args_files"), list) else []
    fields = [
        "generated_at",
        "status",
        "profile",
        "model",
        "quantization",
        "commit",
        "command_sha256",
        "command_airgap_ok",
        "command_has_explicit_nic_none",
        "command_has_legacy_net_none",
        "command_airgap_violations",
        "launch_plan_sha256",
        "expected_launch_sequence_sha256",
        "prompt_count",
        "prompt_suite_sha256",
        "prompt_bytes_total",
        "prompt_bytes_min",
        "prompt_bytes_max",
        "expected_token_prompts",
        "expected_tokens_total",
        "warmup",
        "repeat",
        "planned_warmup_launches",
        "planned_measured_launches",
        "planned_total_launches",
        "max_launches",
        "min_prompt_count",
        "image_path",
        "image_exists",
        "image_size_bytes",
        "image_sha256",
        "qemu_args_file_count",
        "command",
    ]
    row = {
        "generated_at": report.get("generated_at"),
        "status": report.get("status"),
        "profile": report.get("profile"),
        "model": report.get("model"),
        "quantization": report.get("quantization"),
        "commit": report.get("commit"),
        "command_sha256": report.get("command_sha256"),
        "command_airgap_ok": (report.get("command_airgap") or {}).get("ok"),
        "command_has_explicit_nic_none": (report.get("command_airgap") or {}).get("explicit_nic_none"),
        "command_has_legacy_net_none": (report.get("command_airgap") or {}).get("legacy_net_none"),
        "command_airgap_violations": ";".join((report.get("command_airgap") or {}).get("violations") or []),
        "launch_plan_sha256": report.get("launch_plan_sha256"),
        "expected_launch_sequence_sha256": report.get("expected_launch_sequence_sha256"),
        "prompt_count": report.get("prompt_count"),
        "prompt_suite_sha256": prompt_suite.get("suite_sha256"),
        "prompt_bytes_total": prompt_suite.get("prompt_bytes_total"),
        "prompt_bytes_min": prompt_suite.get("prompt_bytes_min"),
        "prompt_bytes_max": prompt_suite.get("prompt_bytes_max"),
        "expected_token_prompts": prompt_suite.get("expected_token_prompts"),
        "expected_tokens_total": prompt_suite.get("expected_tokens_total"),
        "warmup": report.get("warmup"),
        "repeat": report.get("repeat"),
        "planned_warmup_launches": report.get("planned_warmup_launches"),
        "planned_measured_launches": report.get("planned_measured_launches"),
        "planned_total_launches": report.get("planned_total_launches"),
        "max_launches": report.get("max_launches"),
        "min_prompt_count": report.get("min_prompt_count"),
        "image_path": image.get("path"),
        "image_exists": image.get("exists"),
        "image_size_bytes": image.get("size_bytes"),
        "image_sha256": image.get("sha256"),
        "qemu_args_file_count": len(qemu_args_files),
        "command": json.dumps(command, separators=(",", ":")),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def write_dry_run_launch_csv_report(report: dict[str, Any], path: Path) -> None:
    fields = [
        "generated_at",
        "command_sha256",
        "command_airgap_ok",
        "command_has_explicit_nic_none",
        "command_has_legacy_net_none",
        "launch_plan_sha256",
        "launch_index",
        "phase",
        "prompt_index",
        "prompt_id",
        "prompt_sha256",
        "prompt_bytes",
        "expected_tokens",
        "iteration",
    ]
    launch_plan = report.get("launch_plan") if isinstance(report.get("launch_plan"), list) else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in launch_plan:
            writer.writerow(
                {
                    "generated_at": report.get("generated_at"),
                    "command_sha256": report.get("command_sha256"),
                    "command_airgap_ok": (report.get("command_airgap") or {}).get("ok"),
                    "command_has_explicit_nic_none": (report.get("command_airgap") or {}).get("explicit_nic_none"),
                    "command_has_legacy_net_none": (report.get("command_airgap") or {}).get("legacy_net_none"),
                    "launch_plan_sha256": report.get("launch_plan_sha256"),
                    "launch_index": row.get("launch_index"),
                    "phase": row.get("phase"),
                    "prompt_index": row.get("prompt_index"),
                    "prompt_id": row.get("prompt_id"),
                    "prompt_sha256": row.get("prompt_sha256"),
                    "prompt_bytes": row.get("prompt_bytes"),
                    "expected_tokens": row.get("expected_tokens"),
                    "iteration": row.get("iteration"),
                }
            )


def write_dry_run_junit_report(report: dict[str, Any], path: Path) -> None:
    command = report.get("command") or []
    command_text = " ".join(command) if isinstance(command, list) else str(command)
    image = report.get("image") or {}
    qemu_args_files = report.get("qemu_args_files") if isinstance(report.get("qemu_args_files"), list) else []
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_bench_dry_run",
            "tests": "1",
            "failures": "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": "qemu_prompt_bench.dry_run",
            "name": str(report.get("command_sha256") or "planned-command"),
        },
    )
    properties = ET.SubElement(case, "properties")
    for name in (
        "status",
        "profile",
        "model",
        "quantization",
        "commit",
        "prompt_count",
        "planned_warmup_launches",
        "planned_measured_launches",
        "planned_total_launches",
        "max_launches",
        "min_prompt_count",
        "command_sha256",
        "launch_plan_sha256",
        "expected_launch_sequence_sha256",
    ):
        ET.SubElement(
            properties,
            "property",
            {"name": name, "value": format_summary_value(report.get(name))},
        )
    command_airgap = report.get("command_airgap") or {}
    for name in ("ok", "explicit_nic_none", "legacy_net_none"):
        ET.SubElement(
            properties,
            "property",
            {"name": f"command_airgap_{name}", "value": format_summary_value(command_airgap.get(name))},
        )
    ET.SubElement(
        properties,
        "property",
        {
            "name": "command_airgap_violations",
            "value": ";".join(command_airgap.get("violations") or []),
        },
    )
    for name in ("path", "exists", "size_bytes", "sha256"):
        ET.SubElement(
            properties,
            "property",
            {"name": f"image_{name}", "value": format_summary_value(image.get(name))},
        )
    ET.SubElement(
        properties,
        "property",
        {"name": "qemu_args_file_count", "value": str(len(qemu_args_files))},
    )
    ET.SubElement(properties, "property", {"name": "command", "value": command_text})
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def write_report(
    runs: list[BenchRun],
    output_dir: Path,
    *,
    prompt_suite: dict[str, Any] | None = None,
    launch_plan: list[dict[str, Any]] | None = None,
    warmups: list[BenchRun] | None = None,
    max_suite_cv_pct: float | None = None,
    max_prompt_cv_pct: float | None = None,
    max_suite_iqr_pct: float | None = None,
    max_prompt_iqr_pct: float | None = None,
    require_tokens: bool = False,
    require_tok_per_s: bool = False,
    require_memory: bool = False,
    require_ttft_us: bool = False,
    min_tokens: int | None = None,
    min_total_tokens: int | None = None,
    min_tok_per_s: float | None = None,
    min_wall_tok_per_s: float | None = None,
    max_memory_bytes: int | None = None,
    max_ttft_us: int | None = None,
    max_host_overhead_us: int | None = None,
    max_host_overhead_pct: float | None = None,
    max_wall_timeout_pct: float | None = None,
    min_host_child_tok_per_cpu_s: float | None = None,
    min_tokens_per_prompt_byte: float | None = None,
    require_host_child_rss: bool = False,
    max_host_child_rss_bytes: int | None = None,
    max_memory_bytes_per_token: float | None = None,
    max_serial_output_bytes: int | None = None,
    max_serial_output_lines: int | None = None,
    require_guest_prompt_sha256_match: bool = False,
    require_guest_prompt_bytes_match: bool = False,
    require_expected_tokens_match: bool = False,
    max_launches: int | None = None,
    min_prompt_count: int | None = None,
    environment: dict[str, Any] | None = None,
    image: dict[str, Any] | None = None,
    qemu_args_files: list[dict[str, Any]] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_runs = warmups or []
    all_runs = warmup_runs + runs
    suite = suite_summary(runs)
    summaries = summarize_runs(runs)
    expected_launch_sequence = launch_sequence_from_plan(launch_plan or [])
    observed_launch_sequence = launch_sequence_from_runs(all_runs)
    launch_integrity = launch_sequence_integrity(expected_launch_sequence, observed_launch_sequence)
    launch_findings = launch_sequence_findings(launch_integrity)
    findings = variability_findings(
        suite,
        summaries,
        max_suite_cv_pct=max_suite_cv_pct,
        max_prompt_cv_pct=max_prompt_cv_pct,
        max_suite_iqr_pct=max_suite_iqr_pct,
        max_prompt_iqr_pct=max_prompt_iqr_pct,
    )
    telemetry = telemetry_findings(
        runs,
        suite,
        require_tokens=require_tokens,
        require_tok_per_s=require_tok_per_s,
        require_memory=require_memory,
        require_ttft_us=require_ttft_us,
        min_tokens=min_tokens,
        min_total_tokens=min_total_tokens,
        min_tok_per_s=min_tok_per_s,
        min_wall_tok_per_s=min_wall_tok_per_s,
        max_memory_bytes=max_memory_bytes,
        max_ttft_us=max_ttft_us,
        max_host_overhead_us=max_host_overhead_us,
        max_host_overhead_pct=max_host_overhead_pct,
        max_wall_timeout_pct=max_wall_timeout_pct,
        min_host_child_tok_per_cpu_s=min_host_child_tok_per_cpu_s,
        min_tokens_per_prompt_byte=min_tokens_per_prompt_byte,
        require_host_child_rss=require_host_child_rss,
        max_host_child_rss_bytes=max_host_child_rss_bytes,
        max_memory_bytes_per_token=max_memory_bytes_per_token,
        max_serial_output_bytes=max_serial_output_bytes,
        max_serial_output_lines=max_serial_output_lines,
        require_guest_prompt_sha256_match=require_guest_prompt_sha256_match,
        require_guest_prompt_bytes_match=require_guest_prompt_bytes_match,
        require_expected_tokens_match=require_expected_tokens_match,
    )
    all_findings = findings + telemetry + launch_findings
    report = {
        "generated_at": iso_now(),
        "status": report_status(all_runs, all_findings),
        "command": all_runs[0].command if all_runs else [],
        "prompt_suite": prompt_suite or {},
        "launch_plan_sha256": launch_plan_hash(launch_plan or []),
        "expected_launch_sequence_sha256": launch_integrity["expected_launch_sequence_sha256"],
        "observed_launch_sequence_sha256": launch_integrity["observed_launch_sequence_sha256"],
        "launch_sequence_integrity": launch_integrity,
        "launch_sequence_findings": launch_findings,
        "launch_plan": launch_plan or [],
        "environment": environment or {},
        "image": image or {},
        "qemu_args_files": qemu_args_files or [],
        "command_sha256": command_hash(all_runs[0].command) if all_runs else command_hash([]),
        "command_airgap": command_airgap_metadata(all_runs[0].command) if all_runs else command_airgap_metadata([]),
        "profile": all_runs[0].profile if all_runs else "",
        "model": all_runs[0].model if all_runs else "",
        "quantization": all_runs[0].quantization if all_runs else "",
        "commit": all_runs[0].commit if all_runs else "",
        "max_launches": max_launches,
        "min_prompt_count": min_prompt_count,
        "planned_warmup_launches": len(warmup_runs),
        "planned_measured_launches": len(runs),
        "planned_total_launches": len(all_runs),
        "warmups": [asdict(run) for run in warmup_runs],
        "suite_summary": suite,
        "summaries": summaries,
        "phase_summaries": phase_summaries(warmup_runs, runs),
        "variability_gates": {
            "max_suite_cv_pct": max_suite_cv_pct,
            "max_prompt_cv_pct": max_prompt_cv_pct,
            "max_suite_iqr_pct": max_suite_iqr_pct,
            "max_prompt_iqr_pct": max_prompt_iqr_pct,
        },
        "variability_findings": findings,
        "telemetry_gates": {
            "require_tokens": require_tokens,
            "require_tok_per_s": require_tok_per_s,
            "require_memory": require_memory,
            "require_ttft_us": require_ttft_us,
            "min_tokens": min_tokens,
            "min_total_tokens": min_total_tokens,
            "min_tok_per_s": min_tok_per_s,
            "min_wall_tok_per_s": min_wall_tok_per_s,
            "max_memory_bytes": max_memory_bytes,
            "max_ttft_us": max_ttft_us,
            "max_host_overhead_us": max_host_overhead_us,
            "max_host_overhead_pct": max_host_overhead_pct,
            "max_wall_timeout_pct": max_wall_timeout_pct,
            "min_host_child_tok_per_cpu_s": min_host_child_tok_per_cpu_s,
            "min_tokens_per_prompt_byte": min_tokens_per_prompt_byte,
            "require_host_child_rss": require_host_child_rss,
            "max_host_child_rss_bytes": max_host_child_rss_bytes,
            "max_memory_bytes_per_token": max_memory_bytes_per_token,
            "max_serial_output_bytes": max_serial_output_bytes,
            "max_serial_output_lines": max_serial_output_lines,
            "require_guest_prompt_sha256_match": require_guest_prompt_sha256_match,
            "require_guest_prompt_bytes_match": require_guest_prompt_bytes_match,
            "require_expected_tokens_match": require_expected_tokens_match,
            "min_prompt_count": min_prompt_count,
        },
        "telemetry_findings": telemetry,
        "benchmarks": [asdict(run) for run in runs],
    }
    latest = output_dir / "qemu_prompt_bench_latest.json"
    latest_md = output_dir / "qemu_prompt_bench_latest.md"
    latest_csv = output_dir / "qemu_prompt_bench_latest.csv"
    latest_summary_csv = output_dir / "qemu_prompt_bench_summary_latest.csv"
    latest_phase_csv = output_dir / "qemu_prompt_bench_phases_latest.csv"
    latest_launch_csv = output_dir / "qemu_prompt_bench_launches_latest.csv"
    latest_junit = output_dir / "qemu_prompt_bench_junit_latest.xml"
    latest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_md.write_text(markdown_report(report), encoding="utf-8")
    write_csv_report(runs, latest_csv)
    write_summary_csv_report(report, latest_summary_csv)
    write_phase_csv_report(report, latest_phase_csv)
    write_launch_csv_report(report, latest_launch_csv)
    write_launch_jsonl_report(report, output_dir / "qemu_prompt_bench_launches_latest.jsonl")
    write_junit_report(runs, warmup_runs, findings, telemetry, launch_findings, latest_junit)
    stamped = output_dir / f"qemu_prompt_bench_{report['generated_at'].replace(':', '').replace('-', '')}.json"
    stamped.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return latest


def write_csv_report(runs: list[BenchRun], path: Path) -> None:
    fields = [
        "timestamp",
        "commit",
        "benchmark",
        "profile",
        "model",
        "quantization",
        "prompt",
        "prompt_sha256",
        "guest_prompt_sha256",
        "guest_prompt_sha256_match",
        "prompt_bytes",
        "guest_prompt_bytes",
        "guest_prompt_bytes_match",
        "phase",
        "launch_index",
        "iteration",
        "tokens",
        "expected_tokens",
        "expected_tokens_match",
        "elapsed_us",
        "wall_elapsed_us",
        "timeout_seconds",
        "wall_timeout_pct",
        "host_overhead_us",
        "host_overhead_pct",
        "host_child_user_cpu_us",
        "host_child_system_cpu_us",
        "host_child_cpu_us",
        "host_child_cpu_pct",
        "host_child_tok_per_cpu_s",
        "host_child_peak_rss_bytes",
        "ttft_us",
        "tok_per_s",
        "wall_tok_per_s",
        "prompt_bytes_per_s",
        "wall_prompt_bytes_per_s",
        "tokens_per_prompt_byte",
        "us_per_token",
        "wall_us_per_token",
        "memory_bytes",
        "memory_bytes_per_token",
        "stdout_bytes",
        "stderr_bytes",
        "serial_output_bytes",
        "serial_output_lines",
        "returncode",
        "timed_out",
        "exit_class",
        "failure_reason",
        "command_sha256",
        "command_airgap_ok",
        "command_has_explicit_nic_none",
        "command_has_legacy_net_none",
        "command_airgap_violations",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            row = asdict(run)
            writer.writerow({field: row[field] for field in fields})


def write_summary_csv_report(report: dict[str, Any], path: Path) -> None:
    fields = [
        "scope",
        "prompt",
        "prompt_bytes",
        "prompts",
        "runs",
        "ok_runs",
        "failed_runs",
        "ok_run_pct",
        "timed_out_runs",
        "nonzero_exit_runs",
        "tokens_median",
        "total_tokens",
        "elapsed_us_median",
        "total_elapsed_us",
        "host_overhead_us_median",
        "host_overhead_pct_median",
        "host_child_cpu_us_median",
        "host_child_cpu_pct_median",
        "host_child_tok_per_cpu_s_median",
        "host_child_peak_rss_bytes_max",
        "ttft_us_median",
        "ttft_us_p95",
        "tok_per_s_min",
        "tok_per_s_p05",
        "tok_per_s_median",
        "tok_per_s_stdev",
        "tok_per_s_cv_pct",
        "tok_per_s_iqr_pct",
        "tok_per_s_p05_p95_spread_pct",
        "tok_per_s_p95",
        "tok_per_s_max",
        "wall_tok_per_s_p05",
        "wall_tok_per_s_median",
        "wall_tok_per_s_p95",
        "wall_tok_per_s_iqr_pct",
        "wall_tok_per_s_p05_p95_spread_pct",
        "prompt_bytes_per_s_median",
        "wall_prompt_bytes_per_s_median",
        "tokens_per_prompt_byte_median",
        "us_per_token_median",
        "us_per_token_p95",
        "wall_us_per_token_median",
        "wall_us_per_token_p95",
        "memory_bytes_max",
        "memory_bytes_per_token_median",
        "memory_bytes_per_token_max",
        "serial_output_bytes_total",
        "serial_output_bytes_max",
        "serial_output_lines_total",
        "serial_output_lines_max",
        "guest_prompt_sha256_records",
        "guest_prompt_sha256_matches",
        "guest_prompt_sha256_mismatches",
        "guest_prompt_bytes_records",
        "guest_prompt_bytes_matches",
        "guest_prompt_bytes_mismatches",
    ]
    rows: list[dict[str, Any]] = []
    suite = report.get("suite_summary") or {}
    if suite:
        rows.append(
            {
                "scope": "suite",
                "prompt": "",
                "prompt_bytes": suite.get("measured_prompt_bytes_total"),
                **suite,
            }
        )
    for summary in report.get("summaries", []):
        if isinstance(summary, dict):
            rows.append({"scope": "prompt", "prompts": "", "total_tokens": "", "total_elapsed_us": "", **summary})

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_phase_csv_report(report: dict[str, Any], path: Path) -> None:
    fields = [
        "generated_at",
        "phase",
        "launches",
        "prompts",
        "runs",
        "ok_runs",
        "failed_runs",
        "ok_run_pct",
        "timed_out_runs",
        "nonzero_exit_runs",
        "measured_prompt_bytes_total",
        "prompt_bytes_min",
        "prompt_bytes_max",
        "total_tokens",
        "total_elapsed_us",
        "tok_per_s_p05",
        "tok_per_s_median",
        "tok_per_s_p95",
        "wall_tok_per_s_p05",
        "wall_tok_per_s_median",
        "wall_tok_per_s_p95",
        "ttft_us_median",
        "ttft_us_p95",
        "host_overhead_pct_median",
        "host_child_cpu_us_median",
        "host_child_cpu_pct_median",
        "host_child_tok_per_cpu_s_median",
        "host_child_peak_rss_bytes_max",
        "us_per_token_median",
        "wall_us_per_token_median",
        "memory_bytes_max",
        "memory_bytes_per_token_median",
        "memory_bytes_per_token_max",
        "serial_output_bytes_total",
        "serial_output_bytes_max",
        "serial_output_lines_total",
        "serial_output_lines_max",
        "guest_prompt_sha256_records",
        "guest_prompt_sha256_matches",
        "guest_prompt_sha256_mismatches",
        "guest_prompt_bytes_records",
        "guest_prompt_bytes_matches",
        "guest_prompt_bytes_mismatches",
    ]
    phase_rows = report.get("phase_summaries") if isinstance(report.get("phase_summaries"), list) else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in phase_rows:
            writer.writerow(
                {
                    field: report.get("generated_at") if field == "generated_at" else row.get(field)
                    for field in fields
                }
            )


def write_launch_csv_report(report: dict[str, Any], path: Path) -> None:
    fields = [
        "generated_at",
        "command_sha256",
        "command_airgap_ok",
        "command_has_explicit_nic_none",
        "command_has_legacy_net_none",
        "command_airgap_violations",
        "launch_plan_sha256",
        "expected_launch_sequence_sha256",
        "observed_launch_sequence_sha256",
        "launch_index",
        "phase",
        "prompt",
        "prompt_sha256",
        "guest_prompt_sha256",
        "guest_prompt_sha256_match",
        "prompt_bytes",
        "guest_prompt_bytes",
        "guest_prompt_bytes_match",
        "iteration",
        "returncode",
        "timed_out",
        "exit_class",
        "failure_reason",
        "tokens",
        "expected_tokens",
        "expected_tokens_match",
        "elapsed_us",
        "wall_elapsed_us",
        "timeout_seconds",
        "wall_timeout_pct",
        "tok_per_s",
        "wall_tok_per_s",
        "memory_bytes",
        "serial_output_bytes",
        "serial_output_lines",
    ]
    rows = []
    for key in ("warmups", "benchmarks"):
        values = report.get(key)
        if isinstance(values, list):
            rows.extend(row for row in values if isinstance(row, dict))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item.get("launch_index") or 0)):
            run_values = {
                field: row.get(field)
                for field in fields
                if field
                not in {
                    "generated_at",
                    "command_sha256",
                    "command_airgap_ok",
                    "command_has_explicit_nic_none",
                    "command_has_legacy_net_none",
                    "command_airgap_violations",
                    "launch_plan_sha256",
                    "expected_launch_sequence_sha256",
                    "observed_launch_sequence_sha256",
                }
            }
            writer.writerow(
                {
                    "generated_at": report.get("generated_at"),
                    "command_sha256": report.get("command_sha256"),
                    "command_airgap_ok": report.get("command_airgap", {}).get("ok"),
                    "command_has_explicit_nic_none": report.get("command_airgap", {}).get("explicit_nic_none"),
                    "command_has_legacy_net_none": report.get("command_airgap", {}).get("legacy_net_none"),
                    "command_airgap_violations": ";".join(report.get("command_airgap", {}).get("violations") or []),
                    "launch_plan_sha256": report.get("launch_plan_sha256"),
                    "expected_launch_sequence_sha256": report.get("expected_launch_sequence_sha256"),
                    "observed_launch_sequence_sha256": report.get("observed_launch_sequence_sha256"),
                    **run_values,
                }
            )


def launch_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in ("warmups", "benchmarks"):
        values = report.get(key)
        if isinstance(values, list):
            rows.extend(row for row in values if isinstance(row, dict))
    return sorted(rows, key=lambda item: int(item.get("launch_index") or 0))


def write_launch_jsonl_report(report: dict[str, Any], path: Path) -> None:
    top_level_fields = {
        "generated_at": report.get("generated_at"),
        "status": report.get("status"),
        "command_sha256": report.get("command_sha256"),
        "command_airgap_ok": report.get("command_airgap", {}).get("ok"),
        "command_has_explicit_nic_none": report.get("command_airgap", {}).get("explicit_nic_none"),
        "command_has_legacy_net_none": report.get("command_airgap", {}).get("legacy_net_none"),
        "command_airgap_violations": report.get("command_airgap", {}).get("violations") or [],
        "launch_plan_sha256": report.get("launch_plan_sha256"),
        "expected_launch_sequence_sha256": report.get("expected_launch_sequence_sha256"),
        "observed_launch_sequence_sha256": report.get("observed_launch_sequence_sha256"),
        "prompt_suite_sha256": (report.get("prompt_suite") or {}).get("suite_sha256"),
        "profile": report.get("profile"),
        "model": report.get("model"),
        "quantization": report.get("quantization"),
        "commit": report.get("commit"),
    }
    with path.open("w", encoding="utf-8") as handle:
        for row in launch_rows(report):
            handle.write(
                json.dumps(
                    {
                        **top_level_fields,
                        **row,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_junit_report(
    runs: list[BenchRun],
    warmups: list[BenchRun],
    variability_findings: list[dict[str, Any]],
    telemetry_findings: list[dict[str, Any]],
    launch_sequence_findings: list[dict[str, Any]],
    path: Path,
) -> None:
    all_runs = [("warmup", run) for run in warmups] + [("measured", run) for run in runs]
    failed_runs = [run for _, run in all_runs if run.returncode != 0 or run.timed_out]
    failures = (
        len(failed_runs)
        + len(variability_findings)
        + len(telemetry_findings)
        + len(launch_sequence_findings)
    )
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_bench",
            "tests": str(
                len(all_runs)
                + len(variability_findings)
                + len(telemetry_findings)
                + len(launch_sequence_findings)
            ),
            "failures": str(failures),
            "errors": "0",
        },
    )

    for phase, run in all_runs:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": f"qemu_prompt_bench.{phase}",
                "name": f"{run.profile}:{run.model}:{run.quantization}:{run.launch_index}:{run.prompt}:{run.iteration}",
            },
        )
        if run.returncode != 0 or run.timed_out:
            message = (
                f"exit_class={run.exit_class} returncode={run.returncode} "
                f"timed_out={run.timed_out}"
            )
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "qemu_prompt_failure",
                    "message": message,
                },
            )
            failure.text = (
                f"phase={phase}\n"
                f"launch_index={run.launch_index}\n"
                f"prompt={run.prompt}\n"
                f"iteration={run.iteration}\n"
                f"returncode={run.returncode}\n"
                f"timed_out={run.timed_out}\n"
                f"exit_class={run.exit_class}\n"
                f"failure_reason={format_summary_value(run.failure_reason)}\n"
                f"tokens={format_summary_value(run.tokens)}\n"
                f"expected_tokens={format_summary_value(run.expected_tokens)}\n"
                f"expected_tokens_match={format_summary_value(run.expected_tokens_match)}\n"
                f"elapsed_us={run.elapsed_us}\n"
                f"wall_elapsed_us={run.wall_elapsed_us}\n"
                f"timeout_seconds={format_summary_value(run.timeout_seconds)}\n"
                f"wall_timeout_pct={format_summary_value(run.wall_timeout_pct)}\n"
                f"host_overhead_us={run.host_overhead_us}\n"
                f"host_overhead_pct={format_summary_value(run.host_overhead_pct)}\n"
                f"host_child_user_cpu_us={format_summary_value(run.host_child_user_cpu_us)}\n"
                f"host_child_system_cpu_us={format_summary_value(run.host_child_system_cpu_us)}\n"
                f"host_child_cpu_us={format_summary_value(run.host_child_cpu_us)}\n"
                f"host_child_cpu_pct={format_summary_value(run.host_child_cpu_pct)}\n"
                f"host_child_tok_per_cpu_s={format_summary_value(run.host_child_tok_per_cpu_s)}\n"
                f"host_child_peak_rss_bytes={format_summary_value(run.host_child_peak_rss_bytes)}\n"
                f"ttft_us={format_summary_value(run.ttft_us)}\n"
                f"tok_per_s={format_summary_value(run.tok_per_s)}\n"
                f"wall_tok_per_s={format_summary_value(run.wall_tok_per_s)}\n"
                f"prompt_bytes_per_s={format_summary_value(run.prompt_bytes_per_s)}\n"
                f"wall_prompt_bytes_per_s={format_summary_value(run.wall_prompt_bytes_per_s)}\n"
                f"guest_prompt_sha256={format_summary_value(run.guest_prompt_sha256)}\n"
                f"guest_prompt_sha256_match={format_summary_value(run.guest_prompt_sha256_match)}\n"
                f"guest_prompt_bytes={format_summary_value(run.guest_prompt_bytes)}\n"
                f"guest_prompt_bytes_match={format_summary_value(run.guest_prompt_bytes_match)}\n"
                f"us_per_token={format_summary_value(run.us_per_token)}\n"
                f"wall_us_per_token={format_summary_value(run.wall_us_per_token)}\n"
                f"memory_bytes_per_token={format_summary_value(run.memory_bytes_per_token)}\n"
                f"stdout_bytes={run.stdout_bytes}\n"
                f"stderr_bytes={run.stderr_bytes}\n"
                f"serial_output_bytes={run.serial_output_bytes}\n"
                f"serial_output_lines={run.serial_output_lines}\n"
                f"command_sha256={run.command_sha256}\n"
                f"stdout_tail={run.stdout_tail}\n"
                f"stderr_tail={run.stderr_tail}\n"
            )

    for index, finding in enumerate(variability_findings, 1):
        scope = str(finding.get("scope", "unknown"))
        prompt = str(finding.get("prompt") or "suite")
        metric = str(finding.get("metric", "unknown"))
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "qemu_prompt_bench.variability",
                "name": f"{scope}:{prompt}:{metric}:{index}",
            },
        )
        message = (
            f"{metric}={format_summary_value(finding.get('value'))} "
            f"limit={format_summary_value(finding.get('limit'))}"
        )
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "benchmark_variability",
                "message": message,
            },
        )
        failure.text = "\n".join(f"{key}={format_summary_value(value)}" for key, value in sorted(finding.items()))

    for index, finding in enumerate(telemetry_findings, 1):
        prompt = str(finding.get("prompt") or "unknown")
        iteration = str(finding.get("iteration") or "unknown")
        metric = str(finding.get("metric", "unknown"))
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "qemu_prompt_bench.telemetry",
                "name": f"{prompt}:{iteration}:{metric}:{index}",
            },
        )
        message = (
            f"{metric}={format_summary_value(finding.get('value'))} "
            f"limit={format_summary_value(finding.get('limit'))}"
        )
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "benchmark_telemetry",
                "message": message,
            },
        )
        failure.text = "\n".join(f"{key}={format_summary_value(value)}" for key, value in sorted(finding.items()))

    for index, finding in enumerate(launch_sequence_findings, 1):
        metric = str(finding.get("metric", "unknown"))
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "qemu_prompt_bench.launch_sequence",
                "name": f"{metric}:{index}",
            },
        )
        message = (
            f"{metric}={format_summary_value(finding.get('value'))} "
            f"limit={format_summary_value(finding.get('limit'))}"
        )
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "benchmark_launch_sequence",
                "message": message,
            },
        )
        failure.text = "\n".join(f"{key}={format_summary_value(value)}" for key, value in sorted(finding.items()))

    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="TempleOS disk image to boot in QEMU")
    parser.add_argument("--prompts", type=Path, required=True, help="Prompt file: JSON, JSONL, or text split by ---")
    parser.add_argument("--qemu-bin", default="qemu-system-x86_64")
    parser.add_argument(
        "--qemu-arg",
        action="append",
        default=[],
        help="Extra QEMU argument; repeat per token. Use --qemu-arg=-m for values beginning with '-'.",
    )
    parser.add_argument(
        "--qemu-args-file",
        action="append",
        type=Path,
        default=[],
        help="Local file of extra QEMU args: JSON string array or shell-style text with # comments",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Run each prompt this many times before measured repeats; warmups are recorded separately",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Run each prompt this many times")
    parser.add_argument(
        "--max-launches",
        type=int,
        default=None,
        help="Fail before launching QEMU if prompts * (warmup + repeat) exceeds this count",
    )
    parser.add_argument(
        "--min-prompt-count",
        type=int,
        default=None,
        help="Fail before launching QEMU if the prompt suite has fewer prompts",
    )
    parser.add_argument(
        "--max-suite-cv-pct",
        type=float,
        default=None,
        help="Fail if measured suite tok/s coefficient of variation exceeds this percentage",
    )
    parser.add_argument(
        "--max-prompt-cv-pct",
        type=float,
        default=None,
        help="Fail if any prompt tok/s coefficient of variation exceeds this percentage",
    )
    parser.add_argument(
        "--max-suite-iqr-pct",
        type=float,
        default=None,
        help="Fail if measured suite tok/s interquartile spread exceeds this percentage",
    )
    parser.add_argument(
        "--max-prompt-iqr-pct",
        type=float,
        default=None,
        help="Fail if any prompt tok/s interquartile spread exceeds this percentage",
    )
    parser.add_argument("--require-tokens", action="store_true", help="Fail if any measured run omits token count")
    parser.add_argument("--require-tok-per-s", action="store_true", help="Fail if any measured run omits tok/s")
    parser.add_argument("--require-memory", action="store_true", help="Fail if any measured run omits memory telemetry")
    parser.add_argument("--require-ttft-us", action="store_true", help="Fail if any measured run omits TTFT telemetry")
    parser.add_argument("--min-tokens", type=int, default=None, help="Fail if any measured run emits fewer tokens")
    parser.add_argument(
        "--min-total-tokens",
        type=int,
        default=None,
        help="Fail if the measured suite emits fewer total tokens",
    )
    parser.add_argument("--min-tok-per-s", type=float, default=None, help="Fail if any measured run is below tok/s")
    parser.add_argument(
        "--min-wall-tok-per-s",
        type=float,
        default=None,
        help="Fail if any measured run is below host wall-clock tok/s",
    )
    parser.add_argument(
        "--max-memory-bytes",
        type=int,
        default=None,
        help="Fail if any measured run exceeds peak memory bytes or omits memory telemetry",
    )
    parser.add_argument("--max-ttft-us", type=int, default=None, help="Fail if any measured run exceeds TTFT in us")
    parser.add_argument(
        "--max-host-overhead-us",
        type=int,
        default=None,
        help="Fail if any measured run exceeds host orchestration overhead in us",
    )
    parser.add_argument(
        "--max-host-overhead-pct",
        type=float,
        default=None,
        help="Fail if any measured run host overhead exceeds this percentage of guest elapsed time",
    )
    parser.add_argument(
        "--max-wall-timeout-pct",
        type=float,
        default=None,
        help="Fail if any measured run consumes more than this percentage of its timeout budget",
    )
    parser.add_argument(
        "--min-host-child-tok-per-cpu-s",
        type=float,
        default=None,
        help="Fail if any measured run is below host child-process tokens per CPU second",
    )
    parser.add_argument(
        "--min-tokens-per-prompt-byte",
        type=float,
        default=None,
        help="Fail if any measured run emits fewer tokens per prompt input byte",
    )
    parser.add_argument(
        "--require-host-child-rss",
        action="store_true",
        help="Fail if any measured run omits host-observed child peak RSS telemetry",
    )
    parser.add_argument(
        "--max-host-child-rss-bytes",
        type=int,
        default=None,
        help="Fail if any measured run exceeds host-observed child peak RSS bytes",
    )
    parser.add_argument(
        "--max-memory-bytes-per-token",
        type=float,
        default=None,
        help="Fail if any measured run exceeds guest-reported memory bytes per emitted token",
    )
    parser.add_argument(
        "--max-serial-output-bytes",
        type=int,
        default=None,
        help="Fail if any measured run emits more combined stdout/stderr serial bytes",
    )
    parser.add_argument(
        "--max-serial-output-lines",
        type=int,
        default=None,
        help="Fail if any measured run emits more combined stdout/stderr serial lines",
    )
    parser.add_argument(
        "--require-guest-prompt-sha256-match",
        action="store_true",
        help="Fail if guest-reported prompt SHA-256 is missing or differs from the host prompt",
    )
    parser.add_argument(
        "--require-guest-prompt-bytes-match",
        action="store_true",
        help="Fail if guest-reported prompt byte count is missing or differs from the host prompt",
    )
    parser.add_argument(
        "--require-expected-tokens-match",
        action="store_true",
        help="Fail if a prompt declares expected_tokens and the measured run emits a different token count",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--profile", default="default")
    parser.add_argument("--model", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument(
        "--hash-image",
        action="store_true",
        help="Record a SHA-256 digest for the QEMU disk image in benchmark artifacts",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print command without launching QEMU")
    parser.add_argument("qemu_args", nargs=argparse.REMAINDER, help="Extra QEMU arguments after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        print("error: --repeat must be >= 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("error: --warmup must be >= 0", file=sys.stderr)
        return 2
    if args.max_launches is not None and args.max_launches < 0:
        print("error: --max-launches must be >= 0", file=sys.stderr)
        return 2
    if args.min_prompt_count is not None and args.min_prompt_count < 0:
        print("error: --min-prompt-count must be >= 0", file=sys.stderr)
        return 2
    if args.max_suite_cv_pct is not None and args.max_suite_cv_pct < 0:
        print("error: --max-suite-cv-pct must be >= 0", file=sys.stderr)
        return 2
    if args.max_prompt_cv_pct is not None and args.max_prompt_cv_pct < 0:
        print("error: --max-prompt-cv-pct must be >= 0", file=sys.stderr)
        return 2
    if args.max_suite_iqr_pct is not None and args.max_suite_iqr_pct < 0:
        print("error: --max-suite-iqr-pct must be >= 0", file=sys.stderr)
        return 2
    if args.max_prompt_iqr_pct is not None and args.max_prompt_iqr_pct < 0:
        print("error: --max-prompt-iqr-pct must be >= 0", file=sys.stderr)
        return 2
    if args.min_tokens is not None and args.min_tokens < 0:
        print("error: --min-tokens must be >= 0", file=sys.stderr)
        return 2
    if args.min_total_tokens is not None and args.min_total_tokens < 0:
        print("error: --min-total-tokens must be >= 0", file=sys.stderr)
        return 2
    if args.min_tok_per_s is not None and args.min_tok_per_s < 0:
        print("error: --min-tok-per-s must be >= 0", file=sys.stderr)
        return 2
    if args.min_wall_tok_per_s is not None and args.min_wall_tok_per_s < 0:
        print("error: --min-wall-tok-per-s must be >= 0", file=sys.stderr)
        return 2
    if args.max_memory_bytes is not None and args.max_memory_bytes < 0:
        print("error: --max-memory-bytes must be >= 0", file=sys.stderr)
        return 2
    if args.max_ttft_us is not None and args.max_ttft_us < 0:
        print("error: --max-ttft-us must be >= 0", file=sys.stderr)
        return 2
    if args.max_host_overhead_us is not None and args.max_host_overhead_us < 0:
        print("error: --max-host-overhead-us must be >= 0", file=sys.stderr)
        return 2
    if args.max_host_overhead_pct is not None and args.max_host_overhead_pct < 0:
        print("error: --max-host-overhead-pct must be >= 0", file=sys.stderr)
        return 2
    if args.max_wall_timeout_pct is not None and args.max_wall_timeout_pct < 0:
        print("error: --max-wall-timeout-pct must be >= 0", file=sys.stderr)
        return 2
    if args.min_host_child_tok_per_cpu_s is not None and args.min_host_child_tok_per_cpu_s < 0:
        print("error: --min-host-child-tok-per-cpu-s must be >= 0", file=sys.stderr)
        return 2
    if args.min_tokens_per_prompt_byte is not None and args.min_tokens_per_prompt_byte < 0:
        print("error: --min-tokens-per-prompt-byte must be >= 0", file=sys.stderr)
        return 2
    if args.max_host_child_rss_bytes is not None and args.max_host_child_rss_bytes < 0:
        print("error: --max-host-child-rss-bytes must be >= 0", file=sys.stderr)
        return 2
    if args.max_memory_bytes_per_token is not None and args.max_memory_bytes_per_token < 0:
        print("error: --max-memory-bytes-per-token must be >= 0", file=sys.stderr)
        return 2
    if args.max_serial_output_bytes is not None and args.max_serial_output_bytes < 0:
        print("error: --max-serial-output-bytes must be >= 0", file=sys.stderr)
        return 2
    if args.max_serial_output_lines is not None and args.max_serial_output_lines < 0:
        print("error: --max-serial-output-lines must be >= 0", file=sys.stderr)
        return 2

    try:
        root = Path(__file__).resolve().parents[1]
        prompts = load_prompt_cases(args.prompts)
        validate_prompt_count(prompts, args.min_prompt_count)
        validate_launch_budget(
            prompts,
            warmup=args.warmup,
            repeat=args.repeat,
            max_launches=args.max_launches,
        )
        trailing_qemu_args = args.qemu_args[1:] if args.qemu_args[:1] == ["--"] else args.qemu_args
        file_qemu_args = load_qemu_args_files(args.qemu_args_file)
        command = build_command(args.qemu_bin, args.image, file_qemu_args + args.qemu_arg + trailing_qemu_args)
        image = input_file_metadata(args.image, include_sha256=args.hash_image)
        qemu_args_files = qemu_args_files_metadata(args.qemu_args_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        report = dry_run_payload(
            command=command,
            prompts_path=args.prompts,
            prompts=prompts,
            warmup=args.warmup,
            repeat=args.repeat,
            profile=args.profile,
            model=args.model,
            quantization=args.quantization,
            commit=git_commit(root),
            max_launches=args.max_launches,
            min_prompt_count=args.min_prompt_count,
            environment=host_environment(args.qemu_bin),
            image=image,
            qemu_args_files=qemu_args_files,
        )
        output = write_dry_run_report(report, args.output_dir)
        report["dry_run_report"] = str(output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    metadata = {
        "profile": args.profile,
        "model": args.model,
        "quantization": args.quantization,
        "commit": git_commit(root),
    }
    launch_plan = dry_run_launch_plan(prompts, warmup=args.warmup, repeat=args.repeat)
    warmups: list[BenchRun] = []
    runs: list[BenchRun] = []
    for planned in launch_plan:
        prompt_index = int(planned["prompt_index"]) - 1
        phase = str(planned["phase"])
        run = run_prompt(
            command,
            prompts[prompt_index],
            args.timeout,
            metadata,
            phase=phase,
            launch_index=int(planned["launch_index"]),
            iteration=int(planned["iteration"]),
        )
        if phase == "warmup":
            warmups.append(run)
        else:
            runs.append(run)
    output = write_report(
        runs,
        args.output_dir,
        prompt_suite=prompt_suite_metadata(args.prompts, prompts),
        launch_plan=launch_plan,
        warmups=warmups,
        max_suite_cv_pct=args.max_suite_cv_pct,
        max_prompt_cv_pct=args.max_prompt_cv_pct,
        max_suite_iqr_pct=args.max_suite_iqr_pct,
        max_prompt_iqr_pct=args.max_prompt_iqr_pct,
        require_tokens=args.require_tokens,
        require_tok_per_s=args.require_tok_per_s,
        require_memory=args.require_memory,
        require_ttft_us=args.require_ttft_us,
        min_tokens=args.min_tokens,
        min_total_tokens=args.min_total_tokens,
        min_tok_per_s=args.min_tok_per_s,
        min_wall_tok_per_s=args.min_wall_tok_per_s,
        max_memory_bytes=args.max_memory_bytes,
        max_ttft_us=args.max_ttft_us,
        max_host_overhead_us=args.max_host_overhead_us,
        max_host_overhead_pct=args.max_host_overhead_pct,
        max_wall_timeout_pct=args.max_wall_timeout_pct,
        min_host_child_tok_per_cpu_s=args.min_host_child_tok_per_cpu_s,
        min_tokens_per_prompt_byte=args.min_tokens_per_prompt_byte,
        require_host_child_rss=args.require_host_child_rss,
        max_host_child_rss_bytes=args.max_host_child_rss_bytes,
        max_memory_bytes_per_token=args.max_memory_bytes_per_token,
        max_serial_output_bytes=args.max_serial_output_bytes,
        max_serial_output_lines=args.max_serial_output_lines,
        require_guest_prompt_sha256_match=args.require_guest_prompt_sha256_match,
        require_guest_prompt_bytes_match=args.require_guest_prompt_bytes_match,
        require_expected_tokens_match=args.require_expected_tokens_match,
        max_launches=args.max_launches,
        min_prompt_count=args.min_prompt_count,
        environment=host_environment(args.qemu_bin),
        image=image,
        qemu_args_files=qemu_args_files,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    print(f"wrote_json={output}")
    print(f"status={report['status']}")
    print(f"variability_findings={len(report['variability_findings'])}")
    print(f"telemetry_findings={len(report['telemetry_findings'])}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
