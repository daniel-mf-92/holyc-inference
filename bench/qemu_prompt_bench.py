#!/usr/bin/env python3
"""Host-side QEMU prompt benchmark runner for the HolyC inference engine.

The runner launches QEMU once per prompt, captures serial output, extracts token
timing records, and writes normalized results under bench/results. Networking is
always disabled with `-nic none`, and conflicting network flags are rejected.
"""

from __future__ import annotations

import argparse
import csv
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


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    prompt: str


@dataclass(frozen=True)
class BenchRun:
    benchmark: str
    profile: str
    model: str
    quantization: str
    prompt: str
    prompt_sha256: str
    prompt_bytes: int
    iteration: int
    commit: str
    timestamp: str
    tokens: int | None
    elapsed_us: int
    wall_elapsed_us: int
    host_overhead_us: int
    host_overhead_pct: float | None
    host_child_user_cpu_us: int | None
    host_child_system_cpu_us: int | None
    host_child_cpu_us: int | None
    host_child_cpu_pct: float | None
    ttft_us: int | None
    tok_per_s: float | None
    wall_tok_per_s: float | None
    us_per_token: float | None
    wall_us_per_token: float | None
    memory_bytes: int | None
    returncode: int
    timed_out: bool
    command: list[str]
    command_sha256: str
    stdout_tail: str
    stderr_tail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def command_hash(command: list[str]) -> str:
    encoded = json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_bytes(prompt: str) -> int:
    return len(prompt.encode("utf-8"))


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
    }


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
    return PromptCase(prompt_id=prompt_id, prompt=prompt)


def is_network_device_arg(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in NETWORK_DEVICE_MARKERS)


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


def host_overhead_pct(host_overhead_us: int, elapsed_us: int) -> float | None:
    if elapsed_us <= 0:
        return None
    return host_overhead_us * 100.0 / elapsed_us


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
    iteration: int = 1,
) -> BenchRun:
    started = time.monotonic_ns()
    child_cpu_before = child_rusage()
    env = os.environ.copy()
    env["HOLYC_BENCH_PROMPT"] = prompt_case.prompt
    env["HOLYC_BENCH_PROMPT_ID"] = prompt_case.prompt_id

    timed_out = False
    try:
        completed = subprocess.run(
            command,
            input=prompt_case.prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            env=env,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

    wall_elapsed_us = max(1, (time.monotonic_ns() - started) // 1000)
    child_cpu_after = child_rusage()
    child_cpu = child_cpu_delta_us(child_cpu_before, child_cpu_after)
    child_user_cpu_us = child_cpu[0] if child_cpu else None
    child_system_cpu_us = child_cpu[1] if child_cpu else None
    child_total_cpu_us = sum(child_cpu) if child_cpu else None
    payload = parse_bench_payload(stdout + "\n" + stderr)
    tokens = extract_tokens(payload)
    elapsed_us = extract_elapsed_us(payload, wall_elapsed_us)
    host_overhead_us = wall_elapsed_us - elapsed_us
    ttft_us = extract_ttft_us(payload)
    tok_per_s = extract_tok_per_s(payload, tokens, elapsed_us)
    wall_tok_per_s = derived_tok_per_s(tokens, wall_elapsed_us)
    us_per_token = derived_us_per_token(tokens, elapsed_us)
    wall_us_per_token = derived_us_per_token(tokens, wall_elapsed_us)
    memory_bytes = extract_memory_bytes(payload)

    return BenchRun(
        benchmark="qemu_prompt",
        profile=metadata["profile"],
        model=metadata["model"],
        quantization=metadata["quantization"],
        prompt=prompt_case.prompt_id,
        prompt_sha256=prompt_hash(prompt_case.prompt),
        prompt_bytes=prompt_bytes(prompt_case.prompt),
        iteration=iteration,
        commit=metadata["commit"],
        timestamp=iso_now(),
        tokens=tokens,
        elapsed_us=elapsed_us,
        wall_elapsed_us=wall_elapsed_us,
        host_overhead_us=host_overhead_us,
        host_overhead_pct=host_overhead_pct(host_overhead_us, elapsed_us),
        host_child_user_cpu_us=child_user_cpu_us,
        host_child_system_cpu_us=child_system_cpu_us,
        host_child_cpu_us=child_total_cpu_us,
        host_child_cpu_pct=cpu_pct(child_total_cpu_us, wall_elapsed_us),
        ttft_us=ttft_us,
        tok_per_s=tok_per_s,
        wall_tok_per_s=wall_tok_per_s,
        us_per_token=us_per_token,
        wall_us_per_token=wall_us_per_token,
        memory_bytes=memory_bytes,
        returncode=returncode,
        timed_out=timed_out,
        command=command,
        command_sha256=command_hash(command),
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
        ttft_values = [run.ttft_us for run in prompt_runs if run.ttft_us is not None]
        memory_values = [run.memory_bytes for run in prompt_runs if run.memory_bytes is not None]
        ok_runs = [run for run in prompt_runs if run.returncode == 0 and not run.timed_out]
        summaries.append(
            {
                "prompt": prompt_id,
                "prompt_bytes": prompt_runs[0].prompt_bytes,
                "runs": len(prompt_runs),
                "ok_runs": len(ok_runs),
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
                "ttft_us_median": statistics.median(ttft_values) if ttft_values else None,
                "ttft_us_p95": percentile([float(value) for value in ttft_values], 95.0),
                "tok_per_s_min": min(tok_values) if tok_values else None,
                "tok_per_s_p05": percentile(tok_values, 5.0),
                "tok_per_s_median": statistics.median(tok_values) if tok_values else None,
                "tok_per_s_stdev": sample_stdev(tok_values),
                "tok_per_s_cv_pct": coefficient_of_variation_pct(tok_values),
                "tok_per_s_p05_p95_spread_pct": percentile_spread_pct(tok_values, 5.0, 95.0),
                "tok_per_s_max": max(tok_values) if tok_values else None,
                "wall_tok_per_s_median": statistics.median(wall_tok_values) if wall_tok_values else None,
                "wall_tok_per_s_p95": percentile(wall_tok_values, 95.0),
                "us_per_token_median": statistics.median(us_per_token_values)
                if us_per_token_values
                else None,
                "us_per_token_p95": percentile(us_per_token_values, 95.0),
                "wall_us_per_token_median": statistics.median(wall_us_per_token_values)
                if wall_us_per_token_values
                else None,
                "wall_us_per_token_p95": percentile(wall_us_per_token_values, 95.0),
                "memory_bytes_max": max(memory_values) if memory_values else None,
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


def variability_findings(
    suite: dict[str, Any],
    summaries: list[dict[str, Any]],
    *,
    max_suite_cv_pct: float | None = None,
    max_prompt_cv_pct: float | None = None,
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
    return findings


def telemetry_findings(
    runs: list[BenchRun],
    *,
    require_tokens: bool = False,
    require_tok_per_s: bool = False,
    require_memory: bool = False,
    require_ttft_us: bool = False,
    min_tokens: int | None = None,
    min_tok_per_s: float | None = None,
    min_wall_tok_per_s: float | None = None,
    max_memory_bytes: int | None = None,
    max_ttft_us: int | None = None,
    max_host_overhead_us: int | None = None,
    max_host_overhead_pct: float | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for run in runs:
        base = {
            "scope": "measured_run",
            "prompt": run.prompt,
            "iteration": run.iteration,
        }
        if require_tokens and run.tokens is None:
            findings.append({**base, "metric": "tokens", "value": None, "limit": "present"})
        if min_tokens is not None and (run.tokens is None or run.tokens < min_tokens):
            findings.append({**base, "metric": "tokens", "value": run.tokens, "limit": min_tokens})
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
    return findings


def suite_summary(runs: list[BenchRun]) -> dict[str, Any]:
    tok_values = [run.tok_per_s for run in runs if run.tok_per_s is not None]
    wall_tok_values = [run.wall_tok_per_s for run in runs if run.wall_tok_per_s is not None]
    us_per_token_values = [run.us_per_token for run in runs if run.us_per_token is not None]
    wall_us_per_token_values = [run.wall_us_per_token for run in runs if run.wall_us_per_token is not None]
    elapsed_values = [run.elapsed_us for run in runs if run.elapsed_us > 0]
    host_overhead_values = [run.host_overhead_us for run in runs]
    host_overhead_pct_values = [run.host_overhead_pct for run in runs if run.host_overhead_pct is not None]
    host_child_cpu_values = [run.host_child_cpu_us for run in runs if run.host_child_cpu_us is not None]
    host_child_cpu_pct_values = [run.host_child_cpu_pct for run in runs if run.host_child_cpu_pct is not None]
    ttft_values = [run.ttft_us for run in runs if run.ttft_us is not None]
    memory_values = [run.memory_bytes for run in runs if run.memory_bytes is not None]
    token_values = [run.tokens for run in runs if run.tokens is not None]
    prompts = sorted({run.prompt for run in runs})
    ok_runs = [run for run in runs if run.returncode == 0 and not run.timed_out]
    prompt_byte_values = [run.prompt_bytes for run in runs]

    return {
        "prompts": len(prompts),
        "runs": len(runs),
        "ok_runs": len(ok_runs),
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
        "ttft_us_median": statistics.median(ttft_values) if ttft_values else None,
        "ttft_us_p95": percentile([float(value) for value in ttft_values], 95.0),
        "tok_per_s_min": min(tok_values) if tok_values else None,
        "tok_per_s_p05": percentile(tok_values, 5.0),
        "tok_per_s_median": statistics.median(tok_values) if tok_values else None,
        "tok_per_s_stdev": sample_stdev(tok_values),
        "tok_per_s_cv_pct": coefficient_of_variation_pct(tok_values),
        "tok_per_s_p05_p95_spread_pct": percentile_spread_pct(tok_values, 5.0, 95.0),
        "tok_per_s_p95": percentile(tok_values, 95.0),
        "tok_per_s_max": max(tok_values) if tok_values else None,
        "wall_tok_per_s_median": statistics.median(wall_tok_values) if wall_tok_values else None,
        "wall_tok_per_s_p95": percentile(wall_tok_values, 95.0),
        "us_per_token_median": statistics.median(us_per_token_values) if us_per_token_values else None,
        "us_per_token_p95": percentile(us_per_token_values, 95.0),
        "wall_us_per_token_median": statistics.median(wall_us_per_token_values)
        if wall_us_per_token_values
        else None,
        "wall_us_per_token_p95": percentile(wall_us_per_token_values, 95.0),
        "elapsed_us_p95": percentile([float(value) for value in elapsed_values], 95.0),
        "memory_bytes_max": max(memory_values) if memory_values else None,
    }


def report_status(all_runs: list[BenchRun], findings: list[dict[str, Any]]) -> str:
    runs_ok = all(run.returncode == 0 and not run.timed_out for run in all_runs)
    return "pass" if runs_ok and not findings else "fail"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# QEMU Prompt Benchmark",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Prompt suite: {report.get('prompt_suite', {}).get('suite_sha256', '-')}",
        f"Command SHA256: {report.get('command_sha256', '-')}",
        f"Launch budget: {format_summary_value(report.get('max_launches'))}",
        f"Total launches: {format_summary_value(report.get('planned_total_launches'))}",
        f"Warmup runs: {len(report['warmups'])}",
        f"Runs: {len(report['benchmarks'])}",
        "",
    ]
    suite = report.get("suite_summary") or {}
    if suite:
        lines.extend(
            [
                "## Suite Summary",
                "",
                "| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| {prompts} | {runs} | {ok_runs} | {measured_prompt_bytes_total} | {total_tokens} | {total_elapsed_us} | "
                "{host_overhead_us_median} | {host_overhead_pct_median} | {host_child_cpu_us_median} | {host_child_cpu_pct_median} | {ttft_us_median} | {ttft_us_p95} | {tok_per_s_p05} | {tok_per_s_median} | {tok_per_s_p95} | "
                "{wall_tok_per_s_median} | {wall_tok_per_s_p95} | {us_per_token_median} | {us_per_token_p95} | "
                "{wall_us_per_token_median} | {wall_us_per_token_p95} | {memory_bytes_max} |".format(
                    **{key: format_summary_value(value) for key, value in suite.items()}
                ),
                "",
                "| tok/s stdev | tok/s CV % | tok/s P05-P95 spread % |",
                "| ---: | ---: | ---: |",
                "| {tok_per_s_stdev} | {tok_per_s_cv_pct} | {tok_per_s_p05_p95_spread_pct} |".format(
                    **{key: format_summary_value(value) for key, value in suite.items()}
                ),
                "",
                "## Prompt Summary",
                "",
            ]
        )
    if report["summaries"]:
        lines.append(
            "| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | P05-P95 spread % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for summary in report["summaries"]:
            lines.append(
                "| {prompt} | {prompt_bytes} | {runs} | {ok_runs} | {tokens_median} | {elapsed_us_median} | "
                "{host_overhead_us_median} | {host_overhead_pct_median} | {host_child_cpu_us_median} | {host_child_cpu_pct_median} | {ttft_us_median} | {ttft_us_p95} | {tok_per_s_min} | {tok_per_s_p05} | {tok_per_s_median} | {tok_per_s_stdev} | {tok_per_s_cv_pct} | {tok_per_s_p05_p95_spread_pct} | "
                "{tok_per_s_max} | {wall_tok_per_s_median} | {wall_tok_per_s_p95} | {us_per_token_median} | {us_per_token_p95} | "
                "{wall_us_per_token_median} | {wall_us_per_token_p95} | {memory_bytes_max} |".format(
                    **{key: format_summary_value(value) for key, value in summary.items()}
                )
            )
    else:
        lines.append("No benchmark runs recorded.")

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
                "| Scope | Prompt | Iteration | Metric | Value | Limit |",
                "| --- | --- | ---: | --- | ---: | --- |",
            ]
        )
        for finding in report["telemetry_findings"]:
            lines.append(
                "| {scope} | {prompt} | {iteration} | {metric} | {value} | {limit} |".format(
                    **{key: format_summary_value(value) for key, value in finding.items()}
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
    lines = [
        "# QEMU Prompt Benchmark Dry Run",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Prompt suite: {prompt_suite.get('suite_sha256', '-')}",
        f"Command SHA256: {report.get('command_sha256', '-')}",
        f"Prompt count: {report['prompt_count']}",
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
    max_launches: int | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_count = len(prompts)
    planned_warmups = prompt_count * warmup
    planned_measured = prompt_count * repeat
    planned_total = planned_warmups + planned_measured
    return {
        "generated_at": iso_now(),
        "status": "planned",
        "command": command,
        "command_sha256": command_hash(command),
        "prompt_count": prompt_count,
        "prompt_suite": prompt_suite_metadata(prompts_path, prompts),
        "environment": environment or {},
        "warmup": warmup,
        "repeat": repeat,
        "max_launches": max_launches,
        "planned_warmup_launches": planned_warmups,
        "planned_measured_launches": planned_measured,
        "planned_total_launches": planned_total,
    }


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
    latest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_md.write_text(markdown_dry_run_report(report), encoding="utf-8")
    stamped = output_dir / f"qemu_prompt_bench_dry_run_{report['generated_at'].replace(':', '').replace('-', '')}.json"
    stamped.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return latest


def write_report(
    runs: list[BenchRun],
    output_dir: Path,
    *,
    prompt_suite: dict[str, Any] | None = None,
    warmups: list[BenchRun] | None = None,
    max_suite_cv_pct: float | None = None,
    max_prompt_cv_pct: float | None = None,
    require_tokens: bool = False,
    require_tok_per_s: bool = False,
    require_memory: bool = False,
    require_ttft_us: bool = False,
    min_tokens: int | None = None,
    min_tok_per_s: float | None = None,
    min_wall_tok_per_s: float | None = None,
    max_memory_bytes: int | None = None,
    max_ttft_us: int | None = None,
    max_host_overhead_us: int | None = None,
    max_host_overhead_pct: float | None = None,
    max_launches: int | None = None,
    environment: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_runs = warmups or []
    all_runs = warmup_runs + runs
    suite = suite_summary(runs)
    summaries = summarize_runs(runs)
    findings = variability_findings(
        suite,
        summaries,
        max_suite_cv_pct=max_suite_cv_pct,
        max_prompt_cv_pct=max_prompt_cv_pct,
    )
    telemetry = telemetry_findings(
        runs,
        require_tokens=require_tokens,
        require_tok_per_s=require_tok_per_s,
        require_memory=require_memory,
        require_ttft_us=require_ttft_us,
        min_tokens=min_tokens,
        min_tok_per_s=min_tok_per_s,
        min_wall_tok_per_s=min_wall_tok_per_s,
        max_memory_bytes=max_memory_bytes,
        max_ttft_us=max_ttft_us,
        max_host_overhead_us=max_host_overhead_us,
        max_host_overhead_pct=max_host_overhead_pct,
    )
    report = {
        "generated_at": iso_now(),
        "status": report_status(all_runs, findings + telemetry),
        "prompt_suite": prompt_suite or {},
        "environment": environment or {},
        "command_sha256": command_hash(all_runs[0].command) if all_runs else command_hash([]),
        "max_launches": max_launches,
        "planned_warmup_launches": len(warmup_runs),
        "planned_measured_launches": len(runs),
        "planned_total_launches": len(all_runs),
        "warmups": [asdict(run) for run in warmup_runs],
        "suite_summary": suite,
        "summaries": summaries,
        "variability_gates": {
            "max_suite_cv_pct": max_suite_cv_pct,
            "max_prompt_cv_pct": max_prompt_cv_pct,
        },
        "variability_findings": findings,
        "telemetry_gates": {
            "require_tokens": require_tokens,
            "require_tok_per_s": require_tok_per_s,
            "require_memory": require_memory,
            "require_ttft_us": require_ttft_us,
            "min_tokens": min_tokens,
            "min_tok_per_s": min_tok_per_s,
            "min_wall_tok_per_s": min_wall_tok_per_s,
            "max_memory_bytes": max_memory_bytes,
            "max_ttft_us": max_ttft_us,
            "max_host_overhead_us": max_host_overhead_us,
            "max_host_overhead_pct": max_host_overhead_pct,
        },
        "telemetry_findings": telemetry,
        "benchmarks": [asdict(run) for run in runs],
    }
    latest = output_dir / "qemu_prompt_bench_latest.json"
    latest_md = output_dir / "qemu_prompt_bench_latest.md"
    latest_csv = output_dir / "qemu_prompt_bench_latest.csv"
    latest_junit = output_dir / "qemu_prompt_bench_junit_latest.xml"
    latest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_md.write_text(markdown_report(report), encoding="utf-8")
    write_csv_report(runs, latest_csv)
    write_junit_report(runs, warmup_runs, findings, telemetry, latest_junit)
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
        "prompt_bytes",
        "iteration",
        "tokens",
        "elapsed_us",
        "wall_elapsed_us",
        "host_overhead_us",
        "host_overhead_pct",
        "host_child_user_cpu_us",
        "host_child_system_cpu_us",
        "host_child_cpu_us",
        "host_child_cpu_pct",
        "ttft_us",
        "tok_per_s",
        "wall_tok_per_s",
        "us_per_token",
        "wall_us_per_token",
        "memory_bytes",
        "returncode",
        "timed_out",
        "command_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            row = asdict(run)
            writer.writerow({field: row[field] for field in fields})


def write_junit_report(
    runs: list[BenchRun],
    warmups: list[BenchRun],
    variability_findings: list[dict[str, Any]],
    telemetry_findings: list[dict[str, Any]],
    path: Path,
) -> None:
    all_runs = [("warmup", run) for run in warmups] + [("measured", run) for run in runs]
    failed_runs = [run for _, run in all_runs if run.returncode != 0 or run.timed_out]
    failures = len(failed_runs) + len(variability_findings) + len(telemetry_findings)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_bench",
            "tests": str(len(all_runs) + len(variability_findings) + len(telemetry_findings)),
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
                "name": f"{run.profile}:{run.model}:{run.quantization}:{run.prompt}:{run.iteration}",
            },
        )
        if run.returncode != 0 or run.timed_out:
            message = f"returncode={run.returncode} timed_out={run.timed_out}"
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
                f"prompt={run.prompt}\n"
                f"iteration={run.iteration}\n"
                f"returncode={run.returncode}\n"
                f"timed_out={run.timed_out}\n"
                f"tokens={format_summary_value(run.tokens)}\n"
                f"elapsed_us={run.elapsed_us}\n"
                f"wall_elapsed_us={run.wall_elapsed_us}\n"
                f"host_overhead_us={run.host_overhead_us}\n"
                f"host_overhead_pct={format_summary_value(run.host_overhead_pct)}\n"
                f"host_child_user_cpu_us={format_summary_value(run.host_child_user_cpu_us)}\n"
                f"host_child_system_cpu_us={format_summary_value(run.host_child_system_cpu_us)}\n"
                f"host_child_cpu_us={format_summary_value(run.host_child_cpu_us)}\n"
                f"host_child_cpu_pct={format_summary_value(run.host_child_cpu_pct)}\n"
                f"ttft_us={format_summary_value(run.ttft_us)}\n"
                f"tok_per_s={format_summary_value(run.tok_per_s)}\n"
                f"wall_tok_per_s={format_summary_value(run.wall_tok_per_s)}\n"
                f"us_per_token={format_summary_value(run.us_per_token)}\n"
                f"wall_us_per_token={format_summary_value(run.wall_us_per_token)}\n"
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
    parser.add_argument("--require-tokens", action="store_true", help="Fail if any measured run omits token count")
    parser.add_argument("--require-tok-per-s", action="store_true", help="Fail if any measured run omits tok/s")
    parser.add_argument("--require-memory", action="store_true", help="Fail if any measured run omits memory telemetry")
    parser.add_argument("--require-ttft-us", action="store_true", help="Fail if any measured run omits TTFT telemetry")
    parser.add_argument("--min-tokens", type=int, default=None, help="Fail if any measured run emits fewer tokens")
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
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--profile", default="default")
    parser.add_argument("--model", default="")
    parser.add_argument("--quantization", default="")
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
    if args.max_suite_cv_pct is not None and args.max_suite_cv_pct < 0:
        print("error: --max-suite-cv-pct must be >= 0", file=sys.stderr)
        return 2
    if args.max_prompt_cv_pct is not None and args.max_prompt_cv_pct < 0:
        print("error: --max-prompt-cv-pct must be >= 0", file=sys.stderr)
        return 2
    if args.min_tokens is not None and args.min_tokens < 0:
        print("error: --min-tokens must be >= 0", file=sys.stderr)
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

    try:
        root = Path(__file__).resolve().parents[1]
        prompts = load_prompt_cases(args.prompts)
        validate_launch_budget(
            prompts,
            warmup=args.warmup,
            repeat=args.repeat,
            max_launches=args.max_launches,
        )
        trailing_qemu_args = args.qemu_args[1:] if args.qemu_args[:1] == ["--"] else args.qemu_args
        file_qemu_args = load_qemu_args_files(args.qemu_args_file)
        command = build_command(args.qemu_bin, args.image, file_qemu_args + args.qemu_arg + trailing_qemu_args)
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
            max_launches=args.max_launches,
            environment=host_environment(args.qemu_bin),
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
    warmups = [
        run_prompt(command, prompt_case, args.timeout, metadata, iteration=iteration)
        for prompt_case in prompts
        for iteration in range(1, args.warmup + 1)
    ]
    runs = [
        run_prompt(command, prompt_case, args.timeout, metadata, iteration=iteration)
        for prompt_case in prompts
        for iteration in range(1, args.repeat + 1)
    ]
    output = write_report(
        runs,
        args.output_dir,
        prompt_suite=prompt_suite_metadata(args.prompts, prompts),
        warmups=warmups,
        max_suite_cv_pct=args.max_suite_cv_pct,
        max_prompt_cv_pct=args.max_prompt_cv_pct,
        require_tokens=args.require_tokens,
        require_tok_per_s=args.require_tok_per_s,
        require_memory=args.require_memory,
        require_ttft_us=args.require_ttft_us,
        min_tokens=args.min_tokens,
        min_tok_per_s=args.min_tok_per_s,
        min_wall_tok_per_s=args.min_wall_tok_per_s,
        max_memory_bytes=args.max_memory_bytes,
        max_ttft_us=args.max_ttft_us,
        max_host_overhead_us=args.max_host_overhead_us,
        max_host_overhead_pct=args.max_host_overhead_pct,
        max_launches=args.max_launches,
        environment=host_environment(args.qemu_bin),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    print(f"wrote_json={output}")
    print(f"status={report['status']}")
    print(f"variability_findings={len(report['variability_findings'])}")
    print(f"telemetry_findings={len(report['telemetry_findings'])}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
