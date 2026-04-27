#!/usr/bin/env python3
"""Host-side QEMU prompt benchmark runner for the HolyC inference engine.

The runner launches QEMU once per prompt, captures serial output, extracts token
timing records, and writes normalized results under bench/results. Networking is
always disabled with `-nic none`, and conflicting network flags are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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
    iteration: int
    commit: str
    timestamp: str
    tokens: int | None
    elapsed_us: int
    wall_elapsed_us: int
    tok_per_s: float | None
    returncode: int
    timed_out: bool
    command: list[str]
    stdout_tail: str
    stderr_tail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


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


def run_prompt(
    command: list[str],
    prompt_case: PromptCase,
    timeout: float,
    metadata: dict[str, str],
    iteration: int = 1,
) -> BenchRun:
    started = time.monotonic_ns()
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
    payload = parse_bench_payload(stdout + "\n" + stderr)
    tokens = extract_tokens(payload)
    elapsed_us = extract_elapsed_us(payload, wall_elapsed_us)
    tok_per_s = extract_tok_per_s(payload, tokens, elapsed_us)

    return BenchRun(
        benchmark="qemu_prompt",
        profile=metadata["profile"],
        model=metadata["model"],
        quantization=metadata["quantization"],
        prompt=prompt_case.prompt_id,
        prompt_sha256=prompt_hash(prompt_case.prompt),
        iteration=iteration,
        commit=metadata["commit"],
        timestamp=iso_now(),
        tokens=tokens,
        elapsed_us=elapsed_us,
        wall_elapsed_us=wall_elapsed_us,
        tok_per_s=tok_per_s,
        returncode=returncode,
        timed_out=timed_out,
        command=command,
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
        elapsed_values = [run.elapsed_us for run in prompt_runs if run.elapsed_us > 0]
        ok_runs = [run for run in prompt_runs if run.returncode == 0 and not run.timed_out]
        summaries.append(
            {
                "prompt": prompt_id,
                "runs": len(prompt_runs),
                "ok_runs": len(ok_runs),
                "tokens_median": statistics.median(
                    [run.tokens for run in prompt_runs if run.tokens is not None]
                )
                if any(run.tokens is not None for run in prompt_runs)
                else None,
                "elapsed_us_median": statistics.median(elapsed_values) if elapsed_values else None,
                "tok_per_s_min": min(tok_values) if tok_values else None,
                "tok_per_s_median": statistics.median(tok_values) if tok_values else None,
                "tok_per_s_max": max(tok_values) if tok_values else None,
            }
        )
    return summaries


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# QEMU Prompt Benchmark",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Runs: {len(report['benchmarks'])}",
        "",
        "## Prompt Summary",
        "",
    ]
    if report["summaries"]:
        lines.append("| Prompt | Runs | OK | Median tokens | Median elapsed us | Min tok/s | Median tok/s | Max tok/s |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for summary in report["summaries"]:
            lines.append(
                "| {prompt} | {runs} | {ok_runs} | {tokens_median} | {elapsed_us_median} | "
                "{tok_per_s_min} | {tok_per_s_median} | {tok_per_s_max} |".format(
                    **{key: format_summary_value(value) for key, value in summary.items()}
                )
            )
    else:
        lines.append("No benchmark runs recorded.")
    return "\n".join(lines) + "\n"


def format_summary_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report(runs: list[BenchRun], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": iso_now(),
        "status": "pass" if all(run.returncode == 0 and not run.timed_out for run in runs) else "fail",
        "summaries": summarize_runs(runs),
        "benchmarks": [asdict(run) for run in runs],
    }
    latest = output_dir / "qemu_prompt_bench_latest.json"
    latest_md = output_dir / "qemu_prompt_bench_latest.md"
    latest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_md.write_text(markdown_report(report), encoding="utf-8")
    stamped = output_dir / f"qemu_prompt_bench_{report['generated_at'].replace(':', '').replace('-', '')}.json"
    stamped.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return latest


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
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--repeat", type=int, default=1, help="Run each prompt this many times")
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

    root = Path(__file__).resolve().parents[1]
    prompts = load_prompt_cases(args.prompts)
    trailing_qemu_args = args.qemu_args[1:] if args.qemu_args[:1] == ["--"] else args.qemu_args
    try:
        command = build_command(args.qemu_bin, args.image, args.qemu_arg + trailing_qemu_args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps({"command": command, "prompt_count": len(prompts)}, indent=2))
        return 0

    metadata = {
        "profile": args.profile,
        "model": args.model,
        "quantization": args.quantization,
        "commit": git_commit(root),
    }
    runs = [
        run_prompt(command, prompt_case, args.timeout, metadata, iteration=iteration)
        for prompt_case in prompts
        for iteration in range(1, args.repeat + 1)
    ]
    output = write_report(runs, args.output_dir)
    print(f"wrote_json={output}")
    print(f"status={'pass' if all(run.returncode == 0 and not run.timed_out for run in runs) else 'fail'}")
    return 0 if all(run.returncode == 0 and not run.timed_out for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
