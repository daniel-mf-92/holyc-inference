#!/usr/bin/env python3
"""Host-side tests for bench/qemu_prompt_bench.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import qemu_prompt_bench as bench


def test_build_command_forces_air_gap_and_serial_stdio(tmp_path: Path) -> None:
    image = tmp_path / "TempleOS.img"
    command = bench.build_command("qemu-system-x86_64", image, ["-m", "512M", "-smp", "1"])

    assert command[:7] == [
        "qemu-system-x86_64",
        "-nic",
        "none",
        "-serial",
        "stdio",
        "-display",
        "none",
    ]
    assert f"file={image},format=raw,if=ide" in command
    assert "-m" in command


@pytest.mark.parametrize(
    "args",
    [
        ["-nic", "user"],
        ["-nic=user"],
        ["-net", "user"],
        ["-net=user"],
        ["-netdev", "user,id=n0"],
        ["-netdev=user,id=n0"],
        ["-device", "e1000"],
        ["-device=virtio-net-pci"],
    ],
)
def test_reject_network_args_blocks_qemu_networking(args: list[str]) -> None:
    with pytest.raises(ValueError):
        bench.reject_network_args(args)


def test_load_prompt_cases_json_jsonl_and_split_text(tmp_path: Path) -> None:
    json_path = tmp_path / "prompts.json"
    json_path.write_text(
        json.dumps({"prompts": [{"id": "a", "prompt": "Alpha"}, "Beta"]}),
        encoding="utf-8",
    )
    assert bench.load_prompt_cases(json_path) == [
        bench.PromptCase(prompt_id="a", prompt="Alpha"),
        bench.PromptCase(prompt_id="prompt-2", prompt="Beta"),
    ]

    jsonl_path = tmp_path / "prompts.jsonl"
    jsonl_path.write_text('{"prompt_id":"c","text":"Gamma"}\n"Delta"\n', encoding="utf-8")
    assert bench.load_prompt_cases(jsonl_path) == [
        bench.PromptCase(prompt_id="c", prompt="Gamma"),
        bench.PromptCase(prompt_id="prompt-2", prompt="Delta"),
    ]

    text_path = tmp_path / "prompts.txt"
    text_path.write_text("One\n---\nTwo\n", encoding="utf-8")
    assert bench.load_prompt_cases(text_path) == [
        bench.PromptCase(prompt_id="prompt-1", prompt="One"),
        bench.PromptCase(prompt_id="prompt-2", prompt="Two"),
    ]


def test_parse_bench_payload_accepts_json_and_key_value() -> None:
    json_payload = bench.parse_bench_payload(
        'boot\nBENCH_RESULT: {"tokens": 12, "elapsed_us": 3000000, "tok_per_s": 4.0}\n'
    )
    assert json_payload["tokens"] == 12
    assert json_payload["tok_per_s"] == 4.0

    kv_payload = bench.parse_bench_payload("tokens=10 elapsed_us=2500000 tok_per_s_milli=4000")
    assert kv_payload == {
        "tokens": "10",
        "elapsed_us": "2500000",
        "tok_per_s_milli": "4000",
    }


def test_extract_elapsed_us_accepts_guest_microseconds_and_milliseconds() -> None:
    assert bench.extract_elapsed_us({"elapsed_us": "2500000"}) == 2500000
    assert bench.extract_elapsed_us({"duration_ms": "12.5"}) == 12500
    assert bench.extract_elapsed_us({}) is None


def test_run_prompt_normalizes_guest_metrics() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; sys.stdin.read(); "
            "print('BENCH_RESULT: {\"tokens\": 20, \"tok_per_s_milli\": 2500}')"
        ),
    ]
    run = bench.run_prompt(
        command,
        bench.PromptCase(prompt_id="smoke", prompt="Count to five."),
        timeout=10,
        metadata={
            "profile": "smoke",
            "model": "TinyLlama",
            "quantization": "Q4_0",
            "commit": "abc123",
        },
    )

    assert run.returncode == 0
    assert run.timed_out is False
    assert run.tokens == 20
    assert run.tok_per_s == 2.5
    assert run.prompt == "smoke"
    assert run.prompt_sha256 == bench.prompt_hash("Count to five.")


def test_write_report_emits_latest_and_timestamped_json(tmp_path: Path) -> None:
    run = bench.BenchRun(
        benchmark="qemu_prompt",
        profile="default",
        model="model",
        quantization="Q8_0",
        prompt="p1",
        prompt_sha256="00",
        commit="abc123",
        timestamp="2026-04-27T00:00:00Z",
        tokens=4,
        elapsed_us=1000000,
        tok_per_s=4.0,
        returncode=0,
        timed_out=False,
        command=["qemu-system-x86_64", "-nic", "none"],
        stdout_tail="",
        stderr_tail="",
    )

    latest = bench.write_report([run], tmp_path)
    assert latest == tmp_path / "qemu_prompt_bench_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["benchmarks"][0]["tok_per_s"] == 4.0
    assert len(list(tmp_path.glob("qemu_prompt_bench_*.json"))) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
