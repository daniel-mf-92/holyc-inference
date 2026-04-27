#!/usr/bin/env python3
"""Tests for host-side QEMU prompt benchmark tooling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_bench


def test_build_command_forces_air_gap(tmp_path: Path) -> None:
    image = tmp_path / "temple.img"

    command = qemu_prompt_bench.build_command("qemu-system-x86_64", image, ["-m", "512M"])

    assert command[:3] == ["qemu-system-x86_64", "-nic", "none"]
    assert "-serial" in command
    assert f"file={image},format=raw,if=ide" in command


def test_network_args_are_rejected(tmp_path: Path) -> None:
    image = tmp_path / "temple.img"

    try:
        qemu_prompt_bench.build_command("qemu-system-x86_64", image, ["-netdev", "user,id=n0"])
    except ValueError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("expected network argument rejection")

    try:
        qemu_prompt_bench.build_command("qemu-system-x86_64", image, ["-device", "virtio-net-pci"])
    except ValueError as exc:
        assert "network device" in str(exc)
    else:
        raise AssertionError("expected network device rejection")

    for device in ("ne2k_pci", "pcnet", "usb-net", "e1000e"):
        try:
            qemu_prompt_bench.build_command("qemu-system-x86_64", image, ["-device", device])
        except ValueError as exc:
            assert "network device" in str(exc)
        else:
            raise AssertionError(f"expected network device rejection for {device}")

    try:
        qemu_prompt_bench.build_command("qemu-system-x86_64", image, ["-device=rtl8139"])
    except ValueError as exc:
        assert "network device" in str(exc)
    else:
        raise AssertionError("expected network device rejection")


def test_load_prompt_cases_json_and_text(tmp_path: Path) -> None:
    prompt_json = tmp_path / "prompts.json"
    prompt_text = tmp_path / "prompts.txt"
    prompt_json.write_text(json.dumps({"prompts": [{"id": "arc-1", "prompt": "Question?"}]}), encoding="utf-8")
    prompt_text.write_text("first\n---\nsecond\n", encoding="utf-8")

    json_cases = qemu_prompt_bench.load_prompt_cases(prompt_json)
    text_cases = qemu_prompt_bench.load_prompt_cases(prompt_text)

    assert json_cases[0].prompt_id == "arc-1"
    assert json_cases[0].prompt == "Question?"
    assert [case.prompt for case in text_cases] == ["first", "second"]


def test_run_prompt_parses_bench_result_json(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    fake_qemu.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
prompt = sys.stdin.read()
assert prompt == os.environ["HOLYC_BENCH_PROMPT"]
print("BENCH_RESULT: " + json.dumps({"tokens": 64, "elapsed_us": 250000, "tok_per_s": 256.0, "max_rss_kib": 4096}))
""",
        encoding="utf-8",
    )
    fake_qemu.chmod(0o755)

    command = [sys.executable, str(fake_qemu), "-nic", "none"]
    run = qemu_prompt_bench.run_prompt(
        command,
        qemu_prompt_bench.PromptCase("smoke", "Hello TempleOS"),
        timeout=5.0,
        metadata={"profile": "secure-local", "model": "tiny", "quantization": "Q4_0", "commit": "abc"},
    )

    assert run.returncode == 0
    assert run.tokens == 64
    assert run.elapsed_us == 250000
    assert run.tok_per_s == 256.0
    assert run.memory_bytes == 4194304
    assert run.prompt == "smoke"
    assert run.profile == "secure-local"


def test_extract_memory_bytes_accepts_common_units() -> None:
    assert qemu_prompt_bench.extract_memory_bytes({"memory_bytes": 123}) == 123
    assert qemu_prompt_bench.extract_memory_bytes({"max_rss_kib": 2}) == 2048
    assert qemu_prompt_bench.extract_memory_bytes({"peak_memory_kb": 2}) == 2000
    assert qemu_prompt_bench.extract_memory_bytes({"memory_mib": 1.5}) == 1572864
    assert qemu_prompt_bench.extract_memory_bytes({"rss_mb": 1.5}) == 1500000


def test_cli_dry_run_validates_without_launching_qemu(tmp_path: Path, capsys) -> None:
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"A"}\n', encoding="utf-8")

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--",
            "-m",
            "256M",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    report = json.loads((output_dir / "qemu_prompt_bench_dry_run_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_dry_run_latest.md").read_text(encoding="utf-8")
    assert payload["prompt_count"] == 1
    assert payload["prompt_suite"]["prompt_count"] == 1
    assert len(payload["prompt_suite"]["suite_sha256"]) == 64
    assert payload["planned_warmup_launches"] == 1
    assert payload["planned_measured_launches"] == 2
    assert payload["planned_total_launches"] == 3
    assert payload["command"][1:3] == ["-nic", "none"]
    assert payload["dry_run_report"] == str(output_dir / "qemu_prompt_bench_dry_run_latest.json")
    assert report["status"] == "planned"
    assert report["command"][1:3] == ["-nic", "none"]
    assert "QEMU Prompt Benchmark Dry Run" in markdown
    assert "Total launches: 3" in markdown


def test_cli_writes_result_file_with_fake_qemu(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.txt"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text("Measure this prompt.\n", encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
print("tokens=32 elapsed_us=100000 memory_kib=8192")
""",
        encoding="utf-8",
    )
    fake_qemu.chmod(0o755)

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--qemu-bin",
            str(fake_qemu),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["prompt_suite"]["source"] == str(prompts)
    assert report["prompt_suite"]["prompt_count"] == 1
    assert report["benchmarks"][0]["tokens"] == 32
    assert report["benchmarks"][0]["tok_per_s"] == 320.0
    assert report["benchmarks"][0]["memory_bytes"] == 8388608
    assert report["benchmarks"][0]["command"][1:3] == ["-nic", "none"]
    csv_report = (output_dir / "qemu_prompt_bench_latest.csv").read_text(encoding="utf-8")
    assert "benchmark,profile,model,quantization,prompt" in csv_report
    assert "qemu_prompt,default,,,prompt-1" in csv_report


def test_cli_repeat_writes_prompt_summary_and_markdown(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text(
        '{"prompt_id":"one","prompt":"First"}\n{"prompt_id":"two","prompt":"Second"}\n',
        encoding="utf-8",
    )
    fake_qemu.write_text(
        """#!/usr/bin/env python3
import os
prompt_id = os.environ["HOLYC_BENCH_PROMPT_ID"]
tokens = 20 if prompt_id == "one" else 40
memory_bytes = 1000 if prompt_id == "one" else 2000
print(f"tokens={tokens} elapsed_us=100000 memory_bytes={memory_bytes}")
""",
        encoding="utf-8",
    )
    fake_qemu.chmod(0o755)

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--qemu-bin",
            str(fake_qemu),
            "--output-dir",
            str(output_dir),
            "--repeat",
            "3",
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_latest.md").read_text(encoding="utf-8")

    assert len(report["benchmarks"]) == 6
    assert report["suite_summary"]["prompts"] == 2
    assert len(report["prompt_suite"]["suite_sha256"]) == 64
    assert report["suite_summary"]["runs"] == 6
    assert report["suite_summary"]["ok_runs"] == 6
    assert report["suite_summary"]["total_tokens"] == 180
    assert report["suite_summary"]["total_elapsed_us"] == 600000
    assert report["suite_summary"]["tok_per_s_median"] == 300.0
    assert report["suite_summary"]["tok_per_s_p95"] == 400.0
    assert report["suite_summary"]["memory_bytes_max"] == 2000
    assert [run["iteration"] for run in report["benchmarks"][:3]] == [1, 2, 3]
    assert report["summaries"][0]["prompt"] == "one"
    assert report["summaries"][0]["runs"] == 3
    assert report["summaries"][0]["tok_per_s_median"] == 200.0
    assert report["summaries"][0]["memory_bytes_max"] == 1000
    assert "QEMU Prompt Benchmark" in markdown
    assert f"Prompt suite: {report['prompt_suite']['suite_sha256']}" in markdown
    assert "| 2 | 6 | 6 | 180 | 600000 | 300.000 | 400.000 | 2000 |" in markdown
    assert "| one | 3 | 3 | 20 | 100000 | 200.000 | 200.000 | 200.000 | 1000 |" in markdown
    csv_report = (output_dir / "qemu_prompt_bench_latest.csv").read_text(encoding="utf-8")
    assert csv_report.count("\n") == 7
    assert ",one," in csv_report
    assert ",two," in csv_report


def test_cli_warmup_records_separately_from_measured_runs(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"Warm up"}\n', encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
print("tokens=10 elapsed_us=100000")
""",
        encoding="utf-8",
    )
    fake_qemu.chmod(0o755)

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--qemu-bin",
            str(fake_qemu),
            "--output-dir",
            str(output_dir),
            "--warmup",
            "2",
            "--repeat",
            "3",
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_latest.md").read_text(encoding="utf-8")
    csv_report = (output_dir / "qemu_prompt_bench_latest.csv").read_text(encoding="utf-8")

    assert len(report["warmups"]) == 2
    assert len(report["benchmarks"]) == 3
    assert report["summaries"][0]["runs"] == 3
    assert "Warmup runs: 2" in markdown
    assert csv_report.count("\n") == 4


def test_cli_variability_gate_fails_noisy_prompt_runs(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"Noisy"}\n', encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
counter = Path(__file__).with_suffix(".count")
iteration = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(iteration))
elapsed_us = 100000 if iteration != 2 else 200000
print(f"tokens=100 elapsed_us={elapsed_us}")
""",
        encoding="utf-8",
    )
    fake_qemu.chmod(0o755)

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--qemu-bin",
            str(fake_qemu),
            "--output-dir",
            str(output_dir),
            "--repeat",
            "3",
            "--max-prompt-cv-pct",
            "10",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_latest.md").read_text(encoding="utf-8")

    assert report["status"] == "fail"
    assert report["variability_gates"]["max_prompt_cv_pct"] == 10.0
    assert report["variability_findings"][0]["scope"] == "prompt"
    assert report["variability_findings"][0]["prompt"] == "one"
    assert report["variability_findings"][0]["metric"] == "tok_per_s_cv_pct"
    assert "Variability Gate Findings" in markdown


def test_cli_rejects_negative_warmup(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    prompts.write_text('{"prompt_id":"one","prompt":"A"}\n', encoding="utf-8")

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--dry-run",
            "--warmup",
            "-1",
        ]
    )

    assert status == 2


def test_cli_rejects_negative_variability_gate(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    prompts.write_text('{"prompt_id":"one","prompt":"A"}\n', encoding="utf-8")

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--dry-run",
            "--max-suite-cv-pct",
            "-1",
        ]
    )

    assert status == 2
