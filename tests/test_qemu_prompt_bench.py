#!/usr/bin/env python3
"""Tests for host-side QEMU prompt benchmark tooling."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
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


def test_qemu_args_file_parsing_and_air_gap_validation(tmp_path: Path) -> None:
    image = tmp_path / "temple.img"
    text_args = tmp_path / "qemu.args"
    json_args = tmp_path / "qemu_args.json"
    bad_args = tmp_path / "bad.args"
    text_args.write_text("-m 512M\n# comment\n-smp 2\n", encoding="utf-8")
    json_args.write_text(json.dumps(["-cpu", "max"]), encoding="utf-8")
    bad_args.write_text("-device virtio-net-pci\n", encoding="utf-8")

    args = qemu_prompt_bench.load_qemu_args_files([text_args, json_args])
    command = qemu_prompt_bench.build_command("qemu-system-x86_64", image, args)

    assert args == ["-m", "512M", "-smp", "2", "-cpu", "max"]
    assert command[1:3] == ["-nic", "none"]
    assert "-cpu" in command

    try:
        qemu_prompt_bench.build_command(
            "qemu-system-x86_64",
            image,
            qemu_prompt_bench.load_qemu_args_files([bad_args]),
        )
    except ValueError as exc:
        assert "network device" in str(exc)
    else:
        raise AssertionError("expected args-file network device rejection")


def test_load_prompt_cases_json_and_text(tmp_path: Path) -> None:
    prompt_json = tmp_path / "prompts.json"
    prompt_text = tmp_path / "prompts.txt"
    prompt_json.write_text(
        json.dumps({"prompts": [{"id": "arc-1", "prompt": "Question?", "expected_tokens": 4}]}),
        encoding="utf-8",
    )
    prompt_text.write_text("first\n---\nsecond\n", encoding="utf-8")

    json_cases = qemu_prompt_bench.load_prompt_cases(prompt_json)
    text_cases = qemu_prompt_bench.load_prompt_cases(prompt_text)

    assert json_cases[0].prompt_id == "arc-1"
    assert json_cases[0].prompt == "Question?"
    assert json_cases[0].expected_tokens == 4
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
    assert run.guest_prompt_sha256 is None
    assert run.guest_prompt_sha256_match is None
    assert run.elapsed_us == 250000
    assert run.timeout_seconds == 5.0
    assert run.wall_timeout_pct is not None
    assert run.wall_timeout_pct > 0
    assert run.host_overhead_us == run.wall_elapsed_us - run.elapsed_us
    assert run.host_overhead_pct is not None
    assert run.tok_per_s == 256.0
    assert run.wall_tok_per_s is not None
    assert run.wall_tok_per_s > 0
    assert run.us_per_token == 3906.25
    assert run.wall_us_per_token is not None
    assert run.wall_us_per_token > 0
    assert run.memory_bytes == 4194304
    assert run.stdout_bytes == len(run.stdout_tail.encode("utf-8"))
    assert run.stderr_bytes == 0
    assert run.serial_output_bytes == run.stdout_bytes
    assert run.prompt == "smoke"
    assert run.prompt_bytes == 14
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
    assert payload["prompt_suite"]["prompt_bytes_total"] == 1
    assert len(payload["prompt_suite"]["suite_sha256"]) == 64
    assert payload["planned_warmup_launches"] == 1
    assert payload["planned_measured_launches"] == 2
    assert payload["planned_total_launches"] == 3
    assert payload["max_launches"] is None
    assert payload["command"][1:3] == ["-nic", "none"]
    assert payload["command_sha256"] == qemu_prompt_bench.command_hash(payload["command"])
    assert len(payload["command_sha256"]) == 64
    assert payload["image"]["path"] == str(image)
    assert payload["image"]["exists"] is False
    assert payload["image"]["sha256"] is None
    assert payload["qemu_args_files"] == []
    assert payload["environment"]["qemu_bin"] == "qemu-system-x86_64"
    assert payload["environment"]["python"]
    assert payload["dry_run_report"] == str(output_dir / "qemu_prompt_bench_dry_run_latest.json")
    assert report["status"] == "planned"
    assert report["command"][1:3] == ["-nic", "none"]
    assert report["command_sha256"] == payload["command_sha256"]
    assert report["environment"]["qemu_bin"] == "qemu-system-x86_64"
    assert "QEMU Prompt Benchmark Dry Run" in markdown
    assert f"Command SHA256: {payload['command_sha256']}" in markdown
    assert "Total launches: 3" in markdown
    assert "## Inputs" in markdown
    assert "Environment" in markdown


def test_cli_dry_run_accepts_qemu_args_file(tmp_path: Path, capsys) -> None:
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    args_file = tmp_path / "qemu.args"
    output_dir = tmp_path / "results"
    image.write_bytes(b"tiny image")
    prompts.write_text('{"prompt_id":"one","prompt":"A"}\n', encoding="utf-8")
    args_file.write_text("-m 384M\n-smp 2\n", encoding="utf-8")

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--qemu-args-file",
            str(args_file),
            "--hash-image",
            "--dry-run",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    report = json.loads((output_dir / "qemu_prompt_bench_dry_run_latest.json").read_text(encoding="utf-8"))
    assert payload["command"][1:3] == ["-nic", "none"]
    assert "-m" in payload["command"]
    assert "384M" in payload["command"]
    assert "-smp" in payload["command"]
    assert payload["image"]["exists"] is True
    assert payload["image"]["size_bytes"] == len(b"tiny image")
    assert payload["image"]["sha256"] == qemu_prompt_bench.file_sha256(image)
    assert payload["qemu_args_files"][0]["path"] == str(args_file)
    assert payload["qemu_args_files"][0]["sha256"] == qemu_prompt_bench.file_sha256(args_file)
    assert report["image"]["sha256"] == payload["image"]["sha256"]


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
    assert report["prompt_suite"]["prompt_bytes_total"] == 20
    assert report["max_launches"] is None
    assert report["planned_total_launches"] == 1
    assert report["benchmarks"][0]["tokens"] == 32
    assert report["benchmarks"][0]["prompt_bytes"] == 20
    assert report["benchmarks"][0]["timeout_seconds"] == 300.0
    assert report["benchmarks"][0]["wall_timeout_pct"] > 0
    assert report["benchmarks"][0]["host_overhead_us"] == (
        report["benchmarks"][0]["wall_elapsed_us"] - report["benchmarks"][0]["elapsed_us"]
    )
    assert report["benchmarks"][0]["host_overhead_pct"] is not None
    assert report["benchmarks"][0]["tok_per_s"] == 320.0
    assert report["benchmarks"][0]["wall_tok_per_s"] > 0
    assert report["benchmarks"][0]["us_per_token"] == 3125.0
    assert report["benchmarks"][0]["wall_us_per_token"] > 0
    assert report["benchmarks"][0]["memory_bytes"] == 8388608
    assert report["benchmarks"][0]["stdout_bytes"] == len(
        report["benchmarks"][0]["stdout_tail"].encode("utf-8")
    )
    assert report["benchmarks"][0]["stderr_bytes"] == 0
    assert report["benchmarks"][0]["serial_output_bytes"] == report["benchmarks"][0]["stdout_bytes"]
    assert report["suite_summary"]["serial_output_bytes_total"] == report["benchmarks"][0]["serial_output_bytes"]
    assert report["suite_summary"]["serial_output_bytes_max"] == report["benchmarks"][0]["serial_output_bytes"]
    assert report["summaries"][0]["serial_output_bytes_total"] == report["benchmarks"][0]["serial_output_bytes"]
    assert report["summaries"][0]["serial_output_bytes_max"] == report["benchmarks"][0]["serial_output_bytes"]
    assert report["benchmarks"][0]["command"][1:3] == ["-nic", "none"]
    assert report["command_sha256"] == report["benchmarks"][0]["command_sha256"]
    assert report["command_sha256"] == qemu_prompt_bench.command_hash(report["benchmarks"][0]["command"])
    assert report["image"]["path"] == str(image)
    assert report["image"]["exists"] is False
    assert report["qemu_args_files"] == []
    assert report["environment"]["qemu_bin"] == str(fake_qemu)
    assert report["environment"]["qemu_path"] == str(fake_qemu)
    assert report["environment"]["qemu_version"] is None
    csv_report = (output_dir / "qemu_prompt_bench_latest.csv").read_text(encoding="utf-8")
    summary_csv_report = (output_dir / "qemu_prompt_bench_summary_latest.csv").read_text(encoding="utf-8")
    assert "prompt_sha256,guest_prompt_sha256,guest_prompt_sha256_match,prompt_bytes" in csv_report
    assert "host_overhead_us,host_overhead_pct" in csv_report
    assert "timeout_seconds,wall_timeout_pct" in csv_report
    assert "wall_tok_per_s" in csv_report
    assert "us_per_token,wall_us_per_token" in csv_report
    assert "stdout_bytes,stderr_bytes,serial_output_bytes" in csv_report
    assert "timed_out,command_sha256" in csv_report
    assert "qemu_prompt,default,,,prompt-1" in csv_report
    assert summary_csv_report.startswith("scope,prompt,prompt_bytes,prompts,runs,ok_runs")
    assert "serial_output_bytes_total,serial_output_bytes_max" in summary_csv_report
    assert "suite,,20,1,1,1,0,0,0,,32," in summary_csv_report
    assert "prompt,prompt-1,20,,1,1,0,0,0,32,," in summary_csv_report


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
            "--max-launches",
            "6",
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
    assert report["max_launches"] == 6
    assert report["planned_total_launches"] == 6
    assert report["suite_summary"]["ok_runs"] == 6
    assert report["suite_summary"]["measured_prompt_bytes_total"] == 33
    assert report["suite_summary"]["prompt_bytes_min"] == 5
    assert report["suite_summary"]["prompt_bytes_max"] == 6
    assert report["suite_summary"]["total_tokens"] == 180
    assert report["suite_summary"]["total_elapsed_us"] == 600000
    assert report["suite_summary"]["host_overhead_us_median"] is not None
    assert report["suite_summary"]["host_overhead_pct_median"] is not None
    assert report["suite_summary"]["tok_per_s_median"] == 300.0
    assert report["suite_summary"]["tok_per_s_p05"] == 200.0
    assert report["suite_summary"]["tok_per_s_p95"] == 400.0
    assert round(report["suite_summary"]["tok_per_s_iqr_pct"], 3) == 66.667
    assert round(report["suite_summary"]["tok_per_s_p05_p95_spread_pct"], 3) == 66.667
    assert report["suite_summary"]["wall_tok_per_s_median"] > 0
    assert report["suite_summary"]["wall_tok_per_s_iqr_pct"] is not None
    assert report["suite_summary"]["wall_tok_per_s_p95"] > 0
    assert report["suite_summary"]["us_per_token_median"] == 3750.0
    assert report["suite_summary"]["us_per_token_p95"] == 5000.0
    assert report["suite_summary"]["wall_us_per_token_median"] > 0
    assert report["suite_summary"]["wall_us_per_token_p95"] > 0
    assert report["suite_summary"]["memory_bytes_max"] == 2000
    assert [run["iteration"] for run in report["benchmarks"][:3]] == [1, 2, 3]
    assert report["summaries"][0]["prompt"] == "one"
    assert report["summaries"][0]["prompt_bytes"] == 5
    assert report["summaries"][0]["runs"] == 3
    assert report["summaries"][0]["tok_per_s_median"] == 200.0
    assert report["summaries"][0]["tok_per_s_p05"] == 200.0
    assert report["summaries"][0]["tok_per_s_iqr_pct"] == 0.0
    assert report["summaries"][0]["tok_per_s_p05_p95_spread_pct"] == 0.0
    assert report["summaries"][0]["host_overhead_us_median"] is not None
    assert report["summaries"][0]["host_overhead_pct_median"] is not None
    assert report["summaries"][0]["wall_tok_per_s_median"] > 0
    assert report["summaries"][0]["wall_tok_per_s_p95"] > 0
    assert report["summaries"][0]["us_per_token_median"] == 5000.0
    assert report["summaries"][0]["us_per_token_p95"] == 5000.0
    assert report["summaries"][0]["wall_us_per_token_median"] > 0
    assert report["summaries"][0]["wall_us_per_token_p95"] > 0
    assert report["summaries"][0]["memory_bytes_max"] == 1000
    assert "QEMU Prompt Benchmark" in markdown
    assert f"Prompt suite: {report['prompt_suite']['suite_sha256']}" in markdown
    assert f"Command SHA256: {report['command_sha256']}" in markdown
    assert "Median host overhead us" in markdown
    assert "P05 tok/s" in markdown
    assert "tok/s IQR %" in markdown
    assert "tok/s P05-P95 spread %" in markdown
    assert "Median wall tok/s" in markdown
    assert "Median us/token" in markdown
    assert "Serial output bytes total" in markdown
    assert "Prompt Serial Output" in markdown
    assert "| 2 | 6 | 6 | 0 | 0 | 0 | 33 | 180 | 600000 |" in markdown
    assert "| one | 5 | 3 | 3 | 0 | 0 | 0 | 20 | 100000 |" in markdown
    csv_report = (output_dir / "qemu_prompt_bench_latest.csv").read_text(encoding="utf-8")
    summary_csv_report = (output_dir / "qemu_prompt_bench_summary_latest.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()
    assert csv_report.count("\n") == 7
    assert summary_csv_report.count("\n") == 4
    assert "suite,,33,2,6,6,0,0,0,,180," in summary_csv_report
    assert "prompt,one,5,,3,3,0,0,0,20," in summary_csv_report
    assert "prompt,two,6,,3,3,0,0,0,40," in summary_csv_report
    assert ",one," in csv_report
    assert ",two," in csv_report
    assert junit_root.attrib["name"] == "holyc_qemu_prompt_bench"
    assert junit_root.attrib["tests"] == "6"
    assert junit_root.attrib["failures"] == "0"


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
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()
    assert junit_root.attrib["failures"] == "1"
    failure = junit_root.find(".//failure")
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_variability"


def test_cli_iqr_variability_gate_fails_noisy_prompt_runs(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"IQR noisy"}\n', encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
counter = Path(__file__).with_suffix(".count")
iteration = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(iteration))
elapsed_us = 100000 if iteration != 1 else 200000
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
            "--max-prompt-iqr-pct",
            "10",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))

    assert report["status"] == "fail"
    assert report["variability_gates"]["max_prompt_iqr_pct"] == 10.0
    assert report["variability_findings"][0]["scope"] == "prompt"
    assert report["variability_findings"][0]["prompt"] == "one"
    assert report["variability_findings"][0]["metric"] == "tok_per_s_iqr_pct"


def test_cli_telemetry_gate_fails_missing_or_low_metrics(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"Sparse telemetry"}\n', encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
print("tokens=4 elapsed_us=100000")
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
            "--require-memory",
            "--min-tokens",
            "8",
            "--min-tok-per-s",
            "100",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_latest.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()
    failures = junit_root.findall(".//failure")

    assert report["status"] == "fail"
    assert report["telemetry_gates"]["require_memory"] is True
    assert report["telemetry_gates"]["min_tokens"] == 8
    assert len(report["telemetry_findings"]) == 3
    assert {finding["metric"] for finding in report["telemetry_findings"]} == {
        "tokens",
        "tok_per_s",
        "memory_bytes",
    }
    assert "Telemetry Gate Findings" in markdown
    assert junit_root.attrib["failures"] == "3"
    assert {failure.attrib["type"] for failure in failures} == {"benchmark_telemetry"}


def test_cli_ttft_gate_fails_missing_or_high_latency(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text(
        '{"prompt_id":"missing","prompt":"Missing TTFT"}\n{"prompt_id":"slow","prompt":"Slow TTFT"}\n',
        encoding="utf-8",
    )
    fake_qemu.write_text(
        """#!/usr/bin/env python3
import os
prompt_id = os.environ["HOLYC_BENCH_PROMPT_ID"]
if prompt_id == "slow":
    print("tokens=12 elapsed_us=100000 ttft_us=75000")
else:
    print("tokens=12 elapsed_us=100000")
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
            "--require-ttft-us",
            "--max-ttft-us",
            "50000",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_latest.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()

    assert report["status"] == "fail"
    assert report["telemetry_gates"]["require_ttft_us"] is True
    assert report["telemetry_gates"]["max_ttft_us"] == 50000
    assert [finding["metric"] for finding in report["telemetry_findings"]] == ["ttft_us", "ttft_us", "ttft_us"]
    assert {finding["prompt"] for finding in report["telemetry_findings"]} == {"missing", "slow"}
    assert "Telemetry Gate Findings" in markdown
    assert junit_root.attrib["failures"] == "3"


def test_cli_host_overhead_gate_fails_noisy_host_timing(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"Host overhead"}\n', encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
print("tokens=12 elapsed_us=1")
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
            "--max-host-overhead-us",
            "0",
            "--max-host-overhead-pct",
            "0",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_latest.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()

    assert report["status"] == "fail"
    assert report["telemetry_gates"]["max_host_overhead_us"] == 0
    assert report["telemetry_gates"]["max_host_overhead_pct"] == 0.0
    assert {finding["metric"] for finding in report["telemetry_findings"]} == {
        "host_overhead_us",
        "host_overhead_pct",
    }
    assert "Telemetry Gate Findings" in markdown
    assert junit_root.attrib["failures"] == "2"


def test_cli_wall_timeout_budget_gate_fails_near_timeout_runs(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"Timeout budget"}\n', encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
print("tokens=12 elapsed_us=100000")
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
            "--timeout",
            "10",
            "--max-wall-timeout-pct",
            "0",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_latest.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()

    assert report["status"] == "fail"
    assert report["telemetry_gates"]["max_wall_timeout_pct"] == 0.0
    assert report["telemetry_findings"][0]["metric"] == "wall_timeout_pct"
    assert report["telemetry_findings"][0]["value"] > 0
    assert "wall_timeout_pct" in markdown
    assert junit_root.attrib["failures"] == "1"


def test_cli_serial_output_gate_fails_verbose_guest_output(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"Verbose serial"}\n', encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
import sys
print("tokens=12 elapsed_us=100000")
print("verbose debug line", file=sys.stderr)
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
            "--max-serial-output-bytes",
            "8",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()

    assert report["status"] == "fail"
    assert report["telemetry_gates"]["max_serial_output_bytes"] == 8
    assert report["benchmarks"][0]["stderr_bytes"] == len("verbose debug line\n".encode("utf-8"))
    assert report["benchmarks"][0]["serial_output_bytes"] > 8
    assert report["telemetry_findings"] == [
        {
            "scope": "measured_run",
            "prompt": "one",
            "iteration": 1,
            "metric": "serial_output_bytes",
            "value": report["benchmarks"][0]["serial_output_bytes"],
            "limit": 8,
        }
    ]
    assert junit_root.attrib["failures"] == "1"


def test_cli_guest_prompt_sha256_gate_fails_missing_or_mismatch(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text(
        '{"prompt_id":"missing","prompt":"Missing hash"}\n{"prompt_id":"bad","prompt":"Bad hash"}\n',
        encoding="utf-8",
    )
    fake_qemu.write_text(
        """#!/usr/bin/env python3
import os
prompt_id = os.environ["HOLYC_BENCH_PROMPT_ID"]
if prompt_id == "bad":
    print("tokens=12 elapsed_us=100000 prompt_sha256=" + ("0" * 64))
else:
    print("tokens=12 elapsed_us=100000")
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
            "--require-guest-prompt-sha256-match",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_latest.md").read_text(encoding="utf-8")
    csv_report = (output_dir / "qemu_prompt_bench_latest.csv").read_text(encoding="utf-8")
    launch_csv = (output_dir / "qemu_prompt_bench_launches_latest.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()

    assert report["status"] == "fail"
    assert report["telemetry_gates"]["require_guest_prompt_sha256_match"] is True
    assert [run["guest_prompt_sha256_match"] for run in report["benchmarks"]] == [None, False]
    assert [finding["metric"] for finding in report["telemetry_findings"]] == [
        "guest_prompt_sha256_match",
        "guest_prompt_sha256_match",
    ]
    assert {finding["prompt"] for finding in report["telemetry_findings"]} == {"missing", "bad"}
    assert "guest_prompt_sha256_match" in markdown
    assert "guest_prompt_sha256_match" in csv_report
    assert "guest_prompt_sha256_match" in launch_csv
    assert junit_root.attrib["failures"] == "2"


def test_cli_guest_prompt_sha256_gate_passes_matching_hash(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"Exact prompt"}\n', encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
import hashlib
import os
prompt = os.environ["HOLYC_BENCH_PROMPT"]
print(f"tokens=12 elapsed_us=100000 prompt_sha256={hashlib.sha256(prompt.encode()).hexdigest()}")
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
            "--require-guest-prompt-sha256-match",
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["benchmarks"][0]["guest_prompt_sha256"] == report["benchmarks"][0]["prompt_sha256"]
    assert report["benchmarks"][0]["guest_prompt_sha256_match"] is True
    assert report["telemetry_findings"] == []


def test_cli_expected_tokens_gate_fails_mismatched_decode_length(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text(
        '{"prompt_id":"short","prompt":"Short decode","expected_tokens":12}\n'
        '{"prompt_id":"long","prompt":"Long decode","expected_tokens":24}\n',
        encoding="utf-8",
    )
    fake_qemu.write_text(
        """#!/usr/bin/env python3
import os
prompt_id = os.environ["HOLYC_BENCH_PROMPT_ID"]
tokens = 24 if prompt_id == "long" else 8
print(f"tokens={tokens} elapsed_us=100000")
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
            "--require-expected-tokens-match",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "qemu_prompt_bench_latest.md").read_text(encoding="utf-8")
    csv_report = (output_dir / "qemu_prompt_bench_latest.csv").read_text(encoding="utf-8")
    launch_csv = (output_dir / "qemu_prompt_bench_launches_latest.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()

    assert report["status"] == "fail"
    assert report["prompt_suite"]["expected_token_prompts"] == 2
    assert report["prompt_suite"]["expected_tokens_total"] == 36
    assert report["telemetry_gates"]["require_expected_tokens_match"] is True
    assert [run["expected_tokens_match"] for run in report["benchmarks"]] == [False, True]
    assert report["telemetry_findings"] == [
        {
            "scope": "measured_run",
            "launch_index": 1,
            "prompt": "short",
            "iteration": 1,
            "metric": "expected_tokens_match",
            "value": 8,
            "limit": 12,
        }
    ]
    assert "expected_tokens_match" in markdown
    assert "expected_tokens,expected_tokens_match" in csv_report
    assert "expected_tokens" in launch_csv
    assert junit_root.attrib["failures"] == "1"


def test_cli_expected_tokens_gate_ignores_undeclared_prompt_counts(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text(
        '{"prompt_id":"declared","prompt":"Declared decode","expected_tokens":7}\n'
        '{"prompt_id":"undeclared","prompt":"Free decode"}\n',
        encoding="utf-8",
    )
    fake_qemu.write_text(
        """#!/usr/bin/env python3
import os
prompt_id = os.environ["HOLYC_BENCH_PROMPT_ID"]
tokens = 123 if prompt_id == "undeclared" else 7
print(f"tokens={tokens} elapsed_us=100000")
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
            "--require-expected-tokens-match",
        ]
    )

    assert status == 0
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert [run["expected_tokens_match"] for run in report["benchmarks"]] == [True, None]
    assert report["telemetry_findings"] == []


def test_cli_junit_reports_failed_qemu_run(tmp_path: Path) -> None:
    fake_qemu = tmp_path / "fake-qemu.py"
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"Fail"}\n', encoding="utf-8")
    fake_qemu.write_text(
        """#!/usr/bin/env python3
import sys
print("tokens=4 elapsed_us=100000")
print("guest failure", file=sys.stderr)
raise SystemExit(7)
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

    assert status == 1
    report = json.loads((output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
    junit_root = ET.parse(output_dir / "qemu_prompt_bench_junit_latest.xml").getroot()
    failure = junit_root.find(".//failure")
    assert report["status"] == "fail"
    assert junit_root.attrib["tests"] == "1"
    assert junit_root.attrib["failures"] == "1"
    assert failure is not None
    assert failure.attrib["type"] == "qemu_prompt_failure"
    assert "returncode=7" in (failure.text or "")


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

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--dry-run",
            "--max-serial-output-bytes",
            "-1",
        ]
    )

    assert status == 2

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--dry-run",
            "--max-host-overhead-us",
            "-1",
        ]
    )

    assert status == 2

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--dry-run",
            "--max-host-overhead-pct",
            "-1",
        ]
    )

    assert status == 2

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--dry-run",
            "--max-launches",
            "-1",
        ]
    )

    assert status == 2


def test_cli_launch_budget_fails_before_qemu(tmp_path: Path, capsys) -> None:
    prompts = tmp_path / "prompts.jsonl"
    image = tmp_path / "temple.img"
    output_dir = tmp_path / "results"
    prompts.write_text(
        '{"prompt_id":"one","prompt":"A"}\n{"prompt_id":"two","prompt":"B"}\n',
        encoding="utf-8",
    )

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
            "--max-launches",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "planned QEMU launches (6) exceed --max-launches (5)" in captured.err
    assert not (output_dir / "qemu_prompt_bench_dry_run_latest.json").exists()


def test_cli_launch_budget_is_recorded_in_dry_run(tmp_path: Path, capsys) -> None:
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
            "--max-launches",
            "3",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    report = json.loads((output_dir / "qemu_prompt_bench_dry_run_latest.json").read_text(encoding="utf-8"))
    assert payload["max_launches"] == 3
    assert payload["planned_total_launches"] == 3
    assert report["max_launches"] == 3


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

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--dry-run",
            "--max-prompt-iqr-pct",
            "-1",
        ]
    )

    assert status == 2


def test_cli_rejects_negative_telemetry_gate(tmp_path: Path) -> None:
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
            "--min-tokens",
            "-1",
        ]
    )

    assert status == 2

    status = qemu_prompt_bench.main(
        [
            "--image",
            str(image),
            "--prompts",
            str(prompts),
            "--dry-run",
            "--max-ttft-us",
            "-1",
        ]
    )

    assert status == 2
