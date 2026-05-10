#!/usr/bin/env python3
"""Smoke gate for qemu_input_provenance_audit.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qemu_input_provenance_audit
import qemu_prompt_bench


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        prompts = tmp / "prompts.jsonl"
        args_file = tmp / "qemu.args"
        image = tmp / "TempleOS.img"
        artifact = tmp / "qemu_prompt_bench_latest.json"
        output_dir = tmp / "out"

        write_jsonl(
            prompts,
            [
                {"id": "short", "prompt": "Add 2 and 2.", "expected_tokens": 4},
                {"id": "long", "prompt": "Name the primary color in a clear daytime sky.", "expected_tokens": 8},
            ],
        )
        args_file.write_text("-m 256M -display none\n", encoding="utf-8")
        image.write_bytes(b"synthetic image fixture")

        prompt_cases = qemu_prompt_bench.load_prompt_cases(prompts)
        report = {
            "status": "pass",
            "prompt_suite": qemu_prompt_bench.prompt_suite_metadata(prompts, prompt_cases),
            "image": qemu_prompt_bench.input_file_metadata(image, include_sha256=True),
            "qemu_args_files": [qemu_prompt_bench.input_file_metadata(args_file, include_sha256=True)],
            "benchmarks": [],
        }
        artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        status = qemu_input_provenance_audit.main(
            [
                str(artifact),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_input_provenance_audit_smoke",
                "--require-live-inputs",
                "--require-file-sha256",
                "--require-image-metadata",
            ]
        )
        if status != 0:
            return status
        payload = json.loads((output_dir / "qemu_input_provenance_audit_smoke.json").read_text(encoding="utf-8"))
        if payload["status"] != "pass" or payload["summary"]["prompt_live_checked"] != 1:
            return 1

        broken = dict(report)
        broken["prompt_suite"] = dict(report["prompt_suite"], suite_sha256="0" * 64)
        broken["qemu_args_files"] = [dict(report["qemu_args_files"][0], sha256="1" * 64)]
        artifact.write_text(json.dumps(broken, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        status = qemu_input_provenance_audit.main(
            [
                str(artifact),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_input_provenance_audit_smoke_fail",
                "--require-live-inputs",
                "--require-file-sha256",
            ]
        )
        if status == 0:
            return 1
        fail_payload = json.loads((output_dir / "qemu_input_provenance_audit_smoke_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_payload["findings"]}
        return 0 if {"prompt_suite_drift", "sha256_drift"} <= kinds else 1


if __name__ == "__main__":
    raise SystemExit(main())
