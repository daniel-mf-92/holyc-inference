#!/usr/bin/env python3
"""Tests for host-side benchmark artifact air-gap auditing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import airgap_audit


def test_audit_checks_warmup_commands(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "warmups": [
                    {
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "user",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ]
                    }
                ],
                "benchmarks": [
                    {
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    commands_checked, findings = airgap_audit.audit([report])

    assert commands_checked == 2
    assert len(findings) == 2
    assert any("missing explicit `-nic none`" in finding.reason for finding in findings)
    assert any("non-air-gapped" in finding.reason for finding in findings)


def test_audit_checks_bench_matrix_cell_commands(tmp_path: Path) -> None:
    report = tmp_path / "bench_matrix_latest.json"
    report.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "profile": "unsafe",
                        "command": [
                            "qemu-system-x86_64",
                            "-netdev",
                            "user,id=n0",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    },
                    {
                        "profile": "safe",
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    commands_checked, findings = airgap_audit.audit([report])

    assert commands_checked == 2
    assert len(findings) == 2
    assert any("missing explicit `-nic none`" in finding.reason for finding in findings)
    assert any("network backend" in finding.reason for finding in findings)
