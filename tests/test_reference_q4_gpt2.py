#!/usr/bin/env python3
"""Host-side checks for tests/reference_q4_gpt2.py."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tests" / "reference_q4_gpt2.py"


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def test_update_with_manual_token_and_emit_json() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "ref.json"

        write = run_cmd(
            [
                "python3",
                str(SCRIPT),
                "--fixture",
                str(fixture),
                "--update-fixture",
                "--set-token",
                "12345",
                "--seed",
                "7",
            ]
        )
        assert write.returncode == 0, write.stderr

        read = run_cmd(["python3", str(SCRIPT), "--fixture", str(fixture), "--emit-json"])
        assert read.returncode == 0, read.stderr
        payload = json.loads(read.stdout)
        assert payload["next_token_id"] == 12345
        assert payload["seed"] == 7


def test_update_with_capture_cmd() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "ref.json"

        write = run_cmd(
            [
                "python3",
                str(SCRIPT),
                "--fixture",
                str(fixture),
                "--update-fixture",
                "--capture-cmd",
                "printf 'next_token_id=777\\n'",
                "--capture-token-pattern",
                r"next_token_id=(\d+)",
            ]
        )
        assert write.returncode == 0, write.stderr

        data = json.loads(fixture.read_text(encoding="utf-8"))
        assert data["next_token_id"] == 777
        assert data["source"] == "capture-cmd"


def test_missing_fixture_is_error() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "missing.json"
        run = run_cmd(["python3", str(SCRIPT), "--fixture", str(fixture)])
        assert run.returncode != 0
        assert "fixture not found" in run.stderr
