#!/usr/bin/env python3
import csv
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "automation" / "perf-matrix.sh"
LOG_DIR = ROOT / "automation" / "logs"


def _find_newest(pattern: str) -> Path:
    matches = sorted(LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    assert matches, f"no files for pattern {pattern}"
    return matches[0]


def test_secure_vs_dev_perf_matrix_hardening_and_summary():
    env = os.environ.copy()
    env["PERF_REPEATS"] = "2"
    env["PERF_TOKENS"] = "128"
    env["PERF_PROMPT_TOKENS"] = "64"

    run = subprocess.run([str(SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr + "\n" + run.stdout

    csv_match = re.search(r"perf_matrix_csv=(.+)", run.stdout)
    summary_match = re.search(r"perf_matrix_summary=(.+)", run.stdout)
    csv_path = Path(csv_match.group(1)) if csv_match else _find_newest("perf-matrix-*.csv")
    summary_path = Path(summary_match.group(1)) if summary_match else _find_newest("perf-matrix-*.summary.txt")

    assert csv_path.exists(), f"missing csv {csv_path}"
    assert summary_path.exists(), f"missing summary {summary_path}"

    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert len(rows) >= 4

    secure = [r for r in rows if r["profile"] == "secure-local"]
    dev = [r for r in rows if r["profile"] == "dev-local"]
    assert len(secure) == 2
    assert len(dev) == 2

    for row in rows:
        hardening = row["hardening"]
        assert "attestation=on" in hardening
        assert "policy_digest=on" in hardening
        assert "audit_hooks=on" in hardening
        assert int(row["elapsed_us"]) > 0
        assert int(row["tok_per_s_milli"]) > 0

    summary = summary_path.read_text(encoding="utf-8")
    assert "secure_avg_tok_per_s=" in summary
    assert "dev_avg_tok_per_s=" in summary
    assert "secure_overhead_pct=" in summary
