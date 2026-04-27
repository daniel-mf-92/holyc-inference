#!/usr/bin/env python3
"""Synthetic QEMU-compatible benchmark fixture for host-side smoke runs.

This is not an emulator. It accepts the same stdin/env prompt contract used by
qemu_prompt_bench.py and emits deterministic BENCH_RESULT telemetry so benchmark
report generation can be refreshed without booting a guest.
"""

from __future__ import annotations

import json
import os
import sys


TOKEN_COUNTS = {
    "smoke-short": 32,
    "smoke-code": 48,
}


def main() -> int:
    prompt_id = os.environ.get("HOLYC_BENCH_PROMPT_ID", "unknown")
    prompt = sys.stdin.read()
    tokens = TOKEN_COUNTS.get(prompt_id, max(1, len(prompt.split())))
    elapsed_us = tokens * 6250
    print("BENCH_RESULT: " + json.dumps({"tokens": tokens, "elapsed_us": elapsed_us}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
