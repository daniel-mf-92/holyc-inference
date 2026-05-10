#!/usr/bin/env python3
"""Synthetic QEMU-compatible benchmark fixture for host-side smoke runs.

This is not an emulator. It accepts the same stdin/env prompt contract used by
qemu_prompt_bench.py and emits deterministic BENCH_RESULT telemetry so benchmark
report generation can be refreshed without booting a guest.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time


TOKEN_COUNTS = {
    "smoke-short": 32,
    "smoke-code": 48,
}


def main() -> int:
    prompt_id = os.environ.get("HOLYC_BENCH_PROMPT_ID", "unknown")
    prompt = sys.stdin.read()
    tokens = TOKEN_COUNTS.get(prompt_id, max(1, len(prompt.split())))
    elapsed_us = tokens * 6250
    ttft_us = 10_000 + tokens * 50
    memory_bytes = 64 * 1024 * 1024 + tokens * 2048
    time.sleep(elapsed_us / 1_000_000.0)
    print(
        "BENCH_RESULT: "
        + json.dumps(
            {
                "tokens": tokens,
                "elapsed_us": elapsed_us,
                "time_to_first_token_us": ttft_us,
                "memory_bytes": memory_bytes,
                "prompt_bytes": len(prompt.encode("utf-8")),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
