#!/usr/bin/env python3
"""Reference next-token generator for North Star GPT-2 Q4_0 checks.

Default mode reads a local fixture and prints the reference token id.
Update mode can write that fixture either from a provided token or by
capturing command output (for example, a local llama.cpp invocation).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_IDS = [15496, 11, 995]
DEFAULT_MODEL_ID = "gpt2-124m-q4_0"
DEFAULT_SEED = 1234
DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "reference_q4_gpt2.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("fixture must be a JSON object")
    return data


def _extract_token_from_text(text: str, pattern: str) -> int:
    regex = re.compile(pattern)
    matches = list(regex.finditer(text))
    if not matches:
        raise ValueError("no token id match found in captured output")
    matched = matches[-1].group(1) if matches[-1].groups() else matches[-1].group(0)
    return int(matched)


def _capture_token(capture_cmd: str, token_pattern: str) -> tuple[int, str]:
    proc = subprocess.run(
        capture_cmd,
        shell=True,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    merged = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    if proc.returncode != 0:
        raise RuntimeError(f"capture command failed ({proc.returncode})")
    token = _extract_token_from_text(merged, token_pattern)
    return token, merged


def _write_fixture(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _validate_prompt_ids(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("prompt token id list is empty")
    return [int(v) for v in values]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Emit or update GPT-2 Q4_0 reference token fixture")
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--prompt-token-ids", default=",".join(str(x) for x in DEFAULT_PROMPT_IDS))
    ap.add_argument("--emit-json", action="store_true")

    ap.add_argument("--update-fixture", action="store_true")
    ap.add_argument("--set-token", type=int)
    ap.add_argument("--capture-cmd")
    ap.add_argument("--capture-token-pattern", default=r"(-?\\d+)")

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    try:
        prompt_ids = _validate_prompt_ids(args.prompt_token_ids)
    except Exception as exc:
        print(f"ERROR: invalid --prompt-token-ids ({exc})", file=sys.stderr)
        return 2

    if args.update_fixture:
        if args.set_token is None and not args.capture_cmd:
            print("ERROR: --update-fixture requires --set-token or --capture-cmd", file=sys.stderr)
            return 2
        if args.set_token is not None and args.capture_cmd:
            print("ERROR: use only one of --set-token or --capture-cmd", file=sys.stderr)
            return 2

        token = args.set_token
        source = "manual"
        capture_excerpt = None

        if args.capture_cmd:
            try:
                token, captured = _capture_token(args.capture_cmd, args.capture_token_pattern)
                source = "capture-cmd"
                capture_excerpt = captured[-1000:]
            except Exception as exc:
                print(f"ERROR: failed to capture token ({exc})", file=sys.stderr)
                return 1

        payload: dict[str, Any] = {
            "updated_at_utc": _utc_now(),
            "source": source,
            "model_id": args.model_id,
            "seed": args.seed,
            "prompt_token_ids": prompt_ids,
            "next_token_id": int(token),
        }

        if args.capture_cmd:
            payload["capture_cmd"] = args.capture_cmd
            if capture_excerpt is not None:
                payload["capture_excerpt"] = capture_excerpt

        _write_fixture(args.fixture, payload)

    if not args.fixture.exists():
        print(f"ERROR: fixture not found at {args.fixture}", file=sys.stderr)
        return 1

    try:
        data = _load_fixture(args.fixture)
        token = int(data["next_token_id"])
    except Exception as exc:
        print(f"ERROR: invalid fixture at {args.fixture} ({exc})", file=sys.stderr)
        return 1

    if args.emit_json:
        out = {
            "model_id": data.get("model_id", args.model_id),
            "seed": int(data.get("seed", args.seed)),
            "prompt_token_ids": data.get("prompt_token_ids", prompt_ids),
            "next_token_id": token,
            "fixture": str(args.fixture),
        }
        print(json.dumps(out, sort_keys=True))
    else:
        print(token)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
