#!/usr/bin/env bash
# north-star-e2e.sh (holyc-inference) — RED until forward pass works.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Documents/local-codebases/holyc-inference}"
WEIGHTS="${WEIGHTS:-$REPO_DIR/models/gpt2-124m-q4_0.bin}"
REFERENCE="${REFERENCE:-$REPO_DIR/tests/reference_q4_gpt2.py}"
LOG="${LOG:-/tmp/holyc-forward.log}"
TIMEOUT="${TIMEOUT:-60}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "RED: weights missing at $WEIGHTS — earlier IQ items must produce these"
  exit 1
fi

if [[ ! -f "$REFERENCE" ]]; then
  echo "RED: reference script missing at $REFERENCE"
  exit 1
fi

# Run reference for ground truth
ref_token=$(python3 "$REFERENCE" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1 || echo "")
if [[ -z "$ref_token" ]]; then
  echo "RED: reference forward pass failed"
  exit 1
fi

# Run HolyC forward pass in QEMU (placeholder — IQ items fill this in)
HOLYC_RUNNER="$REPO_DIR/automation/run-holyc-forward.sh"
if [[ ! -x "$HOLYC_RUNNER" ]]; then
  echo "RED: HolyC forward-pass runner missing at $HOLYC_RUNNER — earlier IQ items must produce this"
  exit 1
fi

actual_token=$(timeout "$TIMEOUT" "$HOLYC_RUNNER" 2>"$LOG" | tail -1 | grep -oE '[0-9]+' | head -1 || echo "")
if [[ -z "$actual_token" ]]; then
  echo "RED: HolyC forward pass produced no token id (see $LOG)"
  exit 1
fi

if [[ "$actual_token" != "$ref_token" ]]; then
  echo "RED: HolyC token=$actual_token != reference=$ref_token"
  exit 1
fi

echo "GREEN: forward pass matches reference (token=$actual_token)"
exit 0
