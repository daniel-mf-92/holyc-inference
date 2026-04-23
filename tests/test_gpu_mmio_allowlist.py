#!/usr/bin/env python3
"""Parity harness for GPU BAR/MMIO allowlist enforcement helper (IQ-1260)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

GPU_MMIO_ALLOW_OK = 0
GPU_MMIO_ALLOW_ERR_NULL_PTR = 1
GPU_MMIO_ALLOW_ERR_BAD_PARAM = 2
GPU_MMIO_ALLOW_ERR_BAD_TABLE = 3
GPU_MMIO_ALLOW_ERR_OVERLAP = 4

GPU_MMIO_ALLOW_REASON_ALLOW = 0
GPU_MMIO_ALLOW_REASON_DENY_DEFAULT = 1
GPU_MMIO_ALLOW_REASON_BAD_BAR = 2
GPU_MMIO_ALLOW_REASON_BAD_OFFSET = 3
GPU_MMIO_ALLOW_REASON_BAD_WIDTH = 4
GPU_MMIO_ALLOW_REASON_BAD_TABLE = 5
GPU_MMIO_ALLOW_REASON_TABLE_OVERLAP = 6

GPU_MMIO_WIDTH_MASK_1B = 1
GPU_MMIO_WIDTH_MASK_2B = 2
GPU_MMIO_WIDTH_MASK_4B = 4
GPU_MMIO_WIDTH_MASK_8B = 8


@dataclass(frozen=True)
class Entry:
    bar_index: int
    reg_start: int
    reg_end: int
    width_mask: int


def _width_bytes_is_valid(width_bytes: int) -> bool:
    return width_bytes in (1, 2, 4, 8)


def _width_mask_from_bytes(width_bytes: int) -> int:
    return {
        1: GPU_MMIO_WIDTH_MASK_1B,
        2: GPU_MMIO_WIDTH_MASK_2B,
        4: GPU_MMIO_WIDTH_MASK_4B,
        8: GPU_MMIO_WIDTH_MASK_8B,
    }.get(width_bytes, 0)


def _width_mask_is_valid(mask: int) -> bool:
    known = (
        GPU_MMIO_WIDTH_MASK_1B
        | GPU_MMIO_WIDTH_MASK_2B
        | GPU_MMIO_WIDTH_MASK_4B
        | GPU_MMIO_WIDTH_MASK_8B
    )
    return mask > 0 and (mask & ~known) == 0


def _entry_overlap(lhs: Entry, rhs: Entry) -> bool:
    if lhs.bar_index != rhs.bar_index:
        return False
    return lhs.reg_start <= rhs.reg_end and rhs.reg_start <= lhs.reg_end


def _validate_table(table: list[Entry]) -> int:
    for entry in table:
        if entry.bar_index < 0:
            return GPU_MMIO_ALLOW_ERR_BAD_TABLE
        if entry.reg_start < 0 or entry.reg_end < 0 or entry.reg_end < entry.reg_start:
            return GPU_MMIO_ALLOW_ERR_BAD_TABLE
        if not _width_mask_is_valid(entry.width_mask):
            return GPU_MMIO_ALLOW_ERR_BAD_TABLE

    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            if _entry_overlap(table[i], table[j]):
                return GPU_MMIO_ALLOW_ERR_OVERLAP

    return GPU_MMIO_ALLOW_OK


def check_write(
    table: list[Entry],
    bar_index: int,
    reg_offset: int,
    width_bytes: int,
) -> tuple[int, int, int, int]:
    allow = 0
    reason = GPU_MMIO_ALLOW_REASON_DENY_DEFAULT
    match_idx = -1

    if bar_index < 0:
        return GPU_MMIO_ALLOW_ERR_BAD_PARAM, allow, GPU_MMIO_ALLOW_REASON_BAD_BAR, match_idx
    if reg_offset < 0:
        return GPU_MMIO_ALLOW_ERR_BAD_PARAM, allow, GPU_MMIO_ALLOW_REASON_BAD_OFFSET, match_idx
    if not _width_bytes_is_valid(width_bytes):
        return GPU_MMIO_ALLOW_ERR_BAD_PARAM, allow, GPU_MMIO_ALLOW_REASON_BAD_WIDTH, match_idx

    status = _validate_table(table)
    if status == GPU_MMIO_ALLOW_ERR_OVERLAP:
        return status, allow, GPU_MMIO_ALLOW_REASON_TABLE_OVERLAP, match_idx
    if status != GPU_MMIO_ALLOW_OK:
        return status, allow, GPU_MMIO_ALLOW_REASON_BAD_TABLE, match_idx

    width_mask = _width_mask_from_bytes(width_bytes)
    bar_seen = False

    for idx, entry in enumerate(table):
        if entry.bar_index != bar_index:
            continue

        bar_seen = True

        if reg_offset < entry.reg_start or reg_offset > entry.reg_end:
            continue

        if (entry.width_mask & width_mask) == 0:
            return GPU_MMIO_ALLOW_OK, 0, GPU_MMIO_ALLOW_REASON_BAD_WIDTH, -1

        return GPU_MMIO_ALLOW_OK, 1, GPU_MMIO_ALLOW_REASON_ALLOW, idx

    if not bar_seen:
        reason = GPU_MMIO_ALLOW_REASON_BAD_BAR

    return GPU_MMIO_ALLOW_OK, 0, reason, -1


def _sample_allowlist() -> list[Entry]:
    return [
        Entry(bar_index=0, reg_start=0x0000, reg_end=0x00FF, width_mask=GPU_MMIO_WIDTH_MASK_4B),
        Entry(
            bar_index=2,
            reg_start=0x1000,
            reg_end=0x10FF,
            width_mask=GPU_MMIO_WIDTH_MASK_4B | GPU_MMIO_WIDTH_MASK_8B,
        ),
    ]


def _bench_ns_per_call(table: list[Entry], iters: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iters):
        check_write(table, 2, 0x1020, 4)
    elapsed = time.perf_counter_ns() - start
    return elapsed / iters


def test_source_contains_iq1260_symbols() -> None:
    src = Path("src/gpu/mmio_allowlist.HC").read_text(encoding="utf-8")

    assert "class GPUMMIOAllowEntry" in src
    assert "I32 GPUMMIOAllowlistValidateTableChecked(" in src
    assert "I32 GPUMMIOAllowlistCheckWriteChecked(" in src
    assert "GPU_MMIO_ALLOW_REASON_DENY_DEFAULT" in src
    assert "GPU_MMIO_ALLOW_REASON_TABLE_OVERLAP" in src


def test_allow_known_bar_range_and_width() -> None:
    status, allow, reason, match = check_write(_sample_allowlist(), 2, 0x1004, 8)

    assert status == GPU_MMIO_ALLOW_OK
    assert allow == 1
    assert reason == GPU_MMIO_ALLOW_REASON_ALLOW
    assert match == 1


def test_deny_unknown_bar_and_out_of_range_offset() -> None:
    status, allow, reason, match = check_write(_sample_allowlist(), 7, 0x1004, 4)
    assert status == GPU_MMIO_ALLOW_OK
    assert allow == 0
    assert reason == GPU_MMIO_ALLOW_REASON_BAD_BAR
    assert match == -1

    status, allow, reason, match = check_write(_sample_allowlist(), 2, 0x2000, 4)
    assert status == GPU_MMIO_ALLOW_OK
    assert allow == 0
    assert reason == GPU_MMIO_ALLOW_REASON_DENY_DEFAULT
    assert match == -1


def test_deny_width_mismatch_even_when_register_is_covered() -> None:
    status, allow, reason, match = check_write(_sample_allowlist(), 0, 0x0050, 2)

    assert status == GPU_MMIO_ALLOW_OK
    assert allow == 0
    assert reason == GPU_MMIO_ALLOW_REASON_BAD_WIDTH
    assert match == -1


def test_reject_bad_table_and_overlap() -> None:
    bad_table = [Entry(bar_index=0, reg_start=0x80, reg_end=0x70, width_mask=GPU_MMIO_WIDTH_MASK_4B)]
    status, allow, reason, _ = check_write(bad_table, 0, 0x70, 4)
    assert status == GPU_MMIO_ALLOW_ERR_BAD_TABLE
    assert allow == 0
    assert reason == GPU_MMIO_ALLOW_REASON_BAD_TABLE

    overlap = [
        Entry(bar_index=2, reg_start=0x1000, reg_end=0x10FF, width_mask=GPU_MMIO_WIDTH_MASK_4B),
        Entry(bar_index=2, reg_start=0x10F0, reg_end=0x11FF, width_mask=GPU_MMIO_WIDTH_MASK_4B),
    ]
    status, allow, reason, _ = check_write(overlap, 2, 0x10F5, 4)
    assert status == GPU_MMIO_ALLOW_ERR_OVERLAP
    assert allow == 0
    assert reason == GPU_MMIO_ALLOW_REASON_TABLE_OVERLAP


def test_hardening_secure_on_overhead_budget_plan() -> None:
    # Perf-overhead measurement plan for secure-local mode:
    # Compare allowlist check cost against empty-table deny path and keep
    # overhead bounded for policy-gated dispatch checks.
    # This is a deterministic host-side guardrail, not a runtime fast-path bypass.
    secured_table = _sample_allowlist()
    empty_table: list[Entry] = []

    iters = 200_000
    ns_secure = _bench_ns_per_call(secured_table, iters)
    ns_empty = _bench_ns_per_call(empty_table, iters)

    assert ns_secure > 0
    assert ns_empty > 0
    # Keep secure policy overhead bounded to a practical envelope for per-dispatch checks.
    assert ns_secure <= ns_empty * 12.0


if __name__ == "__main__":
    test_source_contains_iq1260_symbols()
    test_allow_known_bar_range_and_width()
    test_deny_unknown_bar_and_out_of_range_offset()
    test_deny_width_mismatch_even_when_register_is_covered()
    test_reject_bad_table_and_overlap()
    test_hardening_secure_on_overhead_budget_plan()
    print("ok")
