#!/usr/bin/env python3
"""Host-side parity harness for PrefixCacheExportAuditRowsChecked (IQ-1290)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_FULL = 3

PREFIX_CACHE_FRESH_EMPTY = 0
PREFIX_CACHE_FRESH_VALID = 1

PREFIX_CACHE_HASH_FNV_PRIME = 1099511628211
PREFIX_CACHE_AUDIT_TUPLE_WIDTH = 7
PREFIX_CACHE_AUDIT_DOMAIN_TAG = 0x5052464155444954


@dataclass
class PrefixCacheEntry:
    valid: int = PREFIX_CACHE_FRESH_EMPTY
    prefix_hash: int = 0
    prefix_tokens: int = 0
    kv_start_token: int = 0
    kv_token_count: int = 0
    last_used_tick: int = 0


@dataclass
class PrefixCache:
    entries: list[PrefixCacheEntry]
    capacity: int
    count: int


def _u64(value: int) -> int:
    return value & 0xFFFFFFFFFFFFFFFF


def _digest(entry_index: int, entry: PrefixCacheEntry) -> int:
    digest = _u64(PREFIX_CACHE_AUDIT_DOMAIN_TAG)
    digest = _u64((digest ^ _u64(entry_index)) * PREFIX_CACHE_HASH_FNV_PRIME)
    digest = _u64((digest ^ _u64(entry.prefix_hash)) * PREFIX_CACHE_HASH_FNV_PRIME)
    digest = _u64((digest ^ _u64(entry.prefix_tokens)) * PREFIX_CACHE_HASH_FNV_PRIME)
    digest = _u64((digest ^ _u64(entry.kv_start_token)) * PREFIX_CACHE_HASH_FNV_PRIME)
    digest = _u64((digest ^ _u64(entry.kv_token_count)) * PREFIX_CACHE_HASH_FNV_PRIME)
    digest = _u64((digest ^ _u64(entry.last_used_tick)) * PREFIX_CACHE_HASH_FNV_PRIME)
    return digest & 0x7FFFFFFFFFFFFFFF


def export_audit_rows_checked(
    cache: PrefixCache,
    row_buffer: list[int] | None,
    row_capacity: int,
) -> tuple[int, int]:
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, 0
    if cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, 0
    if row_capacity < 0 or cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, 0
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, 0

    valid_rows = 0
    for entry in cache.entries:
        if entry.valid == PREFIX_CACHE_FRESH_VALID:
            if min(
                entry.prefix_hash,
                entry.prefix_tokens,
                entry.kv_start_token,
                entry.kv_token_count,
                entry.last_used_tick,
            ) < 0:
                return PREFIX_CACHE_ERR_BAD_PARAM, 0
            if entry.kv_start_token + entry.kv_token_count < entry.kv_start_token:
                return PREFIX_CACHE_ERR_BAD_PARAM, 0
            valid_rows += 1

    if valid_rows != cache.count:
        return PREFIX_CACHE_ERR_BAD_PARAM, 0
    if valid_rows > row_capacity:
        return PREFIX_CACHE_ERR_FULL, 0
    if valid_rows > 0 and row_buffer is None:
        return PREFIX_CACHE_ERR_NULL_PTR, 0

    rows_written = 0
    for idx, entry in enumerate(cache.entries):
        if entry.valid != PREFIX_CACHE_FRESH_VALID:
            continue
        base = rows_written * PREFIX_CACHE_AUDIT_TUPLE_WIDTH
        row_buffer[base + 0] = idx
        row_buffer[base + 1] = entry.prefix_hash
        row_buffer[base + 2] = entry.prefix_tokens
        row_buffer[base + 3] = entry.kv_start_token
        row_buffer[base + 4] = entry.kv_token_count
        row_buffer[base + 5] = entry.last_used_tick
        row_buffer[base + 6] = _digest(idx, entry)
        rows_written += 1

    return PREFIX_CACHE_OK, rows_written


def test_export_rows_fixed_width_with_digest():
    cache = PrefixCache(
        entries=[
            PrefixCacheEntry(1, 101, 16, 64, 16, 12),
            PrefixCacheEntry(0, 0, 0, 0, 0, 0),
            PrefixCacheEntry(1, 202, 32, 80, 32, 20),
        ],
        capacity=3,
        count=2,
    )
    rows = [0] * (2 * PREFIX_CACHE_AUDIT_TUPLE_WIDTH)

    status, wrote = export_audit_rows_checked(cache, rows, row_capacity=2)
    assert status == PREFIX_CACHE_OK
    assert wrote == 2

    row0 = rows[0:7]
    row1 = rows[7:14]
    assert row0 == [0, 101, 16, 64, 16, 12, _digest(0, cache.entries[0])]
    assert row1 == [2, 202, 32, 80, 32, 20, _digest(2, cache.entries[2])]


def test_export_preflight_capacity_guard_no_partial_write():
    cache = PrefixCache(
        entries=[
            PrefixCacheEntry(1, 11, 8, 40, 8, 2),
            PrefixCacheEntry(1, 22, 12, 52, 12, 3),
        ],
        capacity=2,
        count=2,
    )
    rows = [-1] * PREFIX_CACHE_AUDIT_TUPLE_WIDTH

    status, wrote = export_audit_rows_checked(cache, rows, row_capacity=1)
    assert status == PREFIX_CACHE_ERR_FULL
    assert wrote == 0
    assert rows == [-1] * PREFIX_CACHE_AUDIT_TUPLE_WIDTH


def test_export_rejects_count_drift_and_negative_fields():
    drift = PrefixCache(
        entries=[PrefixCacheEntry(1, 7, 4, 20, 4, 9)],
        capacity=1,
        count=0,
    )
    rows = [0] * PREFIX_CACHE_AUDIT_TUPLE_WIDTH
    assert export_audit_rows_checked(drift, rows, row_capacity=1)[0] == PREFIX_CACHE_ERR_BAD_PARAM

    bad = PrefixCache(
        entries=[PrefixCacheEntry(1, 7, 4, 20, 4, -1)],
        capacity=1,
        count=1,
    )
    assert export_audit_rows_checked(bad, rows, row_capacity=1)[0] == PREFIX_CACHE_ERR_BAD_PARAM


if __name__ == "__main__":
    src = Path("src/runtime/prefix_cache.HC").read_text(encoding="utf-8")
    assert "PrefixCacheExportAuditRowsChecked(" in src
    assert "PREFIX_CACHE_AUDIT_TUPLE_WIDTH" in src
    test_export_rows_fixed_width_with_digest()
    test_export_preflight_capacity_guard_no_partial_write()
    test_export_rejects_count_drift_and_negative_fields()
    print("ok")
