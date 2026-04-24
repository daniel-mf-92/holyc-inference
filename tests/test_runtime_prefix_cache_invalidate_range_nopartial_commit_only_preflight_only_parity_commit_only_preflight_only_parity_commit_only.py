#!/usr/bin/env python3
"""Host-side harness for IQ-1353 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_runtime_prefix_cache_invalidate_range_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity import (
    Cache,
    Entry,
    PREFIX_CACHE_ERR_BAD_PARAM,
    PREFIX_CACHE_ERR_NULL_PTR,
    PREFIX_CACHE_FRESH_EMPTY,
    PREFIX_CACHE_FRESH_VALID,
    PREFIX_CACHE_OK,
    _clone_entries,
    invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only,
    invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity,
)


def invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
    cache: Cache | None,
    invalidate_start_token: int,
    invalidate_token_count: int,
):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token < 0 or invalidate_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    snapshot_capacity = cache.capacity
    snapshot_count = cache.count
    snapshot_start = invalidate_start_token
    snapshot_tokens = invalidate_token_count
    snapshot_entry_count = cache.count
    snapshot_entries = _clone_entries(cache.entries)

    status, parity_removed = invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    parity_remaining = cache.count
    if parity_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if parity_removed < 0 or parity_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if parity_removed > snapshot_entry_count or parity_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if parity_removed + parity_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    cache.entries = _clone_entries(snapshot_entries)
    cache.count = snapshot_count

    status, commit_removed = invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    commit_remaining = cache.count
    if commit_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed < 0 or commit_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed > snapshot_entry_count or commit_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed + commit_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    if parity_removed != commit_removed or parity_remaining != commit_remaining:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    return PREFIX_CACHE_OK, parity_removed


def test_iq1353_commit_only_hardening_matches_parity_tuple():
    entries = [Entry() for _ in range(6)]
    entries[0] = Entry(valid=1, prefix_hash=7, prefix_tokens=8, kv_start_token=0, kv_token_count=12)
    entries[1] = Entry(valid=1, prefix_hash=8, prefix_tokens=12, kv_start_token=12, kv_token_count=10)
    entries[2] = Entry(valid=1, prefix_hash=9, prefix_tokens=16, kv_start_token=28, kv_token_count=8)
    entries[4] = Entry(valid=1, prefix_hash=10, prefix_tokens=22, kv_start_token=60, kv_token_count=4)
    cache = Cache(entries=entries, capacity=6, count=4)

    status, removed = (
        invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(cache, 10, 30)
    )
    assert status == PREFIX_CACHE_OK
    assert removed == 3
    assert cache.count == 1
    assert entries[4].valid == PREFIX_CACHE_FRESH_VALID
    assert entries[0].valid == PREFIX_CACHE_FRESH_EMPTY
    assert entries[1].valid == PREFIX_CACHE_FRESH_EMPTY
    assert entries[2].valid == PREFIX_CACHE_FRESH_EMPTY


def test_iq1353_commit_only_hardening_zero_length_is_noop():
    entries = [Entry() for _ in range(3)]
    entries[0] = Entry(valid=1, prefix_hash=41, prefix_tokens=4, kv_start_token=3, kv_token_count=7)
    entries[1] = Entry(valid=1, prefix_hash=42, prefix_tokens=5, kv_start_token=20, kv_token_count=7)
    cache = Cache(entries=entries, capacity=3, count=2)

    status, removed = (
        invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(cache, 8, 0)
    )
    assert status == PREFIX_CACHE_OK
    assert removed == 0
    assert cache.count == 2
    assert entries[0].valid == PREFIX_CACHE_FRESH_VALID
    assert entries[1].valid == PREFIX_CACHE_FRESH_VALID


def test_iq1353_commit_only_hardening_error_keeps_state_unchanged():
    entries = [Entry() for _ in range(2)]
    entries[0] = Entry(valid=1, prefix_hash=51, prefix_tokens=2, kv_start_token=0, kv_token_count=5)
    entries[1] = Entry(valid=1, prefix_hash=52, prefix_tokens=3, kv_start_token=-4, kv_token_count=5)
    cache = Cache(entries=entries, capacity=2, count=2)
    before = _clone_entries(entries)

    status, removed = (
        invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(cache, 0, 10)
    )
    assert status == PREFIX_CACHE_ERR_BAD_PARAM
    assert removed is None
    assert cache.count == 2
    for idx, entry in enumerate(entries):
        assert vars(entry) == vars(before[idx])


def test_holyc_function_body_and_contract_markers_present():
    source = Path("src/runtime/prefix_cache.HC").read_text(encoding="utf-8")
    sig = (
        "I32 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(PrefixCache *cache,"
    )
    assert sig in source
    body = source.split(sig, 1)[1]
    assert (
        "status_parity = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(cache,"
        in body
    )
    assert "cache->entries[idx] = snapshot_entries[idx];" in body
    assert (
        "status_commit = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(cache,"
        in body
    )
    assert "if (parity_removed_count != commit_removed_count ||" in body
    assert "*out_removed_count = parity_removed_count;" in body


if __name__ == "__main__":
    test_iq1353_commit_only_hardening_matches_parity_tuple()
    test_iq1353_commit_only_hardening_zero_length_is_noop()
    test_iq1353_commit_only_hardening_error_keeps_state_unchanged()
    test_holyc_function_body_and_contract_markers_present()
    print("ok")
