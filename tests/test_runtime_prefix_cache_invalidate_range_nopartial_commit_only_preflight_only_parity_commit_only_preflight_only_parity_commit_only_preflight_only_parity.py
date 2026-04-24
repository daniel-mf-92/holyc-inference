#!/usr/bin/env python3
"""Host-side harness for IQ-1356 strict parity diagnostics gate."""

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
)
from test_runtime_prefix_cache_invalidate_range_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only import (
    invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only,
)
from test_runtime_prefix_cache_invalidate_range_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only import (
    invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only,
)


def invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
    cache: Cache | None,
    invalidate_start_token: int,
    invalidate_token_count: int,
    out_removed: list[int] | None,
):
    if cache is None or cache.entries is None or out_removed is None:
        return PREFIX_CACHE_ERR_NULL_PTR
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if invalidate_start_token < 0 or invalidate_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM

    snapshot_capacity = cache.capacity
    snapshot_count = cache.count
    snapshot_start = invalidate_start_token
    snapshot_tokens = invalidate_token_count
    snapshot_entry_count = cache.count
    snapshot_out = out_removed[0]
    snapshot_entries = _clone_entries(cache.entries)

    status = (
        invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
            cache,
            invalidate_start_token,
            invalidate_token_count,
            out_removed,
        )
    )
    if status != PREFIX_CACHE_OK:
        return status

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if out_removed[0] != snapshot_out:
        return PREFIX_CACHE_ERR_BAD_PARAM

    parity_remaining = cache.count
    parity_removed = snapshot_entry_count - parity_remaining
    if parity_removed < 0 or parity_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if parity_removed > snapshot_entry_count or parity_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if parity_removed + parity_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM

    cache.entries = _clone_entries(snapshot_entries)
    cache.count = snapshot_count

    status, commit_removed = (
        invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
            cache,
            invalidate_start_token,
            invalidate_token_count,
        )
    )
    if status != PREFIX_CACHE_OK:
        return status

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if out_removed[0] != snapshot_out:
        return PREFIX_CACHE_ERR_BAD_PARAM

    commit_remaining = cache.count
    if commit_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if commit_removed < 0 or commit_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if commit_removed > snapshot_entry_count or commit_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if commit_removed + commit_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM

    if parity_removed != commit_removed or parity_remaining != commit_remaining:
        return PREFIX_CACHE_ERR_BAD_PARAM

    out_removed[0] = parity_removed
    return PREFIX_CACHE_OK


def test_iq1356_strict_parity_gate_publishes_only_after_match():
    entries = [Entry() for _ in range(6)]
    entries[0] = Entry(valid=1, prefix_hash=7, prefix_tokens=8, kv_start_token=0, kv_token_count=12)
    entries[1] = Entry(valid=1, prefix_hash=8, prefix_tokens=12, kv_start_token=12, kv_token_count=10)
    entries[2] = Entry(valid=1, prefix_hash=9, prefix_tokens=16, kv_start_token=28, kv_token_count=8)
    entries[4] = Entry(valid=1, prefix_hash=10, prefix_tokens=22, kv_start_token=60, kv_token_count=4)
    cache = Cache(entries=entries, capacity=6, count=4)
    out_removed = [4444]

    status = (
        invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            cache,
            10,
            30,
            out_removed,
        )
    )
    assert status == PREFIX_CACHE_OK
    assert out_removed[0] == 3
    assert cache.count == 1
    assert entries[4].valid == PREFIX_CACHE_FRESH_VALID
    assert entries[0].valid == PREFIX_CACHE_FRESH_EMPTY
    assert entries[1].valid == PREFIX_CACHE_FRESH_EMPTY
    assert entries[2].valid == PREFIX_CACHE_FRESH_EMPTY


def test_iq1356_strict_parity_gate_error_keeps_state_and_output():
    entries = [Entry() for _ in range(2)]
    entries[0] = Entry(valid=1, prefix_hash=51, prefix_tokens=2, kv_start_token=0, kv_token_count=5)
    entries[1] = Entry(valid=1, prefix_hash=52, prefix_tokens=3, kv_start_token=-4, kv_token_count=5)
    cache = Cache(entries=entries, capacity=2, count=2)
    out_removed = [111]
    before = _clone_entries(entries)

    status = (
        invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            cache,
            0,
            10,
            out_removed,
        )
    )
    assert status == PREFIX_CACHE_ERR_BAD_PARAM
    assert out_removed[0] == 111
    assert cache.count == 2
    for idx, entry in enumerate(entries):
        assert vars(entry) == vars(before[idx])


def test_holyc_function_body_and_contract_markers_present():
    source = Path("src/runtime/prefix_cache.HC").read_text(encoding="utf-8")
    sig = (
        "I32 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(PrefixCache *cache,"
    )
    assert sig in source
    body = source.split(sig, 1)[1]
    assert (
        "status_parity = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(cache,"
        in body
    )
    assert (
        "status_commit = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(cache,"
        in body
    )
    assert "*out_removed_count != snapshot_out_removed_value" in body
    assert "parity_removed_count = snapshot_entry_count - parity_remaining_count;" in body
    assert "*out_removed_count = parity_removed_count;" in body


if __name__ == "__main__":
    test_iq1356_strict_parity_gate_publishes_only_after_match()
    test_iq1356_strict_parity_gate_error_keeps_state_and_output()
    test_holyc_function_body_and_contract_markers_present()
    print("ok")
