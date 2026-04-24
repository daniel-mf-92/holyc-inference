#!/usr/bin/env python3
"""Host-side harness for IQ-1293 PrefixCacheLookupBestPrefixCheckedNoPartial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_NOT_FOUND = 4

PREFIX_CACHE_FRESH_EMPTY = 0
PREFIX_CACHE_FRESH_VALID = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "src" / "runtime" / "prefix_cache.HC"


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
    entries: list[PrefixCacheEntry] | None
    capacity: int
    count: int


def prefix_cache_lookup_best_prefix_checked_nopartial(
    cache: PrefixCache | None,
    query_hash: int,
    max_prompt_tokens: int,
):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash < 0 or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    snapshot_count = cache.count
    snapshot_capacity = cache.capacity
    snapshot_query_hash = query_hash
    snapshot_max_prompt_tokens = max_prompt_tokens

    best_index = -1
    best_tokens = 0

    for idx, entry in enumerate(cache.entries[:snapshot_capacity]):
        if entry.valid == PREFIX_CACHE_FRESH_VALID:
            if (
                entry.prefix_hash < 0
                or entry.prefix_tokens < 0
                or entry.kv_start_token < 0
                or entry.kv_token_count < 0
                or entry.last_used_tick < 0
            ):
                return PREFIX_CACHE_ERR_BAD_PARAM, None, None

            if entry.prefix_hash == snapshot_query_hash and entry.prefix_tokens <= snapshot_max_prompt_tokens:
                if (
                    best_index < 0
                    or entry.prefix_tokens > best_tokens
                    or (entry.prefix_tokens == best_tokens and idx < best_index)
                ):
                    best_index = idx
                    best_tokens = entry.prefix_tokens

    if best_index < 0:
        return PREFIX_CACHE_ERR_NOT_FOUND, None, None

    if cache.capacity != snapshot_capacity or cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if cache.count != snapshot_count or cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash != snapshot_query_hash or query_hash < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if max_prompt_tokens != snapshot_max_prompt_tokens or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    return PREFIX_CACHE_OK, best_index, best_tokens


def _cache(entries: list[PrefixCacheEntry]) -> PrefixCache:
    valid_count = sum(1 for entry in entries if entry.valid == PREFIX_CACHE_FRESH_VALID)
    return PrefixCache(entries=entries, capacity=len(entries), count=valid_count)


def test_iq1293_longest_prefix_with_stable_tiebreak():
    entries = [PrefixCacheEntry() for _ in range(6)]
    entries[4] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=77, prefix_tokens=32)
    entries[2] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=77, prefix_tokens=32)
    entries[1] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=77, prefix_tokens=24)
    entries[5] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=12, prefix_tokens=64)

    status, best_index, best_tokens = prefix_cache_lookup_best_prefix_checked_nopartial(
        _cache(entries), 77, 32
    )
    assert status == PREFIX_CACHE_OK
    assert (best_index, best_tokens) == (2, 32)


def test_iq1293_respects_max_prompt_token_bound():
    entries = [PrefixCacheEntry() for _ in range(4)]
    entries[0] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=19, prefix_tokens=8)
    entries[1] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=19, prefix_tokens=16)
    entries[2] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=19, prefix_tokens=24)

    status, best_index, best_tokens = prefix_cache_lookup_best_prefix_checked_nopartial(
        _cache(entries), 19, 12
    )
    assert status == PREFIX_CACHE_OK
    assert (best_index, best_tokens) == (0, 8)


def test_iq1293_rejects_negative_entry_fields_for_rollback_safety():
    entries = [PrefixCacheEntry() for _ in range(2)]
    entries[0] = PrefixCacheEntry(
        valid=PREFIX_CACHE_FRESH_VALID,
        prefix_hash=3,
        prefix_tokens=8,
        kv_start_token=-1,
        kv_token_count=8,
        last_used_tick=4,
    )
    status, best_index, best_tokens = prefix_cache_lookup_best_prefix_checked_nopartial(
        _cache(entries), 3, 8
    )
    assert status == PREFIX_CACHE_ERR_BAD_PARAM
    assert best_index is None
    assert best_tokens is None


def test_iq1293_not_found_and_param_guards():
    entries = [PrefixCacheEntry() for _ in range(3)]
    entries[0] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=9, prefix_tokens=4)
    cache = _cache(entries)

    status, _, _ = prefix_cache_lookup_best_prefix_checked_nopartial(cache, 99, 16)
    assert status == PREFIX_CACHE_ERR_NOT_FOUND

    bad_count = PrefixCache(entries=entries, capacity=3, count=4)
    assert prefix_cache_lookup_best_prefix_checked_nopartial(bad_count, 9, 16)[0] == PREFIX_CACHE_ERR_BAD_PARAM
    assert prefix_cache_lookup_best_prefix_checked_nopartial(None, 9, 16)[0] == PREFIX_CACHE_ERR_NULL_PTR
    assert prefix_cache_lookup_best_prefix_checked_nopartial(cache, -1, 16)[0] == PREFIX_CACHE_ERR_BAD_PARAM
    assert prefix_cache_lookup_best_prefix_checked_nopartial(cache, 9, -1)[0] == PREFIX_CACHE_ERR_BAD_PARAM


def test_iq1293_source_contains_no_partial_snapshot_guards():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    signature = (
        "I32 PrefixCacheLookupBestPrefixCheckedNoPartial(PrefixCache *cache,\n"
        "                                                I64 query_hash,\n"
        "                                                I64 max_prompt_tokens,\n"
        "                                                I64 *out_best_index,\n"
        "                                                I64 *out_best_tokens)\n"
    )
    assert signature in source
    assert "snapshot_entry_count = cache->count;" in source
    assert "if (cache->count < 0 || cache->count > cache->capacity)" in source
    assert "if (cache->entries[idx].prefix_hash < 0 ||" in source
    assert "(prefix_tokens == staged_best_tokens && idx < staged_best_index)" in source


if __name__ == "__main__":
    test_iq1293_longest_prefix_with_stable_tiebreak()
    test_iq1293_respects_max_prompt_token_bound()
    test_iq1293_rejects_negative_entry_fields_for_rollback_safety()
    test_iq1293_not_found_and_param_guards()
    test_iq1293_source_contains_no_partial_snapshot_guards()
    print("ok")
