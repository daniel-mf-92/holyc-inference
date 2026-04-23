#!/usr/bin/env python3
"""Host-side parity harness for src/runtime/prefix_cache.HC (IQ-1267)."""

from __future__ import annotations

from dataclasses import dataclass


PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_NOT_FOUND = 4


@dataclass
class PrefixCacheEntry:
    valid: int = 0
    prefix_hash: int = 0
    prefix_tokens: int = 0
    kv_start_token: int = 0
    kv_token_count: int = 0
    last_used_tick: int = 0


class PrefixCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.entries = [PrefixCacheEntry() for _ in range(capacity)]
        self.capacity = capacity
        self.count = 0

    def clear(self) -> int:
        for idx in range(self.capacity):
            self.entries[idx] = PrefixCacheEntry()
        self.count = 0
        return PREFIX_CACHE_OK

    def _find_index(self, prefix_hash: int, prefix_tokens: int):
        if prefix_hash < 0 or prefix_tokens < 0:
            return PREFIX_CACHE_ERR_BAD_PARAM, None
        for idx, entry in enumerate(self.entries):
            if entry.valid and entry.prefix_hash == prefix_hash and entry.prefix_tokens == prefix_tokens:
                return PREFIX_CACHE_OK, idx
        return PREFIX_CACHE_ERR_NOT_FOUND, None

    def lookup(self, prefix_hash: int, prefix_tokens: int, access_tick: int):
        if access_tick < 0:
            return PREFIX_CACHE_ERR_BAD_PARAM, None
        status, idx = self._find_index(prefix_hash, prefix_tokens)
        if status != PREFIX_CACHE_OK:
            return status, None
        entry = self.entries[idx]
        entry.last_used_tick = access_tick
        return PREFIX_CACHE_OK, (entry.kv_start_token, entry.kv_token_count, idx)

    def _victim_lru(self):
        for idx, entry in enumerate(self.entries):
            if not entry.valid:
                return idx
        oldest_idx = 0
        oldest_tick = self.entries[0].last_used_tick
        for idx in range(1, self.capacity):
            if self.entries[idx].last_used_tick < oldest_tick:
                oldest_tick = self.entries[idx].last_used_tick
                oldest_idx = idx
        return oldest_idx

    def insert_or_update(
        self,
        prefix_hash: int,
        prefix_tokens: int,
        kv_start_token: int,
        kv_token_count: int,
        access_tick: int,
    ):
        if min(prefix_hash, prefix_tokens, kv_start_token, kv_token_count, access_tick) < 0:
            return PREFIX_CACHE_ERR_BAD_PARAM, None

        status, idx = self._find_index(prefix_hash, prefix_tokens)
        if status == PREFIX_CACHE_OK:
            e = self.entries[idx]
            e.kv_start_token = kv_start_token
            e.kv_token_count = kv_token_count
            e.last_used_tick = access_tick
            return PREFIX_CACHE_OK, (idx, 0)

        victim = self._victim_lru()
        inserted_new = 0 if self.entries[victim].valid else 1
        if inserted_new:
            self.count += 1
        self.entries[victim] = PrefixCacheEntry(
            valid=1,
            prefix_hash=prefix_hash,
            prefix_tokens=prefix_tokens,
            kv_start_token=kv_start_token,
            kv_token_count=kv_token_count,
            last_used_tick=access_tick,
        )
        return PREFIX_CACHE_OK, (victim, 1)


def test_insert_lookup_update_roundtrip():
    cache = PrefixCache(capacity=3)

    status, meta = cache.insert_or_update(101, 12, 200, 12, 7)
    assert status == PREFIX_CACHE_OK
    assert meta == (0, 1)
    assert cache.count == 1

    status, hit = cache.lookup(101, 12, 9)
    assert status == PREFIX_CACHE_OK
    assert hit == (200, 12, 0)

    status, meta = cache.insert_or_update(101, 12, 240, 14, 11)
    assert status == PREFIX_CACHE_OK
    assert meta == (0, 0)
    assert cache.count == 1

    status, hit = cache.lookup(101, 12, 12)
    assert status == PREFIX_CACHE_OK
    assert hit == (240, 14, 0)


def test_lru_eviction_on_full_capacity():
    cache = PrefixCache(capacity=2)

    assert cache.insert_or_update(1, 8, 10, 8, 1)[0] == PREFIX_CACHE_OK
    assert cache.insert_or_update(2, 9, 20, 9, 2)[0] == PREFIX_CACHE_OK
    assert cache.count == 2

    # Refresh key=2 so key=1 becomes oldest.
    assert cache.lookup(2, 9, 5)[0] == PREFIX_CACHE_OK

    status, meta = cache.insert_or_update(3, 10, 30, 10, 6)
    assert status == PREFIX_CACHE_OK
    # Evicts entry slot 0 that held key=1.
    assert meta == (0, 1)

    assert cache.lookup(1, 8, 7)[0] == PREFIX_CACHE_ERR_NOT_FOUND
    assert cache.lookup(2, 9, 7)[0] == PREFIX_CACHE_OK
    assert cache.lookup(3, 10, 7)[0] == PREFIX_CACHE_OK


def test_bad_params_and_clear_reset():
    cache = PrefixCache(capacity=2)

    status, _ = cache.insert_or_update(-1, 4, 0, 4, 1)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM

    status, _ = cache.lookup(0, 0, -2)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM

    assert cache.insert_or_update(9, 4, 99, 4, 3)[0] == PREFIX_CACHE_OK
    assert cache.count == 1
    assert cache.clear() == PREFIX_CACHE_OK
    assert cache.count == 0
    assert cache.lookup(9, 4, 4)[0] == PREFIX_CACHE_ERR_NOT_FOUND


if __name__ == "__main__":
    test_insert_lookup_update_roundtrip()
    test_lru_eviction_on_full_capacity()
    test_bad_params_and_clear_reset()
    print("ok")
