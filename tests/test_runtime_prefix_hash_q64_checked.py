#!/usr/bin/env python3
"""Host-side harness for IQ-1287 prefix hash (`InferencePromptPrefixHashQ64Checked`)."""

from __future__ import annotations

from pathlib import Path

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2

PREFIX_CACHE_HASH_FNV_OFFSET_BASIS = 1469598103934665603
PREFIX_CACHE_HASH_FNV_PRIME = 1099511628211
PREFIX_CACHE_HASH_DOMAIN_TAG = 0x5052454649584841
MASK_U64 = (1 << 64) - 1
MASK_I63 = (1 << 63) - 1


def _holyc_like_prefix_hash_checked(
    tokens: list[int] | None,
    token_count: int,
    *,
    out_is_null: bool = False,
    tokens_is_null: bool = False,
) -> tuple[int, int | None]:
    if tokens_is_null or out_is_null:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    hash_u64 = PREFIX_CACHE_HASH_FNV_OFFSET_BASIS
    hash_u64 = (hash_u64 ^ PREFIX_CACHE_HASH_DOMAIN_TAG) & MASK_U64
    hash_u64 = (hash_u64 * PREFIX_CACHE_HASH_FNV_PRIME) & MASK_U64

    idx = 0
    while idx < token_count:
        lane = tokens[idx] & 0xFF  # type: ignore[index]
        hash_u64 = (hash_u64 ^ lane) & MASK_U64
        hash_u64 = (hash_u64 * PREFIX_CACHE_HASH_FNV_PRIME) & MASK_U64
        idx += 1

    count_u64 = token_count & MASK_U64
    hash_u64 = (hash_u64 ^ 0xFF) & MASK_U64
    hash_u64 = (hash_u64 * PREFIX_CACHE_HASH_FNV_PRIME) & MASK_U64
    hash_u64 = (hash_u64 ^ (count_u64 & 0xFFFFFFFF)) & MASK_U64
    hash_u64 = (hash_u64 * PREFIX_CACHE_HASH_FNV_PRIME) & MASK_U64
    hash_u64 = (hash_u64 ^ (count_u64 >> 32)) & MASK_U64
    hash_u64 = (hash_u64 * PREFIX_CACHE_HASH_FNV_PRIME) & MASK_U64
    hash_u64 = (hash_u64 ^ count_u64) & MASK_U64
    hash_u64 = (hash_u64 * PREFIX_CACHE_HASH_FNV_PRIME) & MASK_U64

    return PREFIX_CACHE_OK, (hash_u64 & MASK_I63)


def test_source_contains_iq1287_symbols() -> None:
    src = Path("src/runtime/prefix_cache.HC").read_text(encoding="utf-8")
    assert "I32 InferencePromptPrefixHashQ64Checked(" in src
    assert "PREFIX_CACHE_HASH_FNV_OFFSET_BASIS" in src
    assert "PREFIX_CACHE_HASH_FNV_PRIME" in src
    assert "PREFIX_CACHE_HASH_DOMAIN_TAG" in src
    assert "token_count_u64" in src
    assert "hash_u64" in src


def test_null_ptr_guard() -> None:
    status, out_hash = _holyc_like_prefix_hash_checked([1, 2], 2, out_is_null=True)
    assert status == PREFIX_CACHE_ERR_NULL_PTR
    assert out_hash is None

    status, out_hash = _holyc_like_prefix_hash_checked(None, 2, tokens_is_null=True)
    assert status == PREFIX_CACHE_ERR_NULL_PTR
    assert out_hash is None


def test_negative_count_guard() -> None:
    status, out_hash = _holyc_like_prefix_hash_checked([], -1)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM
    assert out_hash is None


def test_known_vectors() -> None:
    vectors = [
        [],
        [0],
        [1],
        [255],
        [1, 2, 3, 4],
        [255, 0, 255, 0, 1, 2, 3],
        list(range(32)),
    ]
    expected = [
        3538738319256905833,
        8436475944519613765,
        6523927703260884690,
        8611418744548151548,
        4524405985828232077,
        5939378721635954397,
        1791953694061267433,
    ]

    observed: list[int] = []
    for toks in vectors:
        status, value = _holyc_like_prefix_hash_checked(toks, len(toks))
        assert status == PREFIX_CACHE_OK
        assert value is not None
        observed.append(value)

    assert observed == expected


def test_length_mixing_domain_separation() -> None:
    base_tokens = [7, 8, 9, 10]
    status_a, hash_a = _holyc_like_prefix_hash_checked(base_tokens, 4)
    status_b, hash_b = _holyc_like_prefix_hash_checked(base_tokens + [0], 5)
    assert status_a == PREFIX_CACHE_OK
    assert status_b == PREFIX_CACHE_OK
    assert hash_a != hash_b


def test_high_bit_is_cleared() -> None:
    for token_count in [0, 1, 2, 7, 32, 128, 255]:
        toks = [((i * 13) + 5) & 0xFF for i in range(token_count)]
        status, value = _holyc_like_prefix_hash_checked(toks, len(toks))
        assert status == PREFIX_CACHE_OK
        assert value is not None
        assert 0 <= value <= MASK_I63


if __name__ == "__main__":
    test_source_contains_iq1287_symbols()
    test_null_ptr_guard()
    test_negative_count_guard()
    test_known_vectors()
    test_length_mixing_domain_separation()
    test_high_bit_is_cleared()
    print("ok")
