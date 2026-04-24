#!/usr/bin/env python3
"""IQ-1205: harness for KVCacheInitQ16CheckedNoAllocCommitOnlyPreflightOnlyParityCommitOnly."""

from __future__ import annotations

from pathlib import Path

KV_Q16_OK = 0
KV_Q16_ERR_NULL_PTR = 1
KV_Q16_ERR_BAD_PARAM = 2
KV_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int | None]:
    if lhs == 0 or rhs == 0:
        return KV_Q16_OK, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return KV_Q16_ERR_OVERFLOW, None
    return KV_Q16_OK, product


def kv_total_cells(
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[int, int | None, int | None]:
    status, token_span = try_mul_i64_checked(kv_heads, head_dim)
    if status != KV_Q16_OK:
        return status, None, None
    status, layer_span = try_mul_i64_checked(token_capacity, token_span)
    if status != KV_Q16_OK:
        return status, None, None
    status, total = try_mul_i64_checked(layer_count, layer_span)
    if status != KV_Q16_OK:
        return status, None, None
    return KV_Q16_OK, layer_span, total


def kv_init_commit_only_preflight_only_parity_commit_only(
    has_k_cache: bool,
    k_cache_capacity: int,
    has_v_cache: bool,
    v_cache_capacity: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    output_alias_bad: bool,
    parity_tuple_override: tuple[int, int, int] | None = None,
    preflight_tuple_override: tuple[int, int, int] | None = None,
) -> tuple[int, tuple[int, int, int] | None]:
    if not has_k_cache or not has_v_cache:
        return KV_Q16_ERR_NULL_PTR, None

    if output_alias_bad:
        return KV_Q16_ERR_BAD_PARAM, None

    if (
        k_cache_capacity < 0
        or v_cache_capacity < 0
        or layer_count < 0
        or token_capacity < 0
        or kv_heads < 0
        or head_dim < 0
    ):
        return KV_Q16_ERR_BAD_PARAM, None

    status, layer_span, total = kv_total_cells(layer_count, token_capacity, kv_heads, head_dim)
    if status != KV_Q16_OK:
        return status, None

    if total > k_cache_capacity or total > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM, None

    canonical = (0, layer_span, total)
    parity_tuple = parity_tuple_override if parity_tuple_override is not None else canonical
    preflight_tuple = preflight_tuple_override if preflight_tuple_override is not None else canonical

    if parity_tuple != preflight_tuple:
        return KV_Q16_ERR_BAD_PARAM, None

    if parity_tuple[0] != 0:
        return KV_Q16_ERR_BAD_PARAM, None

    return KV_Q16_OK, parity_tuple


def test_source_contains_iq1205_contract() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheInitQ16CheckedNoAllocCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheInitQ16CheckedNoAllocCommitOnlyPreflightOnlyParity(" in body
    assert "KVCacheInitQ16CheckedNoAllocCommitOnlyPreflightOnly(" in body
    assert "snapshot_k_cache_capacity" in body
    assert "snapshot_v_cache_capacity" in body
    assert "staged_parity_used_tokens != staged_preflight_used_tokens" in body
    assert "if (staged_parity_used_tokens != 0)" in body
    assert "*out_used_tokens = staged_parity_used_tokens;" in body
    assert "*out_layer_span_cells = staged_parity_layer_span_cells;" in body
    assert "*out_total_cells = staged_parity_total_cells;" in body


def test_deterministic_parity_success() -> None:
    status, out = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=True,
        k_cache_capacity=8192,
        has_v_cache=True,
        v_cache_capacity=8192,
        layer_count=4,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
    )
    assert status == KV_Q16_OK
    assert out == (0, 256, 1024)


def test_null_pointer_rejected() -> None:
    status, out = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=False,
        k_cache_capacity=8192,
        has_v_cache=True,
        v_cache_capacity=8192,
        layer_count=4,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
    )
    assert status == KV_Q16_ERR_NULL_PTR
    assert out is None


def test_output_alias_rejected() -> None:
    status, out = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=True,
        k_cache_capacity=8192,
        has_v_cache=True,
        v_cache_capacity=8192,
        layer_count=4,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=True,
    )
    assert status == KV_Q16_ERR_BAD_PARAM
    assert out is None


def test_overflow_rejected() -> None:
    status, out = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=True,
        k_cache_capacity=I64_MAX,
        has_v_cache=True,
        v_cache_capacity=I64_MAX,
        layer_count=1 << 62,
        token_capacity=4,
        kv_heads=4,
        head_dim=4,
        output_alias_bad=False,
    )
    assert status == KV_Q16_ERR_OVERFLOW
    assert out is None


def test_tuple_mismatch_rejected() -> None:
    status, out = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=True,
        k_cache_capacity=8192,
        has_v_cache=True,
        v_cache_capacity=8192,
        layer_count=4,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
        parity_tuple_override=(0, 256, 1024),
        preflight_tuple_override=(0, 256, 1025),
    )
    assert status == KV_Q16_ERR_BAD_PARAM
    assert out is None


def test_nonzero_used_tokens_rejected() -> None:
    status, out = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=True,
        k_cache_capacity=8192,
        has_v_cache=True,
        v_cache_capacity=8192,
        layer_count=4,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
        parity_tuple_override=(1, 256, 1024),
        preflight_tuple_override=(1, 256, 1024),
    )
    assert status == KV_Q16_ERR_BAD_PARAM
    assert out is None


if __name__ == "__main__":
    test_source_contains_iq1205_contract()
    test_deterministic_parity_success()
    test_null_pointer_rejected()
    test_output_alias_rejected()
    test_overflow_rejected()
    test_tuple_mismatch_rejected()
    test_nonzero_used_tokens_rejected()
    print("ok")
