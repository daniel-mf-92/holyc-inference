#!/usr/bin/env python3
"""IQ-1274: parity-commit-only harness for KVCacheInitQ16CheckedNoAlloc init path."""

KV_Q16_OK = 0
KV_Q16_ERR_NULL_PTR = 1
KV_Q16_ERR_BAD_PARAM = 2
KV_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def try_mul_i64_checked(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return KV_Q16_OK, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return KV_Q16_ERR_OVERFLOW, None
    return KV_Q16_OK, product


def kv_total_cells(layer_count: int, token_capacity: int, kv_heads: int, head_dim: int):
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
    has_v_cache: bool,
    k_cache_capacity: int,
    v_cache_capacity: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    output_alias_bad: bool,
):
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

    preflight_tuple = (0, layer_span, total)
    parity_tuple = (0, layer_span, total)
    if parity_tuple != preflight_tuple:
        return KV_Q16_ERR_BAD_PARAM, None

    return KV_Q16_OK, parity_tuple


def test_parity_commit_only_success_deterministic_tuple():
    status, out = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=True,
        has_v_cache=True,
        k_cache_capacity=8192,
        v_cache_capacity=8192,
        layer_count=4,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
    )
    assert status == KV_Q16_OK
    assert out == (0, 256, 1024)


def test_null_pointer_rejected():
    status, _ = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=False,
        has_v_cache=True,
        k_cache_capacity=256,
        v_cache_capacity=256,
        layer_count=1,
        token_capacity=2,
        kv_heads=4,
        head_dim=4,
        output_alias_bad=False,
    )
    assert status == KV_Q16_ERR_NULL_PTR


def test_output_alias_rejected():
    status, _ = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=True,
        has_v_cache=True,
        k_cache_capacity=256,
        v_cache_capacity=256,
        layer_count=1,
        token_capacity=2,
        kv_heads=4,
        head_dim=4,
        output_alias_bad=True,
    )
    assert status == KV_Q16_ERR_BAD_PARAM


def test_capacity_insufficient_rejected():
    status, _ = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=True,
        has_v_cache=True,
        k_cache_capacity=511,
        v_cache_capacity=512,
        layer_count=2,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
    )
    assert status == KV_Q16_ERR_BAD_PARAM


def test_overflow_rejected():
    status, _ = kv_init_commit_only_preflight_only_parity_commit_only(
        has_k_cache=True,
        has_v_cache=True,
        k_cache_capacity=I64_MAX,
        v_cache_capacity=I64_MAX,
        layer_count=1 << 62,
        token_capacity=4,
        kv_heads=4,
        head_dim=4,
        output_alias_bad=False,
    )
    assert status == KV_Q16_ERR_OVERFLOW
