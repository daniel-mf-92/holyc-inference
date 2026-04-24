#!/usr/bin/env python3
"""IQ-1204: parity harness for KVCacheInitQ16CheckedNoAllocCommitOnlyPreflightOnlyParity."""

KV_Q16_OK = 0
KV_Q16_ERR_NULL_PTR = 1
KV_Q16_ERR_BAD_PARAM = 2
KV_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def try_mul_i64_checked(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return KV_Q16_OK, 0
    prod = lhs * rhs
    if prod < I64_MIN or prod > I64_MAX:
        return KV_Q16_ERR_OVERFLOW, None
    return KV_Q16_OK, prod


def kv_total_cells(layer_count: int, token_capacity: int, kv_heads: int, head_dim: int):
    status, token_span = try_mul_i64_checked(kv_heads, head_dim)
    if status != KV_Q16_OK:
        return status, None
    status, layer_span = try_mul_i64_checked(token_capacity, token_span)
    if status != KV_Q16_OK:
        return status, None
    return try_mul_i64_checked(layer_count, layer_span)


def kv_init_commit_only_preflight_only_parity(
    has_k_cache: bool,
    k_cache_capacity: int,
    has_v_cache: bool,
    v_cache_capacity: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    output_alias_bad: bool,
    preflight_tuple_override=None,
    commit_tuple_override=None,
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

    status, total_cells = kv_total_cells(layer_count, token_capacity, kv_heads, head_dim)
    if status != KV_Q16_OK:
        return status, None

    if total_cells > k_cache_capacity or total_cells > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM, None

    status, layer_span = try_mul_i64_checked(token_capacity, kv_heads * head_dim)
    if status != KV_Q16_OK:
        return status, None

    preflight_tuple = (
        preflight_tuple_override
        if preflight_tuple_override is not None
        else (0, layer_span, total_cells)
    )
    commit_tuple = (
        commit_tuple_override
        if commit_tuple_override is not None
        else (0, layer_span, total_cells)
    )
    if preflight_tuple != commit_tuple:
        return KV_Q16_ERR_BAD_PARAM, None
    return KV_Q16_OK, preflight_tuple


def test_parity_success_deterministic_geometry():
    status, out = kv_init_commit_only_preflight_only_parity(
        has_k_cache=True,
        k_cache_capacity=4096,
        has_v_cache=True,
        v_cache_capacity=4096,
        layer_count=2,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
    )
    assert status == KV_Q16_OK
    assert out == (0, 256, 512)


def test_null_cache_pointer_rejected():
    status, _ = kv_init_commit_only_preflight_only_parity(
        has_k_cache=False,
        k_cache_capacity=4096,
        has_v_cache=True,
        v_cache_capacity=4096,
        layer_count=2,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
    )
    assert status == KV_Q16_ERR_NULL_PTR


def test_output_alias_rejected():
    status, _ = kv_init_commit_only_preflight_only_parity(
        has_k_cache=True,
        k_cache_capacity=4096,
        has_v_cache=True,
        v_cache_capacity=4096,
        layer_count=2,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=True,
    )
    assert status == KV_Q16_ERR_BAD_PARAM


def test_capacity_overflow_rejected():
    status, _ = kv_init_commit_only_preflight_only_parity(
        has_k_cache=True,
        k_cache_capacity=511,
        has_v_cache=True,
        v_cache_capacity=512,
        layer_count=2,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
    )
    assert status == KV_Q16_ERR_BAD_PARAM


def test_i64_overflow_rejected():
    status, _ = kv_init_commit_only_preflight_only_parity(
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


def test_tuple_mismatch_used_tokens_rejected():
    status, out = kv_init_commit_only_preflight_only_parity(
        has_k_cache=True,
        k_cache_capacity=4096,
        has_v_cache=True,
        v_cache_capacity=4096,
        layer_count=2,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
        preflight_tuple_override=(0, 256, 512),
        commit_tuple_override=(1, 256, 512),
    )
    assert status == KV_Q16_ERR_BAD_PARAM
    assert out is None


def test_tuple_mismatch_total_cells_rejected():
    status, out = kv_init_commit_only_preflight_only_parity(
        has_k_cache=True,
        k_cache_capacity=4096,
        has_v_cache=True,
        v_cache_capacity=4096,
        layer_count=2,
        token_capacity=8,
        kv_heads=4,
        head_dim=8,
        output_alias_bad=False,
        preflight_tuple_override=(0, 256, 512),
        commit_tuple_override=(0, 256, 513),
    )
    assert status == KV_Q16_ERR_BAD_PARAM
    assert out is None
