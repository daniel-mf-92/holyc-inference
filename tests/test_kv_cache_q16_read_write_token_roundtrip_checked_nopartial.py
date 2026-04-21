#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartial (IQ-877)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (  # noqa: E402
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_ERR_OVERFLOW,
    KV_Q16_OK,
    kv_cache_q16_compute_layer_token_span_cells_checked,
)
from test_kv_cache_q16_read_token_checked_nopartial import (  # noqa: E402
    kv_cache_q16_read_token_checked_nopartial,
)
from test_kv_cache_q16_write_token_checked_nopartial import (  # noqa: E402
    kv_cache_q16_write_token_checked_nopartial,
)


def kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
    k_cache_q16: list[int] | None,
    k_cache_capacity: int,
    v_cache_q16: list[int] | None,
    v_cache_capacity: int,
    layer_idx: int,
    token_idx: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    k_token_src_q16: list[int] | None,
    k_token_src_capacity: int,
    v_token_src_q16: list[int] | None,
    v_token_src_capacity: int,
    k_token_out_q16: list[int] | None,
    k_token_out_capacity: int,
    v_token_out_q16: list[int] | None,
    v_token_out_capacity: int,
) -> int:
    if (
        k_cache_q16 is None
        or v_cache_q16 is None
        or k_token_src_q16 is None
        or v_token_src_q16 is None
        or k_token_out_q16 is None
        or v_token_out_q16 is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if k_cache_q16 is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM

    if k_token_src_q16 is v_token_src_q16:
        return KV_Q16_ERR_BAD_PARAM
    if k_token_out_q16 is v_token_out_q16:
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_token_src_q16 is k_token_out_q16
        or k_token_src_q16 is v_token_out_q16
        or v_token_src_q16 is k_token_out_q16
        or v_token_src_q16 is v_token_out_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_cache_capacity < 0
        or v_cache_capacity < 0
        or k_token_src_capacity < 0
        or v_token_src_capacity < 0
        or k_token_out_capacity < 0
        or v_token_out_capacity < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        layer_idx < 0
        or token_idx < 0
        or layer_count < 0
        or token_capacity < 0
        or kv_heads < 0
        or head_dim < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    span_out = [0]
    err = kv_cache_q16_compute_layer_token_span_cells_checked(
        kv_heads,
        head_dim,
        span_out,
    )
    if err != KV_Q16_OK:
        return err

    span = span_out[0]
    if span > k_token_src_capacity or span > v_token_src_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if span > k_token_out_capacity or span > v_token_out_capacity:
        return KV_Q16_ERR_BAD_PARAM

    err = kv_cache_q16_write_token_checked_nopartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
    )
    if err != KV_Q16_OK:
        return err

    err = kv_cache_q16_read_token_checked_nopartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity,
    )
    if err != KV_Q16_OK:
        return err

    for idx in range(span):
        if k_token_out_q16[idx] != k_token_src_q16[idx]:
            return KV_Q16_ERR_BAD_PARAM
        if v_token_out_q16[idx] != v_token_src_q16[idx]:
            return KV_Q16_ERR_BAD_PARAM

    return KV_Q16_OK


def explicit_roundtrip_composition(*args: object) -> int:
    return kv_cache_q16_read_write_token_roundtrip_checked_nopartial(*args)


def test_source_contains_roundtrip_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    signature = "I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartial("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "KVCacheQ16WriteTokenCheckedNoPartial" in body
    assert "KVCacheQ16ReadTokenCheckedNoPartial" in body
    assert "KVCacheQ16ComputeLayerTokenSpanCellsChecked" in body
    assert "if (k_token_out_q16[cell_idx] != k_token_src_q16[cell_idx])" in body


def test_known_roundtrip_success() -> None:
    layer_count = 2
    token_capacity = 6
    kv_heads = 3
    head_dim = 2
    span = kv_heads * head_dim
    total = layer_count * token_capacity * span

    k_cache = [-1] * total
    v_cache = [-2] * total

    k_src = [100 + i for i in range(span)]
    v_src = [200 + i for i in range(span)]
    k_out = [777] * span
    v_out = [888] * span

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        1,
        4,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_src,
        len(k_src),
        v_src,
        len(v_src),
        k_out,
        len(k_out),
        v_out,
        len(v_out),
    )
    assert err == KV_Q16_OK
    assert k_out == k_src
    assert v_out == v_src


def test_alias_and_capacity_failures() -> None:
    k_cache = [0] * 64
    v_cache = [0] * 64
    k_src = [1] * 8
    v_src = [2] * 8
    k_out = [0] * 8
    v_out = [0] * 8

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
            None,
            64,
            v_cache,
            64,
            0,
            0,
            1,
            8,
            2,
            4,
            k_src,
            8,
            v_src,
            8,
            k_out,
            8,
            v_out,
            8,
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
            k_cache,
            64,
            v_cache,
            64,
            0,
            0,
            1,
            8,
            2,
            4,
            k_src,
            8,
            v_src,
            8,
            k_src,
            8,
            v_out,
            8,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
            k_cache,
            64,
            v_cache,
            64,
            0,
            0,
            1,
            8,
            2,
            4,
            k_src,
            7,
            v_src,
            8,
            k_out,
            8,
            v_out,
            8,
        )
        == KV_Q16_ERR_BAD_PARAM
    )


def test_overflow_and_randomized_parity() -> None:
    overflow = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
        [0],
        1,
        [0],
        1,
        0,
        0,
        1,
        1,
        I64_MAX,
        2,
        [1],
        1,
        [1],
        1,
        [0],
        1,
        [0],
        1,
    )
    assert overflow == KV_Q16_ERR_OVERFLOW

    rng = random.Random(877)
    for _ in range(120):
        layer_count = rng.randint(1, 4)
        token_capacity = rng.randint(1, 8)
        kv_heads = rng.randint(1, 4)
        head_dim = rng.randint(1, 8)
        span = kv_heads * head_dim
        total = layer_count * token_capacity * span

        k_cache = [-(i + 1) for i in range(total)]
        v_cache = [10000 + i for i in range(total)]

        layer_idx = rng.randrange(layer_count)
        token_idx = rng.randrange(token_capacity)

        k_src = [rng.randint(-32768, 32767) for _ in range(span)]
        v_src = [rng.randint(-32768, 32767) for _ in range(span)]
        k_out = [0] * span
        v_out = [0] * span

        args = (
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            k_out,
            len(k_out),
            v_out,
            len(v_out),
        )

        got = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(*args)
        want = explicit_roundtrip_composition(*args)
        assert got == want == KV_Q16_OK
        assert k_out == k_src
        assert v_out == v_src
