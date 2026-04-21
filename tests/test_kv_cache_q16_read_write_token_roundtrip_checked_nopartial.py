#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartial (IQ-877)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_ERR_OVERFLOW,
    KV_Q16_OK,
    kv_cache_q16_compute_layer_token_span_cells_checked,
)
from test_kv_cache_q16_read_token_checked_nopartial import (
    kv_cache_q16_read_token_checked_nopartial,
)
from test_kv_cache_q16_write_token_checked_nopartial import (
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
        k_token_out_q16 is k_cache_q16
        or k_token_out_q16 is v_cache_q16
        or v_token_out_q16 is k_cache_q16
        or v_token_out_q16 is v_cache_q16
    ):
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

    span_cells = [0]
    err = kv_cache_q16_compute_layer_token_span_cells_checked(
        kv_heads, head_dim, span_cells
    )
    if err != KV_Q16_OK:
        return err

    if span_cells[0] > k_token_src_capacity or span_cells[0] > v_token_src_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if span_cells[0] > k_token_out_capacity or span_cells[0] > v_token_out_capacity:
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

    for idx in range(span_cells[0]):
        if k_token_out_q16[idx] != k_token_src_q16[idx]:
            return KV_Q16_ERR_BAD_PARAM
        if v_token_out_q16[idx] != v_token_src_q16[idx]:
            return KV_Q16_ERR_BAD_PARAM

    return KV_Q16_OK


def explicit_roundtrip_composition(
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
    return kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
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
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity,
    )


def test_source_contains_roundtrip_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    signature = "I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartial("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "KVCacheQ16ComputeLayerTokenSpanCellsChecked" in body
    assert "KVCacheQ16WriteTokenCheckedNoPartial(" in body
    assert "KVCacheQ16ReadTokenCheckedNoPartial(" in body
    assert "while (cell_idx < span_cells)" in body


def test_known_vector_success_roundtrip() -> None:
    layer_count = 3
    token_capacity = 6
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [-101] * total_cells
    v_cache = [-202] * total_cells

    k_src = [1000 + idx for idx in range(span)]
    v_src = [2000 + idx for idx in range(span)]

    k_out = [11] * span
    v_out = [22] * span

    layer_idx = 1
    token_idx = 4

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
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
    assert err == KV_Q16_OK
    assert k_out == k_src
    assert v_out == v_src

    base = ((layer_idx * token_capacity) + token_idx) * span
    assert k_cache[base : base + span] == k_src
    assert v_cache[base : base + span] == v_src


def test_output_capacity_guard_is_no_partial() -> None:
    layer_count = 2
    token_capacity = 5
    kv_heads = 2
    head_dim = 3
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [7] * total_cells
    v_cache = [8] * total_cells

    k_src = [300 + idx for idx in range(span)]
    v_src = [600 + idx for idx in range(span)]

    k_out = [42] * span
    v_out = [43] * span

    before_k_cache = k_cache.copy()
    before_v_cache = v_cache.copy()
    before_k_out = k_out.copy()
    before_v_out = v_out.copy()

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=0,
        token_idx=1,
        layer_count=layer_count,
        token_capacity=token_capacity,
        kv_heads=kv_heads,
        head_dim=head_dim,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        k_token_out_q16=k_out,
        k_token_out_capacity=span - 1,
        v_token_out_q16=v_out,
        v_token_out_capacity=span,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert k_cache == before_k_cache
    assert v_cache == before_v_cache
    assert k_out == before_k_out
    assert v_out == before_v_out


def test_overflow_passthrough_from_span_helper() -> None:
    k_cache = [0] * 8
    v_cache = [0] * 8
    k_src = [1] * 2
    v_src = [2] * 2
    k_out = [3] * 2
    v_out = [4] * 2

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=0,
        token_idx=0,
        layer_count=1,
        token_capacity=1,
        kv_heads=I64_MAX,
        head_dim=2,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        k_token_out_q16=k_out,
        k_token_out_capacity=len(k_out),
        v_token_out_q16=v_out,
        v_token_out_capacity=len(v_out),
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_parity_and_determinism() -> None:
    rng = random.Random(877)

    for _ in range(400):
        layer_count = rng.randint(1, 6)
        token_capacity = rng.randint(1, 20)
        kv_heads = rng.randint(1, 8)
        head_dim = rng.randint(1, 16)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        k_cache_a = [rng.randint(-5000, 5000) for _ in range(total_cells)]
        v_cache_a = [rng.randint(-5000, 5000) for _ in range(total_cells)]
        k_cache_b = k_cache_a.copy()
        v_cache_b = v_cache_a.copy()

        k_src = [rng.randint(-3000, 3000) for _ in range(span)]
        v_src = [rng.randint(-3000, 3000) for _ in range(span)]

        k_out_a = [rng.randint(-111, 111) for _ in range(span)]
        v_out_a = [rng.randint(-111, 111) for _ in range(span)]
        k_out_b = k_out_a.copy()
        v_out_b = v_out_a.copy()

        got = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
            k_cache_a,
            len(k_cache_a),
            v_cache_a,
            len(v_cache_a),
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
            k_out_a,
            len(k_out_a),
            v_out_a,
            len(v_out_a),
        )

        exp = explicit_roundtrip_composition(
            k_cache_b,
            len(k_cache_b),
            v_cache_b,
            len(v_cache_b),
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
            k_out_b,
            len(k_out_b),
            v_out_b,
            len(v_out_b),
        )

        assert got == exp
        assert k_cache_a == k_cache_b
        assert v_cache_a == v_cache_b
        assert k_out_a == k_out_b
        assert v_out_a == v_out_b


if __name__ == "__main__":
    raise SystemExit(
        __import__("pytest").main([__file__, "-q"])
    )
