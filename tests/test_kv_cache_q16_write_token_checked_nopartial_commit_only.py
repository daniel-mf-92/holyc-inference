#!/usr/bin/env python3
"""Parity harness for KVCacheQ16WriteTokenCheckedNoPartialCommitOnly (IQ-876)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_OK,
    kv_cache_q16_compute_layer_token_base_index_checked,
)
from test_kv_cache_q16_write_token_checked_nopartial import (
    kv_cache_q16_write_token_checked_nopartial,
)


def kv_cache_q16_write_token_checked_nopartial_preflight(
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
    out_k_base_index: list[int] | None,
    out_v_base_index: list[int] | None,
    out_span_cells: list[int] | None,
) -> int:
    if out_k_base_index is None or out_v_base_index is None or out_span_cells is None:
        return KV_Q16_ERR_NULL_PTR

    if out_k_base_index is out_v_base_index or out_k_base_index is out_span_cells or out_v_base_index is out_span_cells:
        return KV_Q16_ERR_BAD_PARAM

    if k_cache_q16 is None or v_cache_q16 is None or k_token_src_q16 is None or v_token_src_q16 is None:
        return KV_Q16_ERR_NULL_PTR

    if k_cache_q16 is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM

    if k_cache_capacity < 0 or v_cache_capacity < 0 or k_token_src_capacity < 0 or v_token_src_capacity < 0:
        return KV_Q16_ERR_BAD_PARAM

    if layer_idx < 0 or token_idx < 0 or layer_count < 0 or token_capacity < 0 or kv_heads < 0 or head_dim < 0:
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_token_src_q16 is k_cache_q16
        or k_token_src_q16 is v_cache_q16
        or v_token_src_q16 is k_cache_q16
        or v_token_src_q16 is v_cache_q16
        or k_token_src_q16 is v_token_src_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    k_base = [0]
    span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_base,
        span,
    )
    if err != KV_Q16_OK:
        return err

    if k_base[0] + span[0] > k_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if k_base[0] + span[0] > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if span[0] > k_token_src_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if span[0] > v_token_src_capacity:
        return KV_Q16_ERR_BAD_PARAM

    out_k_base_index[0] = k_base[0]
    out_v_base_index[0] = k_base[0]
    out_span_cells[0] = span[0]
    return KV_Q16_OK


def kv_cache_q16_write_token_checked_nopartial_commit_only(
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
    out_required_span_cells: list[int] | None,
    out_k_base_index: list[int] | None,
    out_v_base_index: list[int] | None,
) -> int:
    if out_required_span_cells is None or out_k_base_index is None or out_v_base_index is None:
        return KV_Q16_ERR_NULL_PTR
    if out_required_span_cells is out_k_base_index or out_required_span_cells is out_v_base_index or out_k_base_index is out_v_base_index:
        return KV_Q16_ERR_BAD_PARAM

    staged_k_base = [0]
    staged_v_base = [0]
    staged_span = [0]
    err = kv_cache_q16_write_token_checked_nopartial_preflight(
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
        staged_k_base,
        staged_v_base,
        staged_span,
    )
    if err != KV_Q16_OK:
        return err

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

    out_required_span_cells[0] = staged_span[0]
    out_k_base_index[0] = staged_k_base[0]
    out_v_base_index[0] = staged_v_base[0]
    return KV_Q16_OK


def test_source_contains_commit_only_helpers() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    assert "I32 KVCacheQ16WriteTokenCheckedNoPartialPreflight(" in source
    assert "I32 KVCacheQ16WriteTokenCheckedNoPartialCommitOnly(" in source
    assert "KVCacheQ16WriteTokenCheckedNoPartialPreflight" in source
    assert "KVCacheQ16WriteTokenCheckedNoPartial(" in source


def test_known_vector_commit_and_diagnostics() -> None:
    layer_count = 3
    token_capacity = 5
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [-1] * total_cells
    v_cache = [-2] * total_cells
    k_src = [111 + idx for idx in range(span)]
    v_src = [222 + idx for idx in range(span)]

    out_span = [0]
    out_k_base = [0]
    out_v_base = [0]

    err = kv_cache_q16_write_token_checked_nopartial_commit_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=2,
        token_idx=1,
        layer_count=layer_count,
        token_capacity=token_capacity,
        kv_heads=kv_heads,
        head_dim=head_dim,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        out_required_span_cells=out_span,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_OK

    expected_base = ((2 * token_capacity) + 1) * span
    assert out_span == [span]
    assert out_k_base == [expected_base]
    assert out_v_base == [expected_base]
    assert k_cache[expected_base : expected_base + span] == k_src
    assert v_cache[expected_base : expected_base + span] == v_src


def test_null_alias_and_no_partial_failure() -> None:
    k_cache = [5] * 32
    v_cache = [6] * 32
    k_src = [7] * 8
    v_src = [8] * 8

    assert (
        kv_cache_q16_write_token_checked_nopartial_commit_only(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            0,
            0,
            1,
            4,
            2,
            4,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            None,
            [0],
            [0],
        )
        == KV_Q16_ERR_NULL_PTR
    )

    out_same = [0]
    assert (
        kv_cache_q16_write_token_checked_nopartial_commit_only(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            0,
            0,
            1,
            4,
            2,
            4,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            out_same,
            out_same,
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    k_before = k_cache.copy()
    v_before = v_cache.copy()
    out_span = [123]
    out_k_base = [456]
    out_v_base = [789]

    err = kv_cache_q16_write_token_checked_nopartial_commit_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=0,
        token_idx=99,
        layer_count=1,
        token_capacity=4,
        kv_heads=2,
        head_dim=4,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        out_required_span_cells=out_span,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert k_cache == k_before
    assert v_cache == v_before
    assert out_span == [123]
    assert out_k_base == [456]
    assert out_v_base == [789]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(876)

    for _ in range(600):
        layer_count = rng.randint(1, 8)
        token_capacity = rng.randint(1, 16)
        kv_heads = rng.randint(1, 8)
        head_dim = rng.randint(1, 16)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        k_cache_a = [rng.randint(-200, 200) for _ in range(total_cells)]
        v_cache_a = [rng.randint(-200, 200) for _ in range(total_cells)]
        k_cache_b = k_cache_a.copy()
        v_cache_b = v_cache_a.copy()

        k_src = [rng.randint(-50, 50) for _ in range(span)]
        v_src = [rng.randint(-50, 50) for _ in range(span)]

        out_span_a = [0]
        out_k_base_a = [0]
        out_v_base_a = [0]
        err_a = kv_cache_q16_write_token_checked_nopartial_commit_only(
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
            out_span_a,
            out_k_base_a,
            out_v_base_a,
        )

        pre_k_base = [0]
        pre_v_base = [0]
        pre_span = [0]
        err_pre = kv_cache_q16_write_token_checked_nopartial_preflight(
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
            pre_k_base,
            pre_v_base,
            pre_span,
        )
        if err_pre == KV_Q16_OK:
            err_exp = kv_cache_q16_write_token_checked_nopartial(
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
            )
            assert err_exp == KV_Q16_OK
        else:
            err_exp = err_pre

        assert err_a == err_exp
        assert k_cache_a == k_cache_b
        assert v_cache_a == v_cache_b

        if err_a == KV_Q16_OK:
            assert out_span_a == pre_span
            assert out_k_base_a == pre_k_base
            assert out_v_base_a == pre_v_base


if __name__ == "__main__":
    test_source_contains_commit_only_helpers()
    test_known_vector_commit_and_diagnostics()
    test_null_alias_and_no_partial_failure()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
