#!/usr/bin/env python3
"""Harness for KVCacheQ16ReserveAndWriteTokenChecked (IQ-1218)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only_parity import (
    kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only,
)
from test_kv_cache_q16_indexing_checked import (
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_OK,
)
from test_kv_cache_q16_write_token_checked_nopartial_commit_only import (
    kv_cache_q16_write_token_checked_nopartial_commit_only,
)


def kv_cache_q16_reserve_and_write_token_checked(
    k_cache_q16: list[int] | None,
    k_cache_capacity: int,
    v_cache_q16: list[int] | None,
    v_cache_capacity: int,
    layer_idx: int,
    token_count: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    k_token_src_q16: list[int] | None,
    k_token_src_capacity: int,
    v_token_src_q16: list[int] | None,
    v_token_src_capacity: int,
    out_required_cells: list[int] | None,
    out_required_bytes: list[int] | None,
    out_written_token_index: list[int] | None,
    out_next_token_count: list[int] | None,
) -> int:
    if (
        out_required_cells is None
        or out_required_bytes is None
        or out_written_token_index is None
        or out_next_token_count is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if (
        out_required_cells is out_required_bytes
        or out_required_cells is out_written_token_index
        or out_required_cells is out_next_token_count
        or out_required_bytes is out_written_token_index
        or out_required_bytes is out_next_token_count
        or out_written_token_index is out_next_token_count
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_required_cells is k_cache_q16
        or out_required_cells is v_cache_q16
        or out_required_cells is k_token_src_q16
        or out_required_cells is v_token_src_q16
        or out_required_bytes is k_cache_q16
        or out_required_bytes is v_cache_q16
        or out_required_bytes is k_token_src_q16
        or out_required_bytes is v_token_src_q16
        or out_written_token_index is k_cache_q16
        or out_written_token_index is v_cache_q16
        or out_written_token_index is k_token_src_q16
        or out_written_token_index is v_token_src_q16
        or out_next_token_count is k_cache_q16
        or out_next_token_count is v_cache_q16
        or out_next_token_count is k_token_src_q16
        or out_next_token_count is v_token_src_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    if token_count < 0 or token_capacity < 0:
        return KV_Q16_ERR_BAD_PARAM
    if token_count >= token_capacity:
        return KV_Q16_ERR_BAD_PARAM

    snapshot = (
        layer_idx,
        token_count,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_cache_capacity,
        v_cache_capacity,
        k_token_src_capacity,
        v_token_src_capacity,
        id(k_cache_q16),
        id(v_cache_q16),
        id(k_token_src_q16),
        id(v_token_src_q16),
        id(out_required_cells),
        id(out_required_bytes),
        id(out_written_token_index),
        id(out_next_token_count),
    )

    staged_required_cells = [0]
    staged_required_bytes = [0]
    staged_written_token = [0]
    err = kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_count,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        staged_required_cells,
        staged_required_bytes,
        staged_written_token,
    )
    if err != KV_Q16_OK:
        return err

    staged_span = [0]
    staged_k_base = [0]
    staged_v_base = [0]
    err = kv_cache_q16_write_token_checked_nopartial_commit_only(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        staged_written_token[0],
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        staged_span,
        staged_k_base,
        staged_v_base,
    )
    if err != KV_Q16_OK:
        return err

    if staged_span[0] != staged_required_cells[0]:
        return KV_Q16_ERR_BAD_PARAM
    if staged_k_base[0] != staged_v_base[0]:
        return KV_Q16_ERR_BAD_PARAM

    staged_next_token_count = staged_written_token[0] + 1
    if staged_next_token_count > token_capacity:
        return KV_Q16_ERR_BAD_PARAM

    if snapshot != (
        layer_idx,
        token_count,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_cache_capacity,
        v_cache_capacity,
        k_token_src_capacity,
        v_token_src_capacity,
        id(k_cache_q16),
        id(v_cache_q16),
        id(k_token_src_q16),
        id(v_token_src_q16),
        id(out_required_cells),
        id(out_required_bytes),
        id(out_written_token_index),
        id(out_next_token_count),
    ):
        return KV_Q16_ERR_BAD_PARAM

    out_required_cells[0] = staged_required_cells[0]
    out_required_bytes[0] = staged_required_bytes[0]
    out_written_token_index[0] = staged_written_token[0]
    out_next_token_count[0] = staged_next_token_count
    return KV_Q16_OK


def explicit_composition(
    k_cache_q16: list[int] | None,
    k_cache_capacity: int,
    v_cache_q16: list[int] | None,
    v_cache_capacity: int,
    layer_idx: int,
    token_count: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    k_token_src_q16: list[int] | None,
    k_token_src_capacity: int,
    v_token_src_q16: list[int] | None,
    v_token_src_capacity: int,
) -> tuple[int, tuple[int, int, int, int]]:
    staged_required_cells = [0]
    staged_required_bytes = [0]
    staged_written_token = [0]
    err = kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_count,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        staged_required_cells,
        staged_required_bytes,
        staged_written_token,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0)

    staged_span = [0]
    staged_k_base = [0]
    staged_v_base = [0]
    err = kv_cache_q16_write_token_checked_nopartial_commit_only(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        staged_written_token[0],
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        staged_span,
        staged_k_base,
        staged_v_base,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0)

    if staged_span[0] != staged_required_cells[0] or staged_k_base[0] != staged_v_base[0]:
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)

    next_token_count = staged_written_token[0] + 1
    if next_token_count > token_capacity:
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)

    return (
        KV_Q16_OK,
        (
            staged_required_cells[0],
            staged_required_bytes[0],
            staged_written_token[0],
            next_token_count,
        ),
    )


def test_source_contains_reserve_and_write_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ReserveAndWriteTokenChecked("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheQ16AppendTokenCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "KVCacheQ16WriteTokenCheckedNoPartialCommitOnly(" in body
    assert "if (staged_span_cells != staged_required_cells)" in body
    assert "*out_next_token_count = staged_next_token_count;" in body


def test_known_vector_success_and_atomic_outputs() -> None:
    layer_count = 3
    token_capacity = 6
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [-1] * total_cells
    v_cache = [-2] * total_cells
    k_src = [100 + i for i in range(span)]
    v_src = [200 + i for i in range(span)]

    out_required_cells = [777]
    out_required_bytes = [778]
    out_written_token = [779]
    out_next_token_count = [780]

    err = kv_cache_q16_reserve_and_write_token_checked(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=1,
        token_count=3,
        layer_count=layer_count,
        token_capacity=token_capacity,
        kv_heads=kv_heads,
        head_dim=head_dim,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        out_required_cells=out_required_cells,
        out_required_bytes=out_required_bytes,
        out_written_token_index=out_written_token,
        out_next_token_count=out_next_token_count,
    )

    assert err == KV_Q16_OK
    assert out_required_cells == [span]
    assert out_required_bytes == [span * 8]
    assert out_written_token == [3]
    assert out_next_token_count == [4]

    base = ((1 * token_capacity) + 3) * span
    assert k_cache[base : base + span] == k_src
    assert v_cache[base : base + span] == v_src


def test_null_alias_and_no_partial_failure() -> None:
    layer_count = 1
    token_capacity = 4
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [11] * total_cells
    v_cache = [22] * total_cells
    k_src = [31 + i for i in range(span)]
    v_src = [41 + i for i in range(span)]

    out_a = [0]
    out_b = [0]
    out_c = [0]
    out_d = [0]

    assert (
        kv_cache_q16_reserve_and_write_token_checked(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            0,
            0,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            None,
            out_b,
            out_c,
            out_d,
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_reserve_and_write_token_checked(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            0,
            0,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            out_a,
            out_a,
            out_c,
            out_d,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    k_before = k_cache.copy()
    v_before = v_cache.copy()
    out_required_cells = [91]
    out_required_bytes = [92]
    out_written_token = [93]
    out_next_token_count = [94]

    err = kv_cache_q16_reserve_and_write_token_checked(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=0,
        token_count=token_capacity,
        layer_count=layer_count,
        token_capacity=token_capacity,
        kv_heads=kv_heads,
        head_dim=head_dim,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        out_required_cells=out_required_cells,
        out_required_bytes=out_required_bytes,
        out_written_token_index=out_written_token,
        out_next_token_count=out_next_token_count,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert k_cache == k_before
    assert v_cache == v_before
    assert out_required_cells == [91]
    assert out_required_bytes == [92]
    assert out_written_token == [93]
    assert out_next_token_count == [94]


def test_randomized_parity_against_explicit_composition() -> None:
    random.seed(1218)

    for _ in range(300):
        layer_count = random.randint(1, 4)
        token_capacity = random.randint(2, 8)
        kv_heads = random.randint(1, 4)
        head_dim = random.randint(1, 8)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        token_count = random.randint(0, token_capacity - 1)
        layer_idx = random.randint(0, layer_count - 1)

        k_cache_a = [random.randint(-200, 200) for _ in range(total_cells)]
        v_cache_a = [random.randint(-200, 200) for _ in range(total_cells)]
        k_cache_b = k_cache_a.copy()
        v_cache_b = v_cache_a.copy()

        k_src = [random.randint(-32768, 32767) for _ in range(span)]
        v_src = [random.randint(-32768, 32767) for _ in range(span)]

        out_required_cells = [I64_MAX]
        out_required_bytes = [I64_MAX]
        out_written_token = [I64_MAX]
        out_next_token_count = [I64_MAX]

        err_impl = kv_cache_q16_reserve_and_write_token_checked(
            k_cache_a,
            len(k_cache_a),
            v_cache_a,
            len(v_cache_a),
            layer_idx,
            token_count,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            out_required_cells,
            out_required_bytes,
            out_written_token,
            out_next_token_count,
        )

        err_exp, tuple_exp = explicit_composition(
            k_cache_b,
            len(k_cache_b),
            v_cache_b,
            len(v_cache_b),
            layer_idx,
            token_count,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
        )

        assert err_impl == err_exp
        if err_impl == KV_Q16_OK:
            assert (
                out_required_cells[0],
                out_required_bytes[0],
                out_written_token[0],
                out_next_token_count[0],
            ) == tuple_exp
            assert k_cache_a == k_cache_b
            assert v_cache_a == v_cache_b


if __name__ == "__main__":
    test_source_contains_reserve_and_write_helper()
    test_known_vector_success_and_atomic_outputs()
    test_null_alias_and_no_partial_failure()
    test_randomized_parity_against_explicit_composition()
    print("ok")
