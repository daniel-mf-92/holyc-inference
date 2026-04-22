#!/usr/bin/env python3
"""Parity harness for KVCacheQ16AppendTokenCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1046)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_OK,
    try_mul_i64_checked,
)
from test_kv_cache_q16_write_token_checked_nopartial_commit_only import (
    kv_cache_q16_write_token_checked_nopartial_preflight,
)
from test_kv_cache_q16_write_token_checked_nopartial_commit_only_preflight_only import (
    kv_cache_q16_write_token_checked_nopartial_commit_only_preflight_only,
)


def kv_cache_q16_append_token_checked_nopartial_preflight(
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
    out_next_token_index: list[int] | None,
) -> int:
    if out_required_cells is None or out_next_token_index is None:
        return KV_Q16_ERR_NULL_PTR

    if out_required_cells is out_next_token_index:
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_required_cells is k_cache_q16
        or out_required_cells is v_cache_q16
        or out_required_cells is k_token_src_q16
        or out_required_cells is v_token_src_q16
        or out_next_token_index is k_cache_q16
        or out_next_token_index is v_cache_q16
        or out_next_token_index is k_token_src_q16
        or out_next_token_index is v_token_src_q16
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
    )

    staged_k_base = [0]
    staged_v_base = [0]
    staged_required_cells = [0]
    err = kv_cache_q16_write_token_checked_nopartial_preflight(
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
        staged_k_base,
        staged_v_base,
        staged_required_cells,
    )
    if err != KV_Q16_OK:
        return err

    if staged_k_base[0] != staged_v_base[0]:
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
    ):
        return KV_Q16_ERR_BAD_PARAM

    out_required_cells[0] = staged_required_cells[0]
    out_next_token_index[0] = token_count
    return KV_Q16_OK


def kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only(
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
    out_next_token_index: list[int] | None,
) -> int:
    if out_required_cells is None or out_required_bytes is None or out_next_token_index is None:
        return KV_Q16_ERR_NULL_PTR

    if (
        out_required_cells is out_required_bytes
        or out_required_cells is out_next_token_index
        or out_required_bytes is out_next_token_index
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
        or out_next_token_index is k_cache_q16
        or out_next_token_index is v_cache_q16
        or out_next_token_index is k_token_src_q16
        or out_next_token_index is v_token_src_q16
    ):
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
    )

    staged_required_cells = [0]
    staged_k_base = [0]
    staged_v_base = [0]
    err = kv_cache_q16_write_token_checked_nopartial_commit_only_preflight_only(
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
        staged_k_base,
        staged_v_base,
    )
    if err != KV_Q16_OK:
        return err

    if staged_k_base[0] != staged_v_base[0]:
        return KV_Q16_ERR_BAD_PARAM

    err, staged_required_bytes = try_mul_i64_checked(staged_required_cells[0], 8)
    if err != KV_Q16_OK:
        return err

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
    ):
        return KV_Q16_ERR_BAD_PARAM

    out_required_cells[0] = staged_required_cells[0]
    out_required_bytes[0] = staged_required_bytes
    out_next_token_index[0] = token_count
    return KV_Q16_OK


def kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only_parity(
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
    out_next_token_index: list[int] | None,
) -> int:
    if out_required_cells is None or out_required_bytes is None or out_next_token_index is None:
        return KV_Q16_ERR_NULL_PTR

    if (
        out_required_cells is out_required_bytes
        or out_required_cells is out_next_token_index
        or out_required_bytes is out_next_token_index
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
        or out_next_token_index is k_cache_q16
        or out_next_token_index is v_cache_q16
        or out_next_token_index is k_token_src_q16
        or out_next_token_index is v_token_src_q16
    ):
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
    )

    staged_cells_commit = [0]
    staged_bytes_commit = [0]
    staged_token_commit = [0]
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
        staged_cells_commit,
        staged_bytes_commit,
        staged_token_commit,
    )
    if err != KV_Q16_OK:
        return err

    staged_cells_preflight = [0]
    staged_token_preflight = [0]
    err = kv_cache_q16_append_token_checked_nopartial_preflight(
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
        staged_cells_preflight,
        staged_token_preflight,
    )
    if err != KV_Q16_OK:
        return err

    err, staged_bytes_preflight = try_mul_i64_checked(staged_cells_preflight[0], 8)
    if err != KV_Q16_OK:
        return err

    if (
        staged_cells_commit[0] != staged_cells_preflight[0]
        or staged_bytes_commit[0] != staged_bytes_preflight
        or staged_token_commit[0] != staged_token_preflight[0]
    ):
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
    ):
        return KV_Q16_ERR_BAD_PARAM

    out_required_cells[0] = staged_cells_commit[0]
    out_required_bytes[0] = staged_bytes_commit[0]
    out_next_token_index[0] = staged_token_commit[0]
    return KV_Q16_OK


def explicit_parity_composition(*args, **kwargs) -> tuple[int, tuple[int, int, int]]:
    preflight_cells = [0]
    preflight_next_token = [0]
    err = kv_cache_q16_append_token_checked_nopartial_preflight(
        *args,
        out_required_cells=preflight_cells,
        out_next_token_index=preflight_next_token,
        **kwargs,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0)

    err, preflight_bytes = try_mul_i64_checked(preflight_cells[0], 8)
    if err != KV_Q16_OK:
        return err, (0, 0, 0)

    commit_cells = [0]
    commit_bytes = [0]
    commit_next_token = [0]
    err = kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only(
        *args,
        out_required_cells=commit_cells,
        out_required_bytes=commit_bytes,
        out_next_token_index=commit_next_token,
        **kwargs,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0)

    if (
        preflight_cells[0] != commit_cells[0]
        or preflight_bytes != commit_bytes[0]
        or preflight_next_token[0] != commit_next_token[0]
    ):
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0)

    return KV_Q16_OK, (preflight_cells[0], preflight_bytes, preflight_next_token[0])


def test_source_contains_iq1046_function() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16AppendTokenCheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "KVCacheQ16AppendTokenCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "KVCacheQ16AppendTokenCheckedNoPartialPreflight(" in body
    assert "KVTryMulI64Checked(staged_required_cells_preflight" in body


def test_known_vector_parity_and_zero_write_contract() -> None:
    layer_count = 3
    token_capacity = 6
    kv_heads = 2
    head_dim = 5
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [1000 + idx for idx in range(total_cells)]
    v_cache = [2000 + idx for idx in range(total_cells)]
    k_before = k_cache.copy()
    v_before = v_cache.copy()
    k_src = [300 + idx for idx in range(span)]
    v_src = [700 + idx for idx in range(span)]

    out_cells = [0]
    out_bytes = [0]
    out_next_token = [0]
    err = kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only_parity(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=1,
        token_count=4,
        layer_count=layer_count,
        token_capacity=token_capacity,
        kv_heads=kv_heads,
        head_dim=head_dim,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        out_required_cells=out_cells,
        out_required_bytes=out_bytes,
        out_next_token_index=out_next_token,
    )
    assert err == KV_Q16_OK
    assert out_cells == [span]
    assert out_bytes == [span * 8]
    assert out_next_token == [4]
    assert k_cache == k_before
    assert v_cache == v_before


def test_alias_and_null_output_guards() -> None:
    k_cache = [11] * 128
    v_cache = [22] * 128
    k_src = [33] * 8
    v_src = [44] * 8

    assert (
        kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only_parity(
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

    shared_out = [0]
    assert (
        kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only_parity(
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
            shared_out,
            shared_out,
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )


def test_parity_matches_explicit_composition_adversarial() -> None:
    rng = random.Random(1046)

    for _ in range(500):
        layer_count = rng.randint(1, 8)
        token_capacity = rng.randint(1, 16)
        kv_heads = rng.randint(1, 8)
        head_dim = rng.randint(1, 32)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        layer_idx = rng.randint(0, layer_count - 1)
        token_count = rng.randint(0, token_capacity - 1)

        k_cache = [rng.randint(-2000, 2000) for _ in range(total_cells)]
        v_cache = [rng.randint(-2000, 2000) for _ in range(total_cells)]
        k_src = [rng.randint(-3000, 3000) for _ in range(span)]
        v_src = [rng.randint(-3000, 3000) for _ in range(span)]

        if rng.random() < 0.2:
            k_cap = total_cells - rng.randint(0, min(span, total_cells))
            v_cap = total_cells - rng.randint(0, min(span, total_cells))
        else:
            k_cap = total_cells
            v_cap = total_cells

        k_src_cap = len(k_src)
        v_src_cap = len(v_src)
        if rng.random() < 0.2:
            k_src_cap = max(0, k_src_cap - rng.randint(1, min(4, k_src_cap)))
        if rng.random() < 0.2:
            v_src_cap = max(0, v_src_cap - rng.randint(1, min(4, v_src_cap)))

        got_cells = [0]
        got_bytes = [0]
        got_next_token = [0]
        got_err = kv_cache_q16_append_token_checked_nopartial_commit_only_preflight_only_parity(
            k_cache,
            k_cap,
            v_cache,
            v_cap,
            layer_idx,
            token_count,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            k_src_cap,
            v_src,
            v_src_cap,
            got_cells,
            got_bytes,
            got_next_token,
        )

        exp_err, exp_tuple = explicit_parity_composition(
            k_cache,
            k_cap,
            v_cache,
            v_cap,
            layer_idx,
            token_count,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            k_src_cap,
            v_src,
            v_src_cap,
        )

        assert got_err == exp_err
        if got_err == KV_Q16_OK:
            assert (got_cells[0], got_bytes[0], got_next_token[0]) == exp_tuple


def test_overflow_path_surfaces_from_required_bytes_mul() -> None:
    huge_span = I64_MAX // 8 + 1
    err, _ = try_mul_i64_checked(huge_span, 8)
    assert err != KV_Q16_OK


if __name__ == "__main__":
    test_source_contains_iq1046_function()
    test_known_vector_parity_and_zero_write_contract()
    test_alias_and_null_output_guards()
    test_parity_matches_explicit_composition_adversarial()
    test_overflow_path_surfaces_from_required_bytes_mul()

