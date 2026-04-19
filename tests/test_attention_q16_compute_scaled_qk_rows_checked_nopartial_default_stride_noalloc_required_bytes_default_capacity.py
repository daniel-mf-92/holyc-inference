#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
)
from test_attention_q16_compute_scaled_qk_rows_checked import try_mul_i64_checked
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def explicit_checked_default_capacity_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def test_source_contains_noalloc_required_bytes_default_capacity_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "status = AttentionTryMulI64Checked(query_row_count," in body
    assert "token_count," in body
    assert (
        "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes("
        in body
    )


def test_known_vectors_and_zero_case() -> None:
    query_row_count = 4
    token_count = 6
    head_dim = 8

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    out_stage_cells = [111]
    out_stage_bytes = [222]
    out_out_cells = [333]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert out_stage_cells == [query_row_count * token_count]
    assert out_stage_bytes == [query_row_count * token_count * 8]
    assert out_out_cells == [query_row_count * token_count]

    out_stage_cells = [9]
    out_stage_bytes = [9]
    out_out_cells = [9]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
        [0],
        1,
        0,
        [0],
        1,
        10,
        4,
        [0],
        1,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert out_stage_cells == [0]
    assert out_stage_bytes == [0]
    assert out_out_cells == [0]


def test_error_paths() -> None:
    out_stage_cells = [1]
    out_stage_bytes = [1]
    out_out_cells = [1]

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
            [0], 1, 1, [0], 1, 1, 1, [0], 1, None, out_stage_bytes, out_out_cells
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
            None,
            1,
            1,
            [0],
            1,
            1,
            1,
            [0],
            1,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
            [0],
            -1,
            1,
            [0],
            1,
            1,
            1,
            [0],
            1,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
        [0],
        I64_MAX,
        huge,
        [0],
        I64_MAX,
        huge,
        1,
        [0],
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_605)

    for _ in range(5000):
        query_row_count = rng.randint(0, 40)
        token_count = rng.randint(0, 40)
        head_dim = rng.randint(0, 40)

        q_capacity = max(0, query_row_count * head_dim + rng.randint(-50, 50))
        k_capacity = max(0, token_count * head_dim + rng.randint(-50, 50))
        out_capacity = max(0, query_row_count * token_count + rng.randint(-50, 50))

        if rng.random() < 0.06:
            query_row_count = -rng.randint(1, 80)
        if rng.random() < 0.06:
            token_count = -rng.randint(1, 80)
        if rng.random() < 0.06:
            head_dim = -rng.randint(1, 80)
        if rng.random() < 0.06:
            q_capacity = -rng.randint(1, 80)
        if rng.random() < 0.06:
            k_capacity = -rng.randint(1, 80)
        if rng.random() < 0.06:
            out_capacity = -rng.randint(1, 80)

        if rng.random() < 0.04:
            query_row_count = (1 << 62) + rng.randint(0, 32)
            token_count = (1 << 62) + rng.randint(0, 32)

        q_rows = None if rng.random() < 0.03 else [0] * max(q_capacity, 1)
        k_rows = None if rng.random() < 0.03 else [0] * max(k_capacity, 1)
        out_scores = None if rng.random() < 0.03 else [0] * max(out_capacity, 1)

        got_stage_cells = [0xAAAA]
        got_stage_bytes = [0xBBBB]
        got_out_cells = [0xCCCC]
        exp_stage_cells = [0xAAAA]
        exp_stage_bytes = [0xBBBB]
        exp_out_cells = [0xCCCC]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_exp = explicit_checked_default_capacity_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
        else:
            assert got_stage_cells == [0xAAAA]
            assert got_stage_bytes == [0xBBBB]
            assert got_out_cells == [0xCCCC]
            assert exp_stage_cells == [0xAAAA]
            assert exp_stage_bytes == [0xBBBB]
            assert exp_out_cells == [0xCCCC]

