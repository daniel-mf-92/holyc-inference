#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytes."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    required_q_cells = [0]
    required_k_cells = [0]
    required_out_cells = [0]
    required_stage_cells = [0]
    required_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        required_q_cells,
        required_k_cells,
        required_out_cells,
        required_stage_cells,
        required_stage_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    out_required_q_cells[0] = required_q_cells[0]
    out_required_k_cells[0] = required_k_cells[0]
    out_required_out_cells[0] = required_out_cells[0]
    out_required_stage_cells[0] = required_stage_cells[0]
    out_required_stage_bytes[0] = required_stage_bytes[0]
    return ATTN_Q16_OK


def explicit_checked_required_bytes_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    required_q_cells = [0]
    required_k_cells = [0]
    required_out_cells = [0]
    required_stage_cells = [0]
    required_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        required_q_cells,
        required_k_cells,
        required_out_cells,
        required_stage_cells,
        required_stage_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    out_required_q_cells[0] = required_q_cells[0]
    out_required_k_cells[0] = required_k_cells[0]
    out_required_out_cells[0] = required_out_cells[0]
    out_required_stage_cells[0] = required_stage_cells[0]
    out_required_stage_bytes[0] = required_stage_bytes[0]
    return ATTN_Q16_OK


def test_source_contains_strided_noalloc_required_bytes_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytes("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly(" in body
    assert "*out_required_stage_bytes = required_stage_bytes;" in body


def test_known_vectors_and_null_outputs() -> None:
    query_row_count = 3
    token_count = 5
    head_dim = 6
    query_row_stride_q16 = 9
    k_row_stride_q16 = 8
    out_row_stride = 7

    q_capacity = query_row_count * query_row_stride_q16
    k_capacity = token_count * k_row_stride_q16
    out_capacity = query_row_count * out_row_stride

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    required_q_cells = [11]
    required_k_cells = [22]
    required_out_cells = [33]
    required_stage_cells = [44]
    required_stage_bytes = [55]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes(
        q_rows,
        q_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows,
        k_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores,
        out_capacity,
        out_row_stride,
        required_q_cells,
        required_k_cells,
        required_out_cells,
        required_stage_cells,
        required_stage_bytes,
    )
    assert err == ATTN_Q16_OK
    assert required_q_cells == [(query_row_count - 1) * query_row_stride_q16 + head_dim]
    assert required_k_cells == [(token_count - 1) * k_row_stride_q16 + head_dim]
    assert required_out_cells == [(query_row_count - 1) * out_row_stride + token_count]
    assert required_stage_cells == [query_row_count * token_count]
    assert required_stage_bytes == [query_row_count * token_count * 8]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes(
        q_rows,
        q_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows,
        k_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores,
        out_capacity,
        out_row_stride,
        None,
        required_k_cells,
        required_out_cells,
        required_stage_cells,
        required_stage_bytes,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_638)

    for _ in range(5000):
        query_row_count = rng.randint(0, 40)
        token_count = rng.randint(0, 40)
        head_dim = rng.randint(0, 40)
        query_row_stride_q16 = rng.randint(0, 45)
        k_row_stride_q16 = rng.randint(0, 45)
        out_row_stride = rng.randint(0, 45)

        q_capacity = max(0, query_row_count * max(query_row_stride_q16, 1) + rng.randint(-60, 60))
        k_capacity = max(0, token_count * max(k_row_stride_q16, 1) + rng.randint(-60, 60))
        out_capacity = max(0, query_row_count * max(out_row_stride, 1) + rng.randint(-60, 60))

        if rng.random() < 0.06:
            query_row_count = -rng.randint(1, 80)
        if rng.random() < 0.06:
            token_count = -rng.randint(1, 80)
        if rng.random() < 0.06:
            head_dim = -rng.randint(1, 80)
        if rng.random() < 0.06:
            query_row_stride_q16 = -rng.randint(1, 80)
        if rng.random() < 0.06:
            k_row_stride_q16 = -rng.randint(1, 80)
        if rng.random() < 0.06:
            out_row_stride = -rng.randint(1, 80)
        if rng.random() < 0.06:
            q_capacity = -rng.randint(1, 80)
        if rng.random() < 0.06:
            k_capacity = -rng.randint(1, 80)
        if rng.random() < 0.06:
            out_capacity = -rng.randint(1, 80)

        q_rows = None if rng.random() < 0.03 else [0] * max(q_capacity, 1)
        k_rows = None if rng.random() < 0.03 else [0] * max(k_capacity, 1)
        out_scores = None if rng.random() < 0.03 else [0] * max(out_capacity, 1)

        got_q_cells = [0xAAAA]
        got_k_cells = [0xBBBB]
        got_out_cells = [0xCCCC]
        got_stage_cells = [0xDDDD]
        got_stage_bytes = [0xEEEE]

        exp_q_cells = [0xAAAA]
        exp_k_cells = [0xBBBB]
        exp_out_cells = [0xCCCC]
        exp_stage_cells = [0xDDDD]
        exp_stage_bytes = [0xEEEE]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_capacity,
            out_row_stride,
            got_q_cells,
            got_k_cells,
            got_out_cells,
            got_stage_cells,
            got_stage_bytes,
        )
        err_exp = explicit_checked_required_bytes_composition(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_capacity,
            out_row_stride,
            exp_q_cells,
            exp_k_cells,
            exp_out_cells,
            exp_stage_cells,
            exp_stage_bytes,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_q_cells == exp_q_cells
            assert got_k_cells == exp_k_cells
            assert got_out_cells == exp_out_cells
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
        else:
            assert got_q_cells == [0xAAAA]
            assert got_k_cells == [0xBBBB]
            assert got_out_cells == [0xCCCC]
            assert got_stage_cells == [0xDDDD]
            assert got_stage_bytes == [0xEEEE]
            assert exp_q_cells == [0xAAAA]
            assert exp_k_cells == [0xBBBB]
            assert exp_out_cells == [0xCCCC]
            assert exp_stage_cells == [0xDDDD]
            assert exp_stage_bytes == [0xEEEE]
