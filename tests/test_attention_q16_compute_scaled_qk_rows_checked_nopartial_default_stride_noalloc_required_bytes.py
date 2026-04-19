#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    stage_cell_capacity: int,
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

    required_stage_cells = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
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
        required_stage_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells[0], 8)
    if err != ATTN_Q16_OK:
        return err

    out_required_stage_cells[0] = required_stage_cells[0]
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = required_out_cells[0]
    return ATTN_Q16_OK


def explicit_checked_required_bytes_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    stage_cell_capacity: int,
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

    required_stage_cells = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
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
        required_stage_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells[0], 8)
    if err != ATTN_Q16_OK:
        return err

    out_required_stage_cells[0] = required_stage_cells[0]
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = required_out_cells[0]
    return ATTN_Q16_OK


def test_source_contains_noalloc_required_bytes_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly("
        in body
    )
    assert "status = AttentionTryMulI64Checked(required_stage_cells," in body
    assert "sizeof(I64)" in body


def test_known_vectors_and_zero_case() -> None:
    query_row_count = 5
    token_count = 7
    head_dim = 9

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    out_stage_cells = [123]
    out_stage_bytes = [456]
    out_out_cells = [789]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        query_row_count * token_count,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert out_stage_cells == [query_row_count * token_count]
    assert out_stage_bytes == [query_row_count * token_count * 8]
    assert out_out_cells == [query_row_count * token_count]

    out_stage_cells = [1]
    out_stage_bytes = [1]
    out_out_cells = [1]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        [0],
        1,
        0,
        [0],
        1,
        7,
        9,
        [0],
        1,
        0,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert out_stage_cells == [0]
    assert out_stage_bytes == [0]
    assert out_out_cells == [0]


def test_error_paths() -> None:
    out_stage_cells = [11]
    out_stage_bytes = [22]
    out_out_cells = [33]

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
            [0],
            1,
            1,
            [0],
            1,
            1,
            1,
            [0],
            1,
            1,
            None,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        [0],
        -1,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        1,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        [0],
        1,
        2,
        [0],
        1,
        3,
        1,
        [0],
        1,
        5,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        [0],
        I64_MAX,
        huge,
        [0],
        I64_MAX,
        huge,
        1,
        [0],
        I64_MAX,
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_601)

    for _ in range(5000):
        query_row_count = rng.randint(0, 35)
        token_count = rng.randint(0, 35)
        head_dim = rng.randint(0, 35)

        q_capacity = max(0, query_row_count * head_dim + rng.randint(-40, 40))
        k_capacity = max(0, token_count * head_dim + rng.randint(-40, 40))
        out_capacity = max(0, query_row_count * token_count + rng.randint(-40, 40))
        stage_cell_capacity = max(
            0, query_row_count * token_count + rng.randint(-40, 40)
        )

        if rng.random() < 0.06:
            query_row_count = -rng.randint(1, 60)
        if rng.random() < 0.06:
            token_count = -rng.randint(1, 60)
        if rng.random() < 0.06:
            head_dim = -rng.randint(1, 60)
        if rng.random() < 0.06:
            q_capacity = -rng.randint(1, 60)
        if rng.random() < 0.06:
            k_capacity = -rng.randint(1, 60)
        if rng.random() < 0.06:
            out_capacity = -rng.randint(1, 60)
        if rng.random() < 0.06:
            stage_cell_capacity = -rng.randint(1, 60)

        if rng.random() < 0.04:
            query_row_count = (1 << 62) + rng.randint(0, 32)
            token_count = (1 << 62) + rng.randint(0, 32)

        q_rows = None if rng.random() < 0.04 else [0] * max(q_capacity, 1)
        k_rows = None if rng.random() < 0.04 else [0] * max(k_capacity, 1)
        out_scores = None if rng.random() < 0.04 else [0] * max(out_capacity, 1)

        got_stage_cells = [0x1111]
        got_stage_bytes = [0x2222]
        got_out_cells = [0x3333]
        exp_stage_cells = [0x1111]
        exp_stage_bytes = [0x2222]
        exp_out_cells = [0x3333]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            stage_cell_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_exp = explicit_checked_required_bytes_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            stage_cell_capacity,
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
            assert got_stage_cells == [0x1111]
            assert got_stage_bytes == [0x2222]
            assert got_out_cells == [0x3333]
            assert exp_stage_cells == [0x1111]
            assert exp_stage_bytes == [0x2222]
            assert exp_out_cells == [0x3333]

