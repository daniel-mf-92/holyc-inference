#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked import (
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
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
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        q_rows_q16 is None
        or k_rows_q16 is None
        or out_scores_q32 is None
        or out_required_stage_cells is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if stage_cell_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    required_q_cells = [0]
    required_k_cells = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        required_q_cells,
        required_k_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = required_stage_cells
    out_required_out_cells[0] = required_out_cells[0]
    return ATTN_Q16_OK


def explicit_checked_noalloc_preflight_composition(
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
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        q_rows_q16 is None
        or k_rows_q16 is None
        or out_scores_q32 is None
        or out_required_stage_cells is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if stage_cell_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    required_q_cells = [0]
    required_k_cells = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        required_q_cells,
        required_k_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err
    if required_stage_cells > stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = required_stage_cells
    out_required_out_cells[0] = required_out_cells[0]
    return ATTN_Q16_OK


def test_source_contains_noalloc_default_stride_preflight_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialPreflightOnly(" in body
    )
    assert (
        "status = AttentionTryMulI64Checked(query_row_count," in body
        and "token_count," in body
    )
    assert "if (required_stage_cells > stage_cell_capacity)" in body


def test_known_vectors_and_stage_capacity_gate() -> None:
    query_row_count = 6
    token_count = 4
    head_dim = 9

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    got_stage = [999]
    got_out = [888]
    exp_stage = [777]
    exp_out = [666]

    err_got = (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
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
            got_stage,
            got_out,
        )
    )
    err_exp = explicit_checked_noalloc_preflight_composition(
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
        exp_stage,
        exp_out,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_stage == exp_stage == [query_row_count * token_count]
    assert got_out == exp_out == [query_row_count * token_count]

    got_stage = [111]
    got_out = [222]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        query_row_count * token_count - 1,
        got_stage,
        got_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert got_stage == [111]
    assert got_out == [222]


def test_error_paths_preserve_outputs() -> None:
    out_stage = [31]
    out_out = [41]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        None,
        0,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        1,
        out_stage,
        out_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out_stage == [31]
    assert out_out == [41]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
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
        out_stage,
        out_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_stage == [31]
    assert out_out == [41]


def test_randomized_parity_against_explicit_checked_composition() -> None:
    rng = random.Random(20260420_596)

    for _ in range(5000):
        query_row_count = rng.randint(0, 96)
        token_count = rng.randint(0, 96)
        head_dim = rng.randint(0, 96)

        q_need = query_row_count * head_dim
        k_need = token_count * head_dim
        out_need = query_row_count * token_count

        q_rows_capacity = max(0, q_need + rng.randint(-8, 8))
        k_rows_capacity = max(0, k_need + rng.randint(-8, 8))
        out_scores_capacity = max(0, out_need + rng.randint(-8, 8))
        stage_cell_capacity = max(0, out_need + rng.randint(-8, 8))

        if rng.random() < 0.05:
            q_rows_capacity = -rng.randint(1, 9)
        if rng.random() < 0.05:
            k_rows_capacity = -rng.randint(1, 9)
        if rng.random() < 0.05:
            out_scores_capacity = -rng.randint(1, 9)
        if rng.random() < 0.05:
            stage_cell_capacity = -rng.randint(1, 9)

        if rng.random() < 0.05:
            query_row_count = -rng.randint(1, 9)
        if rng.random() < 0.05:
            token_count = -rng.randint(1, 9)
        if rng.random() < 0.05:
            head_dim = -rng.randint(1, 9)

        q_rows = None if rng.random() < 0.03 else [0] * max(q_rows_capacity, 1)
        k_rows = None if rng.random() < 0.03 else [0] * max(k_rows_capacity, 1)
        out_scores = None if rng.random() < 0.03 else [0] * max(out_scores_capacity, 1)

        got_stage = [101]
        got_out = [202]
        exp_stage = [303]
        exp_out = [404]

        err_got = (
            attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
                q_rows,
                q_rows_capacity,
                query_row_count,
                k_rows,
                k_rows_capacity,
                token_count,
                head_dim,
                out_scores,
                out_scores_capacity,
                stage_cell_capacity,
                got_stage,
                got_out,
            )
        )
        err_exp = explicit_checked_noalloc_preflight_composition(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            stage_cell_capacity,
            exp_stage,
            exp_out,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_stage == exp_stage
            assert got_out == exp_out
        else:
            assert got_stage == [101]
            assert got_out == [202]


def test_overflow_surface_matches_checked_math() -> None:
    out_stage = [11]
    out_out = [22]

    huge = I64_MAX
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        [0],
        huge,
        huge,
        [0],
        huge,
        huge,
        1,
        [0],
        huge,
        huge,
        out_stage,
        out_out,
    )
    assert err in (ATTN_Q16_ERR_OVERFLOW, ATTN_Q16_ERR_BAD_PARAM)


if __name__ == "__main__":
    test_source_contains_noalloc_default_stride_preflight_only_helper()
    test_known_vectors_and_stage_capacity_gate()
    test_error_paths_preserve_outputs()
    test_randomized_parity_against_explicit_checked_composition()
    test_overflow_surface_matches_checked_math()
    print("attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only=ok")
