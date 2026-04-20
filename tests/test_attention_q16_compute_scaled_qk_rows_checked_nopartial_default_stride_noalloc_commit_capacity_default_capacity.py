#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity,
)

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, default_commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, default_commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_commit_stage_cell_capacity,
        default_commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
    )


def explicit_checked_default_capacity_commit_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, default_commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, default_commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_commit_stage_cell_capacity,
        default_commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
    )


def test_source_contains_default_capacity_commit_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionTryMulI64Checked(query_row_count," in body
    assert "AttentionTryMulI64Checked(staged_scores_capacity," in body
    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(" in body


def test_known_vectors_match_explicit_checked_composition() -> None:
    query_row_count = 5
    token_count = 4
    head_dim = 7
    score_scale_q16 = 28672

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    stage_capacity = out_capacity + 3

    q_rows = [((i * 11) - 29) << 10 for i in range(q_capacity)]
    k_rows = [((17 - (i * 3)) << 11) for i in range(k_capacity)]

    out_a = [0x3A3A] * out_capacity
    out_b = out_a.copy()
    stage_a = [0x4B4B] * stage_capacity
    stage_b = stage_a.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_a,
        out_capacity,
        stage_a,
        stage_capacity,
    )
    err_b = explicit_checked_default_capacity_commit_composition(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_b,
        out_capacity,
        stage_b,
        stage_capacity,
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b


def test_staging_buffer_too_small_is_no_partial() -> None:
    query_row_count = 4
    token_count = 3
    head_dim = 6

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0x7F7F] * out_capacity
    out_before = out_scores.copy()

    staged_scores = [0x1234] * (out_capacity - 1)
    staged_before = staged_scores.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        1 << 16,
        out_scores,
        out_capacity,
        staged_scores,
        out_capacity - 1,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_scores == out_before
    assert staged_scores == staged_before


def test_error_paths() -> None:
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
            None,
            1,
            1,
            [0],
            1,
            1,
            1,
            1,
            [0],
            1,
            [0],
            1,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
            [0],
            1,
            1,
            [0],
            1,
            1,
            1,
            1,
            [0],
            1,
            [0],
            -1,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        [0],
        I64_MAX,
        huge,
        [0],
        I64_MAX,
        huge,
        1,
        1,
        [0],
        I64_MAX,
        [0],
        huge,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_compute_overflow_preserves_output() -> None:
    query_row_count = 2
    token_count = 4
    head_dim = 5

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    stage_capacity = out_capacity + 2

    q_rows = [1 << 16] * q_capacity
    k_rows = [1 << 16] * k_capacity
    q_rows[0] = I64_MIN

    out_scores = [0x4343] * out_capacity
    out_before = out_scores.copy()
    staged = [0x6767] * stage_capacity
    staged_before = staged.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        1 << 16,
        out_scores,
        out_capacity,
        staged,
        stage_capacity,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out_scores == out_before
    assert staged == staged_before


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_613)

    for _ in range(5000):
        query_row_count = rng.randint(0, 22)
        token_count = rng.randint(0, 22)
        head_dim = rng.randint(0, 20)
        score_scale_q16 = rng.randint(-(1 << 16), 1 << 16)

        q_cells = query_row_count * head_dim
        k_cells = token_count * head_dim
        out_cells = query_row_count * token_count

        q_cap = q_cells + rng.randint(0, 6)
        k_cap = k_cells + rng.randint(0, 6)
        out_cap = out_cells + rng.randint(0, 8)
        stage_cap = out_cells + rng.randint(0, 8)

        q_rows = [rng.randint(-(1 << 20), 1 << 20) for _ in range(max(q_cap, 1))]
        k_rows = [rng.randint(-(1 << 20), 1 << 20) for _ in range(max(k_cap, 1))]

        out_a = [rng.randint(-(1 << 12), 1 << 12) for _ in range(max(out_cap, 1))]
        out_b = out_a.copy()

        staged_a = [rng.randint(-(1 << 12), 1 << 12) for _ in range(max(stage_cap, 1))]
        staged_b = staged_a.copy()

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
            q_rows,
            q_cap,
            query_row_count,
            k_rows,
            k_cap,
            token_count,
            head_dim,
            score_scale_q16,
            out_a,
            out_cap,
            staged_a,
            stage_cap,
        )
        err_b = explicit_checked_default_capacity_commit_composition(
            q_rows,
            q_cap,
            query_row_count,
            k_rows,
            k_cap,
            token_count,
            head_dim,
            score_scale_q16,
            out_b,
            out_cap,
            staged_b,
            stage_cap,
        )

        assert err_a == err_b
        if err_a == ATTN_Q16_OK:
            assert out_a == out_b
        else:
            assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_default_capacity_commit_wrapper()
    test_known_vectors_match_explicit_checked_composition()
    test_staging_buffer_too_small_is_no_partial()
    test_error_paths()
    test_compute_overflow_preserves_output()
    test_randomized_parity_against_explicit_composition()
    print("ok")
