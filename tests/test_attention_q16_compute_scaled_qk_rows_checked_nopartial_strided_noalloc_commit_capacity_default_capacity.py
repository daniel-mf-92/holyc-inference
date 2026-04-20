#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
)
from test_attention_q16_compute_scaled_qk_rows_checked import try_mul_i64_checked
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
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

    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
    )


def explicit_checked_default_capacity_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
    )


def test_source_contains_strided_noalloc_commit_capacity_default_capacity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacity("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionTryMulI64Checked(query_row_count," in body
    assert "AttentionTryMulI64Checked(staged_scores_capacity," in body
    assert "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacity(" in body


def test_known_vector_matches_explicit_checked_composition() -> None:
    query_row_count = 2
    token_count = 5
    head_dim = 8
    query_row_stride_q16 = 9
    k_row_stride_q16 = 10
    out_row_stride = 8
    score_scale_q16 = 65536

    q_rows = [
        100,
        -50,
        25,
        -12,
        6,
        -3,
        2,
        -1,
        0,
        0,
        75,
        -30,
        15,
        -7,
        3,
        -1,
        1,
        0,
    ]
    k_rows = ([11, -7, 5, -3, 2, -1, 1, 0, 0, 0] * token_count)[: token_count * k_row_stride_q16]

    out_capacity = (query_row_count - 1) * out_row_stride + token_count
    stage_capacity = query_row_count * token_count

    out_a = [333] * out_capacity
    out_b = out_a.copy()
    stage_a = [0] * stage_capacity
    stage_b = [0] * stage_capacity

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
        q_rows,
        len(q_rows),
        query_row_count,
        query_row_stride_q16,
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_a,
        out_capacity,
        out_row_stride,
        stage_a,
        len(stage_a),
    )

    err_b = explicit_checked_default_capacity_composition(
        q_rows,
        len(q_rows),
        query_row_count,
        query_row_stride_q16,
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_b,
        out_capacity,
        out_row_stride,
        stage_b,
        len(stage_b),
    )

    assert err_a == ATTN_Q16_OK
    assert err_a == err_b
    assert out_a == out_b


def test_error_surfaces_and_no_partial_guarantee() -> None:
    q_rows = [1, 2, 3, 4]
    k_rows = [1, 2, 3, 4]
    out = [9, 9, 9, 9]
    out_before = out.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
        q_rows,
        4,
        1,
        4,
        k_rows,
        4,
        1,
        4,
        4,
        65536,
        out,
        4,
        1,
        [0],
        0,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_before

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
        None,
        4,
        1,
        4,
        k_rows,
        4,
        1,
        4,
        4,
        65536,
        out,
        4,
        1,
        [0],
        1,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out == out_before


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(20260420_634)

    for _ in range(3000):
        query_row_count = rng.randint(0, 10)
        token_count = rng.randint(0, 12)
        head_dim = rng.randint(0, 16)

        query_row_stride_q16 = rng.randint(0, 20)
        if query_row_count > 0 and head_dim > 0 and query_row_stride_q16 < head_dim:
            query_row_stride_q16 = head_dim + rng.randint(0, 4)

        k_row_stride_q16 = rng.randint(0, 20)
        if token_count > 0 and head_dim > 0 and k_row_stride_q16 < head_dim:
            k_row_stride_q16 = head_dim + rng.randint(0, 4)

        out_row_stride = rng.randint(0, 20)
        if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
            out_row_stride = token_count + rng.randint(0, 4)

        q_need = 0
        if query_row_count > 0:
            q_need = (query_row_count - 1) * query_row_stride_q16 + head_dim
        k_need = token_count * k_row_stride_q16
        out_need = 0
        if query_row_count > 0 and token_count > 0:
            out_need = (query_row_count - 1) * out_row_stride + token_count

        q_capacity = max(0, q_need + rng.randint(-2, 3))
        k_capacity = max(0, k_need + rng.randint(-2, 3))
        out_capacity = max(0, out_need + rng.randint(-2, 3))

        stage_need = query_row_count * token_count
        stage_capacity = max(0, stage_need + rng.randint(-2, 3))

        q_rows = [rng.randint(-1200, 1200) for _ in range(q_capacity)]
        k_rows = [rng.randint(-1200, 1200) for _ in range(k_capacity)]
        out_a = [rng.randint(-1200, 1200) for _ in range(out_capacity)]
        out_b = out_a.copy()
        stage_a = [rng.randint(-1200, 1200) for _ in range(stage_capacity)]
        stage_b = stage_a.copy()
        score_scale_q16 = rng.randint(-131072, 131072)

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_a,
            out_capacity,
            out_row_stride,
            stage_a,
            stage_capacity,
        )

        err_b = explicit_checked_default_capacity_composition(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_b,
            out_capacity,
            out_row_stride,
            stage_b,
            stage_capacity,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_strided_noalloc_commit_capacity_default_capacity_wrapper()
    test_known_vector_matches_explicit_checked_composition()
    test_error_surfaces_and_no_partial_guarantee()
    test_randomized_parity_vs_explicit_checked_composition()
    print("ok")
