#!/usr/bin/env python3
"""Parity harness for ...CommitCapacityAliasSafeDefaultCapacity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity(
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
    **addr,
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

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
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
        **addr,
    )


def explicit_checked_default_capacity_alias_safe_composition(
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
    **addr,
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity(
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
        staged_scores_q32,
        staged_scores_capacity,
        **addr,
    )


def test_source_contains_alias_safe_default_capacity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafeDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionTryMulI64Checked(query_row_count," in body
    assert "AttentionTryMulI64Checked(staged_scores_capacity," in body
    assert (
        "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafe("
        in body
    )


def test_known_vectors_and_alias_overlap_path() -> None:
    q_rows = [1, 2, 3, 4, 5, 6]
    k_rows = [7, 8, 9, 10, 11, 12]
    out = [0] * 6
    stage = [0] * 6

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity(
        q_rows,
        6,
        2,
        3,
        k_rows,
        6,
        2,
        3,
        3,
        65536,
        out,
        6,
        3,
        stage,
        6,
    )
    assert err == ATTN_Q16_OK

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity(
        q_rows,
        6,
        2,
        3,
        k_rows,
        6,
        2,
        3,
        3,
        65536,
        out,
        6,
        3,
        stage,
        6,
        q_base_addr=0x100000,
        k_base_addr=0x200000,
        out_base_addr=0x300000,
        stage_base_addr=0x100008,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM


def test_randomized_parity_and_overflow() -> None:
    rng = random.Random(20260420_651)

    for _ in range(400):
        query_row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 8)
        head_dim = rng.randint(0, 10)

        query_row_stride_q16 = max(head_dim, head_dim + rng.randint(0, 3))
        k_row_stride_q16 = max(head_dim, head_dim + rng.randint(0, 3))
        out_row_stride = max(token_count, token_count + rng.randint(0, 3))

        q_need = 0 if query_row_count == 0 else (query_row_count - 1) * query_row_stride_q16 + head_dim
        k_need = 0 if token_count == 0 else (token_count - 1) * k_row_stride_q16 + head_dim
        out_need = 0 if (query_row_count == 0 or token_count == 0) else (query_row_count - 1) * out_row_stride + token_count
        stage_need = query_row_count * token_count

        q_capacity = max(0, q_need + rng.randint(-1, 2))
        k_capacity = max(0, k_need + rng.randint(-1, 2))
        out_capacity = max(0, out_need + rng.randint(-1, 2))
        stage_capacity = max(0, stage_need + rng.randint(-1, 2))

        q_rows = [rng.randint(-500, 500) for _ in range(max(1, q_capacity))]
        k_rows = [rng.randint(-500, 500) for _ in range(max(1, k_capacity))]
        out_a = [rng.randint(-100, 100) for _ in range(max(1, out_capacity))]
        out_b = out_a.copy()
        stage_a = [0 for _ in range(max(1, stage_capacity))]
        stage_b = [0 for _ in range(max(1, stage_capacity))]

        q_base_addr = 0x100000 + rng.randint(0, 128) * 8
        k_base_addr = 0x200000 + rng.randint(0, 128) * 8
        out_base_addr = 0x300000 + rng.randint(0, 128) * 8
        stage_base_addr = 0x400000 + rng.randint(0, 128) * 8
        if rng.random() < 0.2:
            stage_base_addr = out_base_addr + rng.randint(0, 2) * 8

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            65536,
            out_a,
            out_capacity,
            out_row_stride,
            stage_a,
            stage_capacity,
            q_base_addr=q_base_addr,
            k_base_addr=k_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )
        err_b = explicit_checked_default_capacity_alias_safe_composition(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            65536,
            out_b,
            out_capacity,
            out_row_stride,
            stage_b,
            stage_capacity,
            q_base_addr=q_base_addr,
            k_base_addr=k_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        assert err_a == err_b
        if err_a == ATTN_Q16_OK:
            assert out_a == out_b

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity(
        [0],
        1,
        I64_MAX,
        2,
        [0],
        1,
        2,
        2,
        1,
        65536,
        [0],
        1,
        2,
        [0],
        1,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


if __name__ == "__main__":
    test_source_contains_alias_safe_default_capacity_wrapper()
    test_known_vectors_and_alias_overlap_path()
    test_randomized_parity_and_overflow()
    print("ok")
