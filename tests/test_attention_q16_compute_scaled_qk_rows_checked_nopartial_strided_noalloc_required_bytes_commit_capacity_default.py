#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityDefault."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    I64_MAX,
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_default(
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
    commit_stage_byte_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
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
    if commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity(
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
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )


def explicit_checked_required_bytes_commit_capacity_default_composition(
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
    commit_stage_byte_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
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
    if commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity(
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
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )


def test_source_contains_strided_noalloc_required_bytes_commit_capacity_default_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityDefault("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "status = AttentionTryMulI64Checked(query_row_count," in body
    assert "token_count," in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity("
        in body
    )


def test_known_vectors_and_capacity_rejection() -> None:
    query_row_count = 4
    token_count = 6
    head_dim = 5
    query_row_stride_q16 = 8
    k_row_stride_q16 = 9
    out_row_stride = 7

    q_capacity = query_row_count * query_row_stride_q16
    k_capacity = token_count * k_row_stride_q16
    out_capacity = query_row_count * out_row_stride
    required_stage_cells = query_row_count * token_count
    required_stage_bytes = required_stage_cells * 8
    required_out_cells = (query_row_count - 1) * out_row_stride + token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    out_stage_cells = [111]
    out_stage_bytes = [222]
    out_out_cells = [333]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_default(
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
        required_stage_bytes,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert out_stage_cells == [required_stage_cells]
    assert out_stage_bytes == [required_stage_bytes]
    assert out_out_cells == [required_out_cells]

    out_stage_cells = [9]
    out_stage_bytes = [9]
    out_out_cells = [9]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_default(
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
        required_stage_bytes - 8,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_stage_cells == [9]
    assert out_stage_bytes == [9]
    assert out_out_cells == [9]


def test_error_paths() -> None:
    out_stage_cells = [1]
    out_stage_bytes = [1]
    out_out_cells = [1]

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_default(
            [0],
            1,
            1,
            1,
            [0],
            1,
            1,
            1,
            1,
            [0],
            1,
            1,
            8,
            None,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_default(
            [0],
            -1,
            1,
            1,
            [0],
            1,
            1,
            1,
            1,
            [0],
            1,
            1,
            8,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_default(
        [0],
        I64_MAX,
        huge,
        1,
        [0],
        I64_MAX,
        3,
        1,
        1,
        [0],
        I64_MAX,
        3,
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_checked_composition() -> None:
    rng = random.Random(20260420_636)

    for _ in range(5000):
        query_row_count = rng.randint(0, 40)
        token_count = rng.randint(0, 40)
        head_dim = rng.randint(0, 40)
        query_row_stride_q16 = rng.randint(max(1, head_dim), max(1, head_dim + 24))
        k_row_stride_q16 = rng.randint(max(1, head_dim), max(1, head_dim + 24))
        out_row_stride = rng.randint(max(1, token_count), max(1, token_count + 24))

        q_capacity = max(0, query_row_count * query_row_stride_q16 + rng.randint(-80, 80))
        k_capacity = max(0, token_count * k_row_stride_q16 + rng.randint(-80, 80))
        out_capacity = max(0, query_row_count * out_row_stride + rng.randint(-80, 80))
        commit_stage_byte_capacity = max(
            0, query_row_count * token_count * 8 + rng.randint(-256, 256)
        )

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
        if rng.random() < 0.06:
            commit_stage_byte_capacity = -rng.randint(1, 80)

        if rng.random() < 0.04:
            query_row_count = (1 << 62) + rng.randint(0, 16)
            token_count = (1 << 62) + rng.randint(0, 16)
            commit_stage_byte_capacity = I64_MAX

        q_rows = None if rng.random() < 0.03 else [0] * max(1, q_capacity)
        k_rows = None if rng.random() < 0.03 else [0] * max(1, k_capacity)
        out_scores = None if rng.random() < 0.03 else [0] * max(1, out_capacity)

        got_stage_cells = [0xAAAA]
        got_stage_bytes = [0xBBBB]
        got_out_cells = [0xCCCC]
        exp_stage_cells = [0xAAAA]
        exp_stage_bytes = [0xBBBB]
        exp_out_cells = [0xCCCC]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_default(
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
            commit_stage_byte_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_exp = explicit_checked_required_bytes_commit_capacity_default_composition(
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
            commit_stage_byte_capacity,
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


if __name__ == "__main__":
    test_source_contains_strided_noalloc_required_bytes_commit_capacity_default_helper()
    test_known_vectors_and_capacity_rejection()
    test_error_paths()
    test_randomized_parity_against_explicit_checked_composition()
    print("ok")
